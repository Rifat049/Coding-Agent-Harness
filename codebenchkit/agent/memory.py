"""Long-term memory for a coding agent: episodic, semantic, procedural. Plain text, no vectors.

    episodic    one row per agent run: task, outcome, files, errors, notes.   (what happened)
    semantic    durable facts: conventions, pitfalls, commands, preferences.  (what is true)
    procedural  SKILL.md files: how to do a recurring kind of task.           (how to act)

Retrieval is SQLite FTS5 with BM25: ranked keyword search, fully local, inspectable
with `sqlite3`. Following waku-agent's "keyword top-k, no embedding" choice; for
code it also matters that identifiers like `parse_duration` or `ECONNRESET` match
exactly rather than "semantically nearby". The BM25 score is then blended with
scope (this repo > global), support (how many episodes confirmed a fact), and
recency, and the result is cut to a character budget.

Only consolidation (consolidate.py, a cheap model) writes semantic facts from
episodes. The run itself writes episodes only, so one bad run cannot poison
memory directly.
"""
from __future__ import annotations

import contextlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

# Bump when the index-time tokenizer or FTS layout changes; the index is rebuilt once on open.
INDEX_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY,
    subject TEXT NOT NULL,
    content TEXT NOT NULL,
    kind TEXT NOT NULL DEFAULT 'convention',     -- convention|pitfall|command|preference|architecture
    scope TEXT NOT NULL DEFAULT 'global',        -- global | repo:<name>
    support INTEGER NOT NULL DEFAULT 1,          -- episodes that confirmed it
    evidence TEXT NOT NULL DEFAULT '[]',         -- json list of episode ids
    source TEXT NOT NULL DEFAULT 'consolidation',
    active INTEGER NOT NULL DEFAULT 1,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    last_used_at REAL
);
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(subject, content, kind, content='facts', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
  INSERT INTO facts_fts(rowid, subject, content, kind) VALUES (new.id, new.subject, new.content, new.kind);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, subject, content, kind) VALUES ('delete', old.id, old.subject, old.content, old.kind);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE OF subject, content, kind ON facts BEGIN
  INSERT INTO facts_fts(facts_fts, rowid, subject, content, kind) VALUES ('delete', old.id, old.subject, old.content, old.kind);
  INSERT INTO facts_fts(rowid, subject, content, kind) VALUES (new.id, new.subject, new.content, new.kind);
END;

CREATE TABLE IF NOT EXISTS episodes (
    id INTEGER PRIMARY KEY,
    created_at REAL NOT NULL,
    scope TEXT NOT NULL,
    task TEXT NOT NULL,
    outcome TEXT NOT NULL,                       -- success | failure | partial | aborted
    summary TEXT NOT NULL DEFAULT '',
    files TEXT NOT NULL DEFAULT '[]',
    errors TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '',
    stats TEXT NOT NULL DEFAULT '{}',
    consolidated INTEGER NOT NULL DEFAULT 0
);
CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(task, summary, errors, notes, content='episodes', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
  INSERT INTO episodes_fts(rowid, task, summary, errors, notes) VALUES (new.id, new.task, new.summary, new.errors, new.notes);
END;
CREATE TRIGGER IF NOT EXISTS episodes_ad AFTER DELETE ON episodes BEGIN
  INSERT INTO episodes_fts(episodes_fts, rowid, task, summary, errors, notes) VALUES ('delete', old.id, old.task, old.summary, old.errors, old.notes);
END;
CREATE TRIGGER IF NOT EXISTS episodes_au AFTER UPDATE OF task, summary, errors, notes ON episodes BEGIN
  INSERT INTO episodes_fts(episodes_fts, rowid, task, summary, errors, notes) VALUES ('delete', old.id, old.task, old.summary, old.errors, old.notes);
  INSERT INTO episodes_fts(rowid, task, summary, errors, notes) VALUES (new.id, new.task, new.summary, new.errors, new.notes);
END;

CREATE TABLE IF NOT EXISTS consolidations (
    id INTEGER PRIMARY KEY,
    at REAL NOT NULL,
    model TEXT NOT NULL,
    episode_ids TEXT NOT NULL,
    applied TEXT NOT NULL,
    rejected TEXT NOT NULL,
    latency_s REAL,                              -- small-model wall time, all attempts
    attempts INTEGER                             -- 2 = the JSON retry was needed
);

-- Repo -> global promotions are proposals: consolidation records them, `memory promote` applies.
CREATE TABLE IF NOT EXISTS promotions (
    id INTEGER PRIMARY KEY,
    fact_id INTEGER NOT NULL,
    from_scope TEXT NOT NULL,
    to_scope TEXT NOT NULL DEFAULT 'global',
    repos TEXT NOT NULL,                         -- json list of the 2+ repos in the evidence
    evidence TEXT NOT NULL,                      -- json list of the confirming episode ids
    proposed_at REAL NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',      -- pending | applied
    applied_at REAL
);
"""

STOPWORDS = set("""a an the and or but if then else of to in on at for with by from as is are was were be been
being it its this that these those i you we they he she my our your their me us them do does did done
can could should would will shall may might must not no yes so than too very just also into out up down
over under again more most some any each all both few other such only own same about please make add
fix use using get set new code file files function implement write create update change""".split())

FACT_KINDS = ("convention", "pitfall", "command", "preference", "architecture")


def _tokens(text: str) -> list[str]:
    """Keywords for FTS: words of 2+ chars, plus camelCase and snake_case pieces, minus stopwords."""
    out = []
    for w in re.findall(r"\w{2,}", text or ""):
        out.append(w)
        for part in (p for p in w.split("_") if p):
            if part != w:
                out.append(part)
            camel = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+", part)
            if len(camel) > 1:
                out.extend(camel)
    seen = []
    for w in (x.lower() for x in out):
        if len(w) >= 2 and w not in STOPWORDS and w not in seen:
            seen.append(w)
    return seen[:40]


def fts_query(text: str) -> str:
    toks = _tokens(text)
    # Quote each token: FTS5 operators (AND/OR/NEAR) and punctuation become plain terms.
    return " OR ".join(f'"{t}"' for t in toks)


@dataclass
class Retrieved:
    facts: list[dict]
    episodes: list[dict]
    skills: list[dict]
    query: str = ""
    gated_out: bool = False

    def empty(self) -> bool:
        return not (self.facts or self.episodes or self.skills)


class Memory:
    def __init__(self, path: str | Path, skills_dir: str | Path | None = None,
                 fact_budget_chars: int = 2500, episode_budget_chars: int = 1500, top_facts: int = 8,
                 top_episodes: int = 3):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.create_function("cbk_tokens", 1, lambda text: " ".join(_tokens(text)))
        # Recreate draft triggers so old databases get the same identifier tokenizer.
        schema = SCHEMA
        for table, columns in (("facts", ("subject", "content", "kind")),
                               ("episodes", ("task", "summary", "errors", "notes"))):
            for suffix in ("ai", "ad", "au"):
                self.conn.execute(f"DROP TRIGGER IF EXISTS {table}_{suffix}")
            for prefix in ("new", "old"):
                for column in columns:
                    schema = schema.replace(f"{prefix}.{column}", f"cbk_tokens({prefix}.{column})")
        self.conn.executescript(schema)
        cols = {r[1] for r in self.conn.execute("PRAGMA table_info(consolidations)")}
        for col, typ in (("latency_s", "REAL"), ("attempts", "INTEGER")):
            if col not in cols:                  # databases created before these columns
                self.conn.execute(f"ALTER TABLE consolidations ADD COLUMN {col} {typ}")
        # External-content FTS 'rebuild' does not run the tokenizer SQL function, so re-index by
        # hand, but only when the index predates the current tokenizer (not on every open).
        if self.conn.execute("PRAGMA user_version").fetchone()[0] < INDEX_VERSION:
            for table, columns in (("facts", ("subject", "content", "kind")),
                                   ("episodes", ("task", "summary", "errors", "notes"))):
                self.conn.execute(f"INSERT INTO {table}_fts({table}_fts) VALUES ('delete-all')")
                names = ','.join(columns)
                values = ','.join(f"cbk_tokens({c})" for c in columns)
                self.conn.execute(f"INSERT INTO {table}_fts(rowid,{names}) SELECT id,{values} FROM {table}")
            self.conn.execute(f"PRAGMA user_version = {INDEX_VERSION}")
        self.conn.commit()
        self._tx_depth = 0
        self.skills = SkillLibrary(Path(skills_dir) if skills_dir else self.path.parent / "skills")
        self.fact_budget = fact_budget_chars
        self.episode_budget = episode_budget_chars
        self.top_facts = top_facts
        self.top_episodes = top_episodes

    # ------------------------------------------------------------------ transactions
    @contextlib.contextmanager
    def transaction(self):
        """Group writes: inside this block the CRUD helpers do not commit, and any exception
        rolls everything back. Consolidation relies on this to be all-or-nothing."""
        if self._tx_depth == 0 and self.conn.in_transaction:
            self.conn.commit()
        self._tx_depth += 1
        try:
            yield self
        except BaseException:
            self._tx_depth -= 1
            if self._tx_depth == 0:
                self.conn.rollback()
            raise
        self._tx_depth -= 1
        if self._tx_depth == 0:
            self.conn.commit()

    def _commit(self) -> None:
        if self._tx_depth == 0:
            self.conn.commit()

    # ------------------------------------------------------------------ writes
    def close(self) -> None:
        """Release the SQLite connection; closing twice is harmless."""
        self.conn.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def add_fact(self, subject: str, content: str, kind: str = "convention", scope: str = "global",
                 evidence: list[int] | None = None, source: str = "user") -> int:
        now = time.time()
        cur = self.conn.execute(
            "INSERT INTO facts (subject, content, kind, scope, evidence, source, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (subject.strip()[:80], content.strip(), kind if kind in FACT_KINDS else "convention",
             scope, json.dumps(evidence or []), source, now, now))
        self._commit()
        return cur.lastrowid

    def update_fact(self, fid: int, content: str | None = None, add_evidence: list[int] | None = None,
                    bump_support: bool = False, kind: str | None = None) -> bool:
        row = self.conn.execute("SELECT * FROM facts WHERE id=? AND active=1", (fid,)).fetchone()
        if not row:
            return False
        ev = sorted(set(json.loads(row["evidence"]) + list(add_evidence or [])))
        self.conn.execute(
            "UPDATE facts SET content=?, kind=?, evidence=?, support=support+?, updated_at=? WHERE id=?",
            (content.strip() if content else row["content"], kind if kind in FACT_KINDS else row["kind"],
             json.dumps(ev), 1 if bump_support else 0, time.time(), fid))
        self._commit()
        return True

    def confirm_fact(self, fid: int, evidence: list[int] | None = None) -> bool:
        """Another episode agrees with this fact: support + 1, evidence extended."""
        return self.update_fact(fid, add_evidence=evidence, bump_support=True)

    def propose_promotion(self, fid: int, from_scope: str, repos: list[str], evidence: list[int]) -> int:
        """Record (or refresh) the pending repo -> global proposal for a fact. Returns its id."""
        row = self.conn.execute("SELECT id, evidence FROM promotions WHERE fact_id=? AND status='pending'",
                                (fid,)).fetchone()
        if row:
            ev = sorted(set(json.loads(row["evidence"]) + list(evidence)))
            self.conn.execute("UPDATE promotions SET repos=?, evidence=?, proposed_at=? WHERE id=?",
                              (json.dumps(sorted(repos)), json.dumps(ev), time.time(), row["id"]))
            self._commit()
            return row["id"]
        cur = self.conn.execute(
            "INSERT INTO promotions (fact_id, from_scope, repos, evidence, proposed_at) VALUES (?,?,?,?,?)",
            (fid, from_scope, json.dumps(sorted(repos)), json.dumps(sorted(set(evidence))), time.time()))
        self._commit()
        return cur.lastrowid

    def pending_promotions(self) -> list[dict]:
        q = ("SELECT p.*, f.subject, f.content FROM promotions p JOIN facts f ON f.id = p.fact_id "
             "WHERE p.status='pending' AND f.active=1 AND f.scope = p.from_scope ORDER BY p.id")
        return [dict(r) for r in self.conn.execute(q)]

    def promote_fact(self, fid: int) -> dict | None:
        """Apply the pending promotion for a fact: scope -> global. None if there is none."""
        pending = [p for p in self.pending_promotions() if p["fact_id"] == fid]
        if not pending:
            return None
        p = pending[0]
        with self.transaction():
            self.set_fact_scope(fid, p["to_scope"])
            self.conn.execute("UPDATE promotions SET status='applied', applied_at=? WHERE id=?", (time.time(), p["id"]))
        return {**p, "status": "applied"}

    def set_fact_scope(self, fid: int, scope: str) -> bool:
        cur = self.conn.execute("UPDATE facts SET scope=?, updated_at=? WHERE id=? AND active=1",
                                (scope, time.time(), fid))
        self._commit()
        return cur.rowcount > 0

    def episode_scopes(self, ids) -> dict[int, str]:
        ids = [i for i in ids if isinstance(i, int)]
        if not ids:
            return {}
        q = f"SELECT id, scope FROM episodes WHERE id IN ({','.join('?' * len(ids))})"
        return {r["id"]: r["scope"] for r in self.conn.execute(q, ids)}

    def get_fact(self, fid: int) -> dict | None:
        row = self.conn.execute("SELECT * FROM facts WHERE id=?", (fid,)).fetchone()
        return dict(row) if row else None

    def forget_fact(self, fid: int) -> bool:
        cur = self.conn.execute("UPDATE facts SET active=0, updated_at=? WHERE id=? AND active=1", (time.time(), fid))
        self._commit()
        return cur.rowcount > 0

    def add_episode(self, scope: str, task: str, outcome: str, summary: str = "", files=None,
                    errors: str = "", notes: list[str] | None = None, stats: dict | None = None) -> int:
        cur = self.conn.execute(
            "INSERT INTO episodes (created_at, scope, task, outcome, summary, files, errors, notes, stats) "
            "VALUES (?,?,?,?,?,?,?,?,?)",
            (time.time(), scope, task[:2000], outcome, summary[:2000], json.dumps(sorted(files or [])[:50]),
             errors[-2000:], "\n".join(notes or [])[:2000], json.dumps(stats or {}, default=str)))
        self._commit()
        return cur.lastrowid

    def mark_consolidated(self, ids: list[int]) -> None:
        self.conn.executemany("UPDATE episodes SET consolidated=1 WHERE id=?", [(i,) for i in ids])
        self._commit()

    # ------------------------------------------------------------------ reads
    def facts(self, scope: str | None = None, active: bool = True, limit: int = 500) -> list[dict]:
        q = "SELECT * FROM facts WHERE active=?" + (" AND scope=?" if scope else "") + " ORDER BY id DESC LIMIT ?"
        args = (int(active), scope, limit) if scope else (int(active), limit)
        return [dict(r) for r in self.conn.execute(q, args)]

    def episodes(self, unconsolidated_only: bool = False, limit: int = 500, newest_first: bool = False) -> list[dict]:
        q = ("SELECT * FROM episodes" + (" WHERE consolidated=0" if unconsolidated_only else "")
             + (" ORDER BY id DESC" if newest_first else " ORDER BY id") + " LIMIT ?")
        return [dict(r) for r in self.conn.execute(q, (limit,))]

    def search_facts(self, query: str, scope: str = "global", limit: int | None = None,
                     any_scope: bool = False) -> list[dict]:
        """FTS over active facts in `scope` and global (every scope with any_scope)."""
        limit = self.top_facts if limit is None else max(0, limit)
        if not limit:
            return []
        fq = fts_query(query)
        if not fq:
            return []
        rows = self.conn.execute(
            "SELECT f.*, bm25(facts_fts, 2.0, 1.0, 0.5) AS score FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid "
            "WHERE facts_fts MATCH ? AND f.active = 1 AND (f.scope = 'global' OR f.scope = ? OR ?) "
            "ORDER BY score LIMIT 40", (fq, scope, int(any_scope))).fetchall()
        return rerank_facts([dict(r) for r in rows], scope)[:limit]

    def search_episodes(self, query: str, scope: str = "global", limit: int | None = None) -> list[dict]:
        limit = self.top_episodes if limit is None else max(0, limit)
        if not limit:
            return []
        fq = fts_query(query)
        if not fq:
            return []
        rows = self.conn.execute(
            "SELECT e.*, bm25(episodes_fts, 2.0, 1.0, 1.0, 1.5) AS score FROM episodes_fts "
            "JOIN episodes e ON e.id = episodes_fts.rowid WHERE episodes_fts MATCH ? "
            "AND (e.scope = ? OR e.scope = 'global') ORDER BY score LIMIT 20", (fq, scope)).fetchall()
        now = time.time()
        ranked = [dict(r) for r in rows]
        top = max((-d["score"] for d in ranked), default=1.0) or 1.0
        for d in ranked:
            d["rank"] = (-d["score"] / top + (0.3 if d["scope"] == scope and scope != "global" else 0.0)
                         + 0.3 * math.exp(-(now - d["created_at"]) / 86400 / 30))
        ranked.sort(key=lambda d: (-d["rank"], -d["id"]))
        return ranked[:limit]

    def retrieve(self, query: str, scope: str = "global") -> Retrieved:
        # Charge exactly what render emits, including headers, tags and separators.
        facts = _budget(self.search_facts(query, scope), lambda f: len(_fact_line(f)) + 1,
                        self.fact_budget - len(FACT_HEADER) - 1)
        eps = _budget(self.search_episodes(query, scope), lambda e: len(_episode_line(e)) + 1,
                      self.episode_budget - len(EPISODE_HEADER) - 1)
        if facts:
            ids = [f["id"] for f in facts]
            self.conn.execute(f"UPDATE facts SET last_used_at=? WHERE id IN ({','.join('?' * len(ids))})",
                              [time.time(), *ids])
            self._commit()
        return Retrieved(facts, eps, self.skills.match(query), query)

    @staticmethod
    def render(r: Retrieved, empty: str = "") -> str:
        parts = []
        if r.facts:
            parts.append(FACT_HEADER)
            parts += [_fact_line(f) for f in r.facts]
        if r.episodes:
            parts.append(EPISODE_HEADER)
            parts += [_episode_line(e) for e in r.episodes]
        return "\n".join(parts) if parts else empty

    def stats(self) -> dict:
        one = lambda q: self.conn.execute(q).fetchone()[0]  # noqa: E731
        return {"facts": one("SELECT COUNT(*) FROM facts WHERE active=1"),
                "episodes": one("SELECT COUNT(*) FROM episodes"),
                "unconsolidated": one("SELECT COUNT(*) FROM episodes WHERE consolidated=0"),
                "skills": len(self.skills.all()), "consolidations": one("SELECT COUNT(*) FROM consolidations"),
                "pending_promotions": one("SELECT COUNT(*) FROM promotions WHERE status='pending'")}

    def export_markdown(self) -> str:
        lines = ["# Agent memory", ""]
        for scope in sorted({f["scope"] for f in self.facts()}):
            lines += [f"## {scope}", ""]
            for f in self.facts(scope):
                lines.append(f"- [F{f['id']}] **{f['kind']}** ({f['subject']}, support {f['support']}): {f['content']}")
            lines.append("")
        lines += ["## Recent episodes", ""]
        for e in self.episodes(limit=20, newest_first=True):
            lines.append(f"- [E{e['id']}] {e['outcome']}: {e['task'][:120]}")
        return "\n".join(lines) + "\n"


# Rerank weights. BM25 is normalised to [0, 1] within the result set first: raw FTS5 BM25 values
# shrink toward zero when a term appears in most rows, so unnormalised they would let the
# boosts swamp relevance on a small store, or vanish on a large one.
W_BM25, W_SAME_REPO, W_SUPPORT, W_RECENCY, W_PITFALL = 1.0, 0.35, 0.15, 0.2, 0.1
RECENCY_DAYS = 90


def rerank_facts(rows: list[dict], scope: str, now: float | None = None) -> list[dict]:
    """BM25 + same-repo boost + log(support) + recency decay + a small pitfall boost."""
    now = time.time() if now is None else now
    top = max((-r["score"] for r in rows), default=1.0) or 1.0
    for d in rows:
        rel = max(0.0, -d["score"]) / top
        same_repo = 1.0 if d["scope"] == scope and scope != "global" else 0.0
        support = math.log1p(max(0, d["support"] - 1))
        recency = math.exp(-max(0.0, now - d["updated_at"]) / 86400 / RECENCY_DAYS)
        pitfall = 1.0 if d["kind"] == "pitfall" else 0.0
        d["rank"] = (W_BM25 * rel + W_SAME_REPO * same_repo + W_SUPPORT * support
                     + W_RECENCY * recency + W_PITFALL * pitfall)
    rows.sort(key=lambda d: (-d["rank"], -d["id"]))
    return rows


FACT_HEADER = "Known facts (from past work; verify before relying on them):"
EPISODE_HEADER = "Similar past runs (verify before relying on them):"


def _fact_line(f: dict) -> str:
    return f"- [F{f['id']}] ({f['kind']}) {f['content']}"


def _episode_line(e: dict) -> str:
    date = time.strftime("%Y-%m-%d", time.localtime(e["created_at"]))
    summary = e["summary"] or e["notes"] or ""
    return (f"- [E{e['id']}] {date} {e['outcome'].upper()}: {e['task'][:140]}"
            + (f" -> {summary[:240]}" if summary else ""))


def _budget(items: list[dict], size, budget: int) -> list[dict]:
    out, used = [], 0
    for it in items:
        s = size(it)
        if used + s > budget:
            continue
        out.append(it)
        used += s
    return out


# ---------------------------------------------------------------------- retrieval gate
GATE_PROMPT = """You decide whether a coding agent should load its long-term memory before a task.
Memory holds project conventions, known pitfalls, useful commands, and summaries of past runs.
Reply with ONLY JSON: {{"retrieve": true|false, "query": "<keywords to search with>"}}
Retrieve for anything touching this project's code, tests, tooling, or past work. Skip for
self-contained questions (general knowledge, arithmetic, small talk).

Task: {task}"""


def gate(task: str, mode: str = "heuristic", small=None) -> tuple[bool, str]:
    """Should this turn load memory, and with what query? Fails open (retrieve) on any error."""
    if mode == "always":
        return True, task
    if mode == "never":
        return False, ""
    if mode == "llm":
        try:
            from .llm import parse_json_object
            obj = parse_json_object(small.complete(GATE_PROMPT.format(task=task[:2000]), max_tokens=600))
            if (isinstance(obj, dict) and type(obj.get("retrieve")) is bool
                    and isinstance(obj.get("query"), str)):
                return obj["retrieve"], obj["query"].strip() or task
        except Exception as exc:  # Provider failures and malformed output must fail open...
            from .llm import FatalProviderError
            if isinstance(exc, FatalProviderError):
                raise               # ...except billing/auth, which stops the run

        return True, task
    # heuristic: a coding task almost always benefits; skip only content-free messages
    return (len(_tokens(task)) >= 2), task


# ---------------------------------------------------------------------- procedural memory
class SkillLibrary:
    """SKILL.md files with `name` and `description` frontmatter (Anthropic Agent Skills shape).
    Progressive disclosure: descriptions are always scanned; a body is injected only when its
    description matches the task. Consolidation proposes skills into skills/_proposed/ and a
    human promotes them; nothing proposed is ever loaded automatically."""

    def __init__(self, root: Path, max_chars: int = 3000):
        self.root = root
        self.max_chars = max_chars

    def all(self) -> list[dict]:
        out = []
        if not self.root.exists():
            return out
        for p in sorted(self.root.glob("*/SKILL.md")):
            if not self._safe(p):
                continue
            try:
                # Only metadata is read during discovery, never unmatched bodies.
                with p.open(encoding="utf-8") as stream:
                    if stream.readline().strip() != "---":
                        continue
                    lines = []
                    for _ in range(100):
                        line = stream.readline(4096)
                        if line.strip() == "---":
                            break
                        if not line:
                            break
                        lines.append(line)
                    else:
                        continue
            except (OSError, UnicodeError):
                continue
            front = dict((k.strip(), v.strip().strip("'\"")) for k, _, v in
                         (ln.partition(":") for ln in lines if ":" in ln))
            if "name" in front and "description" in front:
                out.append({"name": front["name"], "description": front["description"],
                            "path": str(p)})
        return out

    def _safe(self, path: Path) -> bool:
        """Reject symlinks, including aliases into quarantined proposals."""
        try:
            relative = path.relative_to(self.root)
            return (not self.root.is_symlink()
                    and not any(p.startswith("_") for p in relative.parts)
                    and not any(self.root.joinpath(*relative.parts[:i]).is_symlink()
                                for i in range(1, len(relative.parts) + 1)))
        except ValueError:
            return False

    def match(self, task: str, limit: int = 2) -> list[dict]:
        if limit <= 0 or self.max_chars <= 0:
            return []
        toks = set(_tokens(task))
        scored = []
        for s in self.all():
            overlap = len(toks & set(_tokens(s["name"] + " " + s["description"])))
            if overlap >= 2 or s["name"].lower() in task.lower():
                scored.append((overlap, s))
        scored.sort(key=lambda t: -t[0])
        result, remaining = [], self.max_chars
        for _, s in scored[:limit]:
            path = Path(s["path"])
            if not self._safe(path):
                continue
            try:
                with path.open(encoding="utf-8") as stream:
                    stream.readline()
                    for _ in range(100):
                        if stream.readline(4096).strip() == "---":
                            break
                    body = stream.read(remaining).strip()
            except (OSError, UnicodeError):
                continue
            if body:
                result.append({**s, "body": body})
                remaining -= len(body)
            if remaining <= 0:
                break
        return result

    def propose(self, name: str, description: str, body: str) -> Path:
        slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-")[:50] or "skill"
        d = self.root / "_proposed" / slug
        d.mkdir(parents=True, exist_ok=True)
        p = d / "SKILL.md"
        p.write_text(f"---\nname: {slug}\ndescription: {description.strip()}\n---\n\n{body.strip()}\n", encoding="utf-8")
        return p
