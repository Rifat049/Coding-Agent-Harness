#!/usr/bin/env bash
# 5-minute CodeBenchKit demo (see docs/demo.md).
#   scripts/demo.sh          replay saved runs: no network, no key, no cost (about 1 minute)
#   scripts/demo.sh --live   also run qwen2.5-coder:7b live on 2 tasks via local Ollama (free, a few minutes)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
LIVE=0
[[ "${1:-}" == "--live" ]] && LIVE=1
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

step "1. The machine is ready (no remote calls)"
$PY -m codebenchkit doctor --offline || true

step "2. The checks are trustworthy: every task is fully specified, references pass, starting code fails"
$PY -m codebenchkit utility-selfcheck

step "3. One task: the spec the model gets (rules R1..Rn) - u01_parse_duration"
awk '/^## Tests/{exit} /^## Specification/{f=1} f' benchmarks/utility_v1/tasks/u01_parse_duration/workspace/README.md | head -30

step "4. Every metric fires: offline mocks (good, lazy = false success, probe = reads hidden tests, garble = invalid output)"
OUT=$(mktemp -d)
for v in mock mock:lazy mock:probe mock:garble; do
  $PY -m codebenchkit utility-bench --model "$v" --tasks u03,u05 --out "$OUT" --name "${v/:/_}" >/dev/null
  r="$OUT/${v/:/_}/report.md"
  val() { grep -E "^\| \**$1" "$r" | head -1 | awk -F'|' '{print $3}' | sed 's/\*\*//g; s/%.*/%/; s/^ *//'; }
  printf '%-12s pass %-7s false success %-7s cheat %-7s invalid output per turn %s\n' "$v" \
    "$(val 'Pass rate')" "$(val 'False-success rate')" "$(val 'Cheat rate')" "$(val 'Invalid-output rate\**, per turn')"
done

step "5. Saved results: every arm on utility-v1 (Qwen2.5-Coder 7B/14B, DeepSeek), 10 tasks x 5 samples, fixed budget"
sed -n '1,/^## Failure categories/p' results/utility/arms.md | sed '$d'

step "6. Stretch, saved: paired comparisons at a fixed budget"
for f in results/utility/compare_*.md; do
  [[ -f "$f" ]] || continue
  echo "--- $f"; sed -n '1,/## Head-to-head/p' "$f" | head -25; grep -E '^(Both pass|B minus A|Verdict)' "$f"
done

if [[ $LIVE == 1 ]]; then
  step "7. Live: qwen2.5-coder:7b via local Ollama on 2 tasks (free)"
  $PY -m codebenchkit doctor --model qwen-coder --offline
  $PY -m codebenchkit utility-bench --model qwen-coder --tasks u01,u10 --samples 1 --out "$OUT" --name live
fi
step "Done. Full write-up: docs/report.md"
