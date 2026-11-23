"""doctor: never prints key values; red only for what the configured run actually needs."""
import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from codebenchkit.cli import main
from codebenchkit.doctor import FAIL, OK, WARN, run

SECRET = "sk-THIS-MUST-NEVER-BE-PRINTED-1234567890"


class DoctorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        # Offline: the DeepSeek balance check is tested on its own (DeepSeekBalanceTests).
        from codebenchkit.doctor import Check
        stub = patch("codebenchkit.doctor._deepseek_balance", return_value=Check("DeepSeek balance", OK, "stubbed"))
        stub.start()
        self.addCleanup(stub.stop)

    def env(self, **extra):
        base = {"PATH": os.environ.get("PATH", ""), "HOME": self.tmp.name, "CBK_HOME": str(self.repo / "home"),
                "QWEN_BASE_URL": "http://127.0.0.1:9/v1"}
        return patch.dict(os.environ, {**base, **extra}, clear=True)

    def output(self, *args):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(["doctor", "--repo", str(self.repo), *args])
        return code, buf.getvalue()

    def test_key_values_are_never_printed(self):
        with self.env(DEEPSEEK_API_KEY=SECRET, ANTHROPIC_API_KEY=SECRET, DASHSCOPE_API_KEY=SECRET):
            code, out = self.output()
        self.assertNotIn(SECRET, out)
        self.assertNotIn(SECRET[:12], out)
        self.assertIn("DEEPSEEK_API_KEY", out)
        self.assertIn("ignores it on purpose", out)
        self.assertEqual(code, 0)

    def test_missing_main_key_is_red_with_a_fix(self):
        with self.env():
            checks = {c.name: c for c in run(self.repo)}
            code, out = self.output()
        self.assertEqual(checks["DEEPSEEK_API_KEY"].status, FAIL)
        self.assertIn("export DEEPSEEK_API_KEY", checks["DEEPSEEK_API_KEY"].fix)
        self.assertEqual(checks["OPENAI_API_KEY"].status, WARN)      # not needed for this config
        self.assertEqual(code, 1)

    def test_missing_main_key_is_only_a_warning_offline(self):
        # A fresh clone with no key: offline checks, mocks and saved results need none.
        with self.env():
            checks = {c.name: c for c in run(self.repo, offline=True)}
            code, out = self.output("--offline")
        self.assertEqual(checks["DEEPSEEK_API_KEY"].status, WARN)
        self.assertIn("live runs", checks["DEEPSEEK_API_KEY"].detail)
        self.assertEqual(code, 0)

    def test_no_small_model_under_auto_is_a_warning(self):
        with self.env(DEEPSEEK_API_KEY="k"):
            with patch("codebenchkit.agent.cli.small_unavailable", lambda spec: f"{spec} down"):
                checks = {c.name: c for c in run(self.repo)}
        self.assertEqual(checks["Small model"].status, WARN)
        self.assertIn("auto -> none", checks["Small model"].detail)
        self.assertEqual(checks["Ollama"].status, WARN)
        self.assertIn("ollama.com/download", checks["Ollama"].fix)

    def test_bad_config_is_red(self):
        (self.repo / ".cbk").mkdir()
        (self.repo / ".cbk" / "config.toml").write_text("[agent]\napproval = 'yolo'\n")
        with self.env(DEEPSEEK_API_KEY="k"):
            checks = {c.name: c for c in run(self.repo)}
        self.assertEqual(checks["Config"].status, FAIL)

    def test_core_checks_pass_here(self):
        with self.env(DEEPSEEK_API_KEY="k"):
            checks = {c.name: c for c in run(self.repo)}
        for name in ("Python", "SQLite FTS5", "Benchmark data", "Memory dir"):
            self.assertEqual(checks[name].status, OK, checks[name].detail)


if __name__ == "__main__":
    unittest.main()


class DeepSeekBalanceTests(unittest.TestCase):
    """doctor reads GET /user/balance (mocked here) and never prints the key."""

    def _run(self, payload=None, error=None, expected=0.5):
        import io
        import urllib.error
        from unittest.mock import patch as _patch
        from codebenchkit import doctor

        class Resp(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        seen = {}

        def fake(req, timeout=0):
            seen["url"], seen["auth"] = req.full_url, req.headers.get("Authorization")
            if error:
                raise urllib.error.HTTPError(req.full_url, error, "x", {}, io.BytesIO(b"{}"))
            return Resp(__import__("json").dumps(payload).encode())
        with _patch.dict(os.environ, {"DEEPSEEK_API_KEY": "sk-secret-value"}), \
                _patch("codebenchkit.doctor.urllib.request.urlopen", side_effect=fake):
            c = doctor._deepseek_balance(expected)
        self.assertNotIn("sk-secret-value", c.detail + c.fix)
        self.assertTrue(seen["url"].endswith("/user/balance"))
        self.assertEqual(seen["auth"], "Bearer sk-secret-value")
        return c

    def test_enough_balance_ok(self):
        c = self._run({"is_available": True, "balance_infos": [{"currency": "USD", "total_balance": "3.00"}]})
        self.assertEqual(c.status, "ok")
        self.assertIn("3.00 USD", c.detail)

    def test_cny_balance_below_expected_cost_warns(self):
        c = self._run({"is_available": True, "balance_infos": [{"currency": "CNY", "total_balance": "2.00"}]}, expected=0.5)
        self.assertEqual(c.status, "warn")
        self.assertIn("below the next run's expected cost $0.50", c.detail)
        self.assertIn("CNY/USD", c.detail)

    def test_unavailable_account_fails(self):
        c = self._run({"is_available": False, "balance_infos": [{"currency": "USD", "total_balance": "0.00"}]})
        self.assertEqual(c.status, "fail")
        self.assertIn("top_up", c.fix)

    def test_wrong_key_fails(self):
        c = self._run(error=401)
        self.assertEqual(c.status, "fail")
        self.assertIn("401", c.detail)
