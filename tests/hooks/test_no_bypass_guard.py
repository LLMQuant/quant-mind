import unittest

from tests.hooks._load import load_hook

guard = load_hook("pre_tool_use_no_bypass")


class EvaluateTests(unittest.TestCase):
    def _command(self, command: str):
        return guard.evaluate(
            {"tool_name": "Bash", "tool_input": {"command": command}}
        )

    def test_denies_commit_no_verify(self):
        decision = self._command("git commit --no-verify -m 'feat: x'")
        self.assertIsNotNone(decision)
        self.assertEqual(
            decision["hookSpecificOutput"]["permissionDecision"], "deny"
        )

    def test_denies_push_no_verify(self):
        self.assertIsNotNone(self._command("git push --no-verify origin main"))

    def test_allows_normal_commit(self):
        self.assertIsNone(self._command("git commit -m 'feat: x'"))

    def test_allows_no_verify_without_git(self):
        self.assertIsNone(self._command("some-linter --no-verify src/"))

    def test_ignores_non_bash_tools(self):
        self.assertIsNone(
            guard.evaluate({"tool_name": "Edit", "tool_input": {}})
        )

    def test_ignores_missing_command(self):
        self.assertIsNone(guard.evaluate({"tool_name": "Bash"}))


if __name__ == "__main__":
    unittest.main()
