import unittest

from tests.hooks._load import load_hook

cm = load_hook("commit_msg_check")


class SubjectExtractionTests(unittest.TestCase):
    def test_skips_blank_and_comment_lines(self):
        message = "\n# Please enter the commit message\nfeat: add x\n"
        self.assertEqual(cm.subject_of(message), "feat: add x")

    def test_empty_message_yields_empty_subject(self):
        self.assertEqual(cm.subject_of("\n\n# only comments\n"), "")


class CheckSubjectTests(unittest.TestCase):
    def test_accepts_conventional_with_scope(self):
        self.assertEqual(cm.check_subject("feat(mind): add retriever"), [])

    def test_accepts_conventional_without_scope(self):
        self.assertEqual(cm.check_subject("chore: bump deps"), [])

    def test_rejects_non_conventional_subject(self):
        errors = cm.check_subject("added a thing")
        self.assertTrue(any("Conventional Commit" in e for e in errors))

    def test_rejects_cjk_subject(self):
        errors = cm.check_subject("feat: 增加功能")
        self.assertTrue(any("CJK" in e for e in errors))

    def test_rejects_unknown_type(self):
        self.assertTrue(cm.check_subject("wip(core): halfway"))

    def test_exempts_merge_and_revert(self):
        self.assertEqual(cm.check_subject("Merge branch 'main' into feat"), [])
        self.assertEqual(cm.check_subject('Revert "feat: x"'), [])

    def test_empty_subject_is_not_flagged(self):
        self.assertEqual(cm.check_subject(""), [])


if __name__ == "__main__":
    unittest.main()
