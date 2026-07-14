import re
import unittest
from pathlib import Path


class TestContextEntryPoints(unittest.TestCase):
    def test_context_index_targets_exist(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        context_indexes = (
            "contexts/README.md",
            "contexts/dev/README.md",
            "contexts/usage/README.md",
        )
        markdown_link = re.compile(r"\[[^]]+]\(([^)]+)\)")

        for index_path in context_indexes:
            with self.subTest(index=index_path):
                index = repo_root / index_path
                self.assertTrue(
                    index.is_file(), f"missing context index: {index_path}"
                )

            for target in markdown_link.findall(
                index.read_text(encoding="utf-8")
            ):
                path = target.split("#", maxsplit=1)[0]
                if not path or path.startswith(("http://", "https://")):
                    continue
                resolved = (index.parent / path).resolve()
                with self.subTest(index=index_path, target=target):
                    self.assertTrue(
                        resolved.exists(),
                        f"missing context target from {index_path}: {target}",
                    )

    def test_required_files_link_to_context_entry_point(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        required_links = {
            "AGENTS.md": "contexts/README.md",
            "CLAUDE.md": "contexts/README.md",
            ".agents/skills/quantmind-dev/SKILL.md": "../../../contexts/README.md",
            ".claude/skills/quantmind-dev/SKILL.md": "../../../contexts/README.md",
        }

        for source_path, target in required_links.items():
            source = repo_root / source_path
            with self.subTest(source=source_path):
                self.assertIn(
                    f"]({target})",
                    source.read_text(encoding="utf-8"),
                    f"{source_path} must link to contexts/README.md",
                )
                self.assertTrue(
                    (source.parent / target).resolve().is_file(),
                    f"broken context entry link from {source_path}: {target}",
                )
