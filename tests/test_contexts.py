import re
import unittest
from pathlib import Path


class TestContextEntryPoints(unittest.TestCase):
    def test_context_index_targets_exist(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        context_indexes = (
            "contexts/README.md",
            "contexts/dev/README.md",
            "contexts/dev/github-writing.md",
            "contexts/dev/labels.md",
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

    def test_label_guide_has_required_routes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        required_links = {
            "contexts/dev/README.md": "labels.md",
            ".agents/skills/quantmind-dev/SKILL.md": (
                "../../../contexts/dev/labels.md"
            ),
            ".claude/skills/quantmind-dev/SKILL.md": (
                "../../../contexts/dev/labels.md"
            ),
        }

        for source_path, target in required_links.items():
            source = repo_root / source_path
            with self.subTest(source=source_path):
                self.assertIn(
                    f"]({target})",
                    source.read_text(encoding="utf-8"),
                    f"{source_path} must route to the canonical label guide",
                )
                self.assertTrue(
                    (source.parent / target).resolve().is_file(),
                    f"broken label guide link from {source_path}: {target}",
                )

    def test_label_guide_has_complete_taxonomy(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        guide = (repo_root / "contexts/dev/labels.md").read_text(
            encoding="utf-8"
        )
        expected_labels = {
            "type": {
                "type: bug",
                "type: feature",
                "type: refactor",
                "type: docs",
                "type: maintenance",
                "type: design",
            },
            "area": {
                "area: contexts",
                "area: harness",
                "area: knowledge",
                "area: configs",
                "area: preprocess",
                "area: flows",
                "area: mind",
                "area: utils",
                "area: examples",
                "area: packaging",
            },
            "impact": {
                "impact: breaking",
                "impact: live-network",
            },
        }

        for dimension, expected in expected_labels.items():
            found = set(re.findall(rf"`({dimension}: [a-z-]+)`", guide))
            with self.subTest(dimension=dimension):
                self.assertEqual(found, expected)

    def test_github_writing_guide_has_required_routes(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        required_links = {
            "contexts/dev/README.md": "github-writing.md",
            "contexts/dev/labels.md": "github-writing.md",
            "AGENTS.md": "contexts/dev/github-writing.md",
            "CLAUDE.md": "contexts/dev/github-writing.md",
            ".agents/skills/quantmind-dev/SKILL.md": (
                "../../../contexts/dev/github-writing.md"
            ),
            ".claude/skills/quantmind-dev/SKILL.md": (
                "../../../contexts/dev/github-writing.md"
            ),
            ".agents/skills/quantmind-dev/references/pull-request.md": (
                "../../../../contexts/dev/github-writing.md"
            ),
            ".claude/skills/quantmind-dev/references/pull-request.md": (
                "../../../../contexts/dev/github-writing.md"
            ),
        }

        for source_path, target in required_links.items():
            source = repo_root / source_path
            with self.subTest(source=source_path):
                self.assertIn(
                    f"]({target})",
                    source.read_text(encoding="utf-8"),
                    f"{source_path} must route to the GitHub writing guide",
                )
                self.assertTrue(
                    (source.parent / target).resolve().is_file(),
                    f"broken GitHub writing guide link: {target}",
                )

        marker = "<!-- github-prose-style:"
        templates = (
            ".github/ISSUE_TEMPLATE/bug_report.md",
            ".github/ISSUE_TEMPLATE/feature_request.md",
            ".github/PULL_REQUEST_TEMPLATE.md",
        )
        for template_path in templates:
            with self.subTest(template=template_path):
                template = repo_root / template_path
                self.assertIn(
                    marker,
                    template.read_text(encoding="utf-8"),
                    f"{template_path} must remind authors not to hard-wrap",
                )

    def test_quantmind_dev_skill_mirrors_match(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        mirror_paths = (
            "SKILL.md",
            "references/commit.md",
            "references/develop-components.md",
            "references/pull-request.md",
        )

        for relative_path in mirror_paths:
            agents_skill = (
                repo_root / ".agents/skills/quantmind-dev" / relative_path
            )
            claude_skill = (
                repo_root / ".claude/skills/quantmind-dev" / relative_path
            )
            with self.subTest(path=relative_path):
                self.assertEqual(
                    agents_skill.read_bytes(), claude_skill.read_bytes()
                )
