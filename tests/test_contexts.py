import re
import unittest
from pathlib import Path


def _guide_text(repo_root: Path, rel_path: str) -> str:
    """Read a guide file, inlining any ``@path`` import lines.

    Claude Code's memory-import syntax lets ``CLAUDE.md`` delegate to
    ``AGENTS.md`` with a single ``@AGENTS.md`` line. Expanding it here checks a
    guide by its effective content, so the single-source layout still satisfies
    the literal-content assertions below.
    """
    text = (repo_root / rel_path).read_text(encoding="utf-8")

    def _expand(match: "re.Match[str]") -> str:
        return (repo_root / match.group(1)).read_text(encoding="utf-8")

    return re.sub(r"(?m)^@([^\s`]+)\s*$", _expand, text)


class TestContextEntryPoints(unittest.TestCase):
    def test_context_pages_support_progressive_disclosure(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        context_root = repo_root / "contexts"
        contents_link = re.compile(r"^- \[[^]]+]\(#([^)]+)\)$", re.MULTILINE)

        def github_anchor(heading: str) -> str:
            normalized = re.sub(r"[^\w\s-]", "", heading.lower())
            return re.sub(r"[\s-]+", "-", normalized).strip("-")

        for page in sorted(context_root.rglob("*.md")):
            lines = page.read_text(encoding="utf-8").splitlines()
            preview_lines = lines[:80]
            preview = "\n".join(preview_lines)
            contents_targets = set(contents_link.findall(preview))
            detail_targets = {
                github_anchor(line.removeprefix("## "))
                for line in lines
                if line.startswith("## ")
                and line not in {"## Quick Summary", "## Contents"}
            }
            with self.subTest(page=page.relative_to(repo_root)):
                self.assertIn("## Quick Summary", preview)
                self.assertIn("## Contents", preview)
                self.assertLess(
                    preview_lines.index("## Quick Summary"),
                    preview_lines.index("## Contents"),
                )
                self.assertEqual(contents_targets, detail_targets)

    def test_agent_guides_define_progressive_context_loading(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]

        for guide_path in ("AGENTS.md", "CLAUDE.md"):
            guide = _guide_text(repo_root, guide_path)
            with self.subTest(guide=guide_path):
                self.assertIn("## Progressive Context Loading", guide)
                self.assertIn("Read lines 1-80 first.", guide)
                self.assertIn("read the entire page", guide)
                self.assertIn(
                    "The preview routes work; it does not replace the",
                    guide,
                )

    def test_context_index_targets_exist(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        context_indexes = (
            "contexts/README.md",
            "contexts/dev/README.md",
            "contexts/dev/github-writing.md",
            "contexts/dev/labels.md",
            "contexts/design/README.md",
            "contexts/design/flow/news.md",
            "contexts/design/flow/paper.md",
            "contexts/design/library/local.md",
            "contexts/design/operations/naming.md",
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
                    _guide_text(repo_root, source_path),
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
                    _guide_text(repo_root, source_path),
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
                "area: rag",
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
                    _guide_text(repo_root, source_path),
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
