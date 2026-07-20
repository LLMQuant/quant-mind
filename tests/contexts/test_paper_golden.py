import json
import unittest
from pathlib import Path

import pymupdf

_GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "paper" / "golden"


class PaperGoldenFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.expected = json.loads(
            (_GOLDEN / "expected.json").read_text(encoding="utf-8")
        )
        with pymupdf.open(_GOLDEN / "paper.pdf") as document:
            cls.page_texts = [page.get_text() for page in document]

    def test_document_uses_pdf_golden_span_contract(self) -> None:
        document = self.expected["document"]

        self.assertEqual(document["span_unit"], "pdf_page")
        self.assertEqual(document["indexing"], "1-based-inclusive")
        self.assertEqual(len(self.page_texts), document["page_count"])
        self.assertIn(document["title"], self.page_texts[0])

    def test_titles_paths_spans_and_anchors_match_pdf(self) -> None:
        page_count = self.expected["document"]["page_count"]
        nodes = {node["key"]: node for node in self.expected["nodes"]}

        for key, node in nodes.items():
            with self.subTest(node=key):
                start = node["span"]["start"]
                end = node["span"]["end"]
                self.assertGreaterEqual(start, 1)
                self.assertLessEqual(start, end)
                self.assertLessEqual(end, page_count)

                span_text = "\n".join(self.page_texts[start - 1 : end])
                self.assertIn(node["title"], span_text)

                anchor_pages = set()
                for anchor in node["anchors"]:
                    page = anchor["page"]
                    anchor_pages.add(page)
                    self.assertGreaterEqual(page, start)
                    self.assertLessEqual(page, end)
                    self.assertIn(anchor["text"], self.page_texts[page - 1])

                if start != end:
                    self.assertTrue({start, end}.issubset(anchor_pages))

                expected_path = []
                cursor = node
                path_keys = set()
                while cursor is not None:
                    cursor_key = cursor["key"]
                    self.assertNotIn(cursor_key, path_keys)
                    path_keys.add(cursor_key)
                    expected_path.append(cursor["title"])
                    parent = cursor["parent"]
                    cursor = nodes[parent] if parent is not None else None
                self.assertEqual(node["path"], list(reversed(expected_path)))

    def test_topology_satisfies_declared_tree_invariants(self) -> None:
        required_invariants = {
            "single_root",
            "node_keys_unique",
            "parents_exist",
            "children_exist",
            "parent_child_bidirectional",
            "all_nodes_reachable",
            "acyclic",
            "spans_within_document",
        }
        self.assertEqual(set(self.expected["invariants"]), required_invariants)

        nodes_list = self.expected["nodes"]
        nodes = {node["key"]: node for node in nodes_list}
        self.assertEqual(len(nodes), len(nodes_list))

        roots = [node for node in nodes_list if node["parent"] is None]
        self.assertEqual(len(roots), 1)
        root = roots[0]
        self.assertEqual(root["key"], self.expected["document"]["root"])

        child_owners = {key: 0 for key in nodes}
        for key, node in nodes.items():
            with self.subTest(node=key):
                self.assertEqual(
                    len(node["children"]), len(set(node["children"]))
                )
                if node["parent"] is not None:
                    self.assertIn(node["parent"], nodes)
                for child_key in node["children"]:
                    self.assertIn(child_key, nodes)
                    self.assertEqual(nodes[child_key]["parent"], key)
                    child_owners[child_key] += 1

        self.assertEqual(child_owners[root["key"]], 0)
        for key, owner_count in child_owners.items():
            if key != root["key"]:
                self.assertEqual(owner_count, 1)

        visiting = set()
        visited = set()

        def visit(key: str) -> None:
            self.assertNotIn(key, visiting, f"cycle detected at {key}")
            if key in visited:
                return
            visiting.add(key)
            for child_key in nodes[key]["children"]:
                visit(child_key)
            visiting.remove(key)
            visited.add(key)

        visit(root["key"])
        self.assertEqual(visited, set(nodes))

    def test_fixture_preserves_overlapping_sibling_spans(self) -> None:
        nodes = {node["key"]: node for node in self.expected["nodes"]}
        method = nodes["method"]
        limitations = nodes["limitations"]

        self.assertEqual(method["parent"], limitations["parent"])
        self.assertLessEqual(
            limitations["span"]["start"], method["span"]["end"]
        )


if __name__ == "__main__":
    unittest.main()
