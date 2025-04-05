import unittest

from autoscholar.knowledge.paper import Paper


class TestPaper(unittest.TestCase):
    """Test the Paper class."""

    def setUp(self):
        """Setup test data."""
        self.test_title = "Test paper title"
        self.test_abstract = "This is a test abstract"
        self.test_url = "http://example.com/paper"
        self.test_pdf_url = "http://example.com/paper.pdf"
        self.test_full_text = "This is the full text of the paper"
        self.test_meta_info = {"authors": ["Zhang San", "Li Si"]}

    def test_paper_creation(self):
        """Test the creation of a paper object."""
        paper = Paper(
            title=self.test_title,
            abstract=self.test_abstract,
            url=self.test_url,
            pdf_url=self.test_pdf_url,
            full_text=self.test_full_text,
            meta_info=self.test_meta_info,
        )

        self.assertEqual(paper.title, self.test_title)
        self.assertEqual(paper.abstract, self.test_abstract)
        self.assertEqual(paper.url, self.test_url)
        self.assertEqual(paper.pdf_url, self.test_pdf_url)
        self.assertEqual(paper.full_text, self.test_full_text)
        self.assertEqual(paper.meta_info, self.test_meta_info)
        self.assertIsNotNone(paper.id)  # Ensure a UUID is generated

    def test_to_dict_and_from_dict(self):
        """Test the dictionary conversion method."""
        original_paper = Paper(
            title=self.test_title, abstract=self.test_abstract
        )

        # Convert to dictionary
        paper_dict = original_paper.to_dict()

        # Create object from dictionary
        new_paper = Paper.from_dict(paper_dict)

        self.assertEqual(original_paper.title, new_paper.title)
        self.assertEqual(original_paper.abstract, new_paper.abstract)
        self.assertEqual(original_paper.id, new_paper.id)

    def test_get_text_for_embedding(self):
        """Test the method to get the text for embedding."""
        paper = Paper(title=self.test_title, abstract=self.test_abstract)

        expected_text = f"{self.test_title}\n\n{self.test_abstract}"
        self.assertEqual(paper.get_text_for_embedding(), expected_text)

    def test_set_embedding(self):
        """Test the method to set the embedding."""
        paper = Paper(title=self.test_title, abstract=self.test_abstract)

        embedding = [0.1, 0.2, 0.3]
        paper.set_embedding(embedding)
        self.assertEqual(paper.embedding, embedding)

    def test_json_operations(self):
        """Test the JSON related operations."""
        # Create test data
        papers = [
            Paper(title="Paper 1", abstract="Abstract 1"),
            Paper(title="Paper 2", abstract="Abstract 2"),
        ]

        # Test to_json_list
        json_str = Paper.to_json_list(papers)

        # Test from_json_list
        loaded_papers = Paper.from_json_list(json_str)

        self.assertEqual(len(loaded_papers), 2)
        self.assertEqual(loaded_papers[0].title, "Paper 1")
        self.assertEqual(loaded_papers[1].title, "Paper 2")

    def test_paper_repr(self):
        """Test the string representation of a paper."""
        paper = Paper(
            title=self.test_title, abstract=self.test_abstract, paper_id="1234"
        )

        expected_repr = f"Paper(paper_id='1234', title='{self.test_title}')"
        self.assertEqual(repr(paper), expected_repr)

        expected_str = f"Paper({paper.id}): {self.test_title}"
        self.assertEqual(str(paper), expected_str)

    def test_custom_text_for_embedding(self):
        """Test using a custom function for embedding text."""
        paper = Paper(
            title=self.test_title,
            abstract=self.test_abstract,
            meta_info={"keywords": ["test", "unit testing"]},
        )

        def custom_text_fn(paper):
            return f"{paper.title} {paper.abstract} Keywords: {' '.join(paper.meta_info['keywords'])}"

        expected_text = f"{self.test_title} {self.test_abstract} Keywords: test unit testing"
        self.assertEqual(
            paper.get_text_for_embedding(custom_text_fn), expected_text
        )
