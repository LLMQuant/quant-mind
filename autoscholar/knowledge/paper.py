# scholarauto/knowledge/paper.py

import json
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union


class Paper:
    """Class representing a research paper as a basic knowledge entity.

    A minimal data structure that stores essential paper information.
    Additional metadata can be stored in meta_info dictionary.
    """

    def __init__(
        self,
        title: str = "",
        abstract: str = "",
        id: Optional[str] = None,
        paper_id: Optional[str] = None,
        url: Optional[str] = None,
        pdf_url: Optional[str] = None,
        code_url: Optional[str] = None,
        full_text: Optional[str] = None,
        embedding: Optional[List[float]] = None,
        meta_info: Optional[Dict[str, Any]] = None,
    ):
        """Initialize a Paper knowledge entity.

        Parameters:
            title: Paper title
            abstract: Paper abstract
            id: Unique identifier for the paper (defaults to UUID)
            paper_id: Unique identifier for the paper (defaults to UUID)
            url: URL to the paper (optional)
            pdf_url: URL to the PDF version (optional)
            full_text: Complete text content (optional)
            embedding: Embedding of the paper (optional)
            meta_info: Dictionary for flexible metadata storage (optional)
        """
        self.id = id or str(uuid.uuid4())
        self.paper_id = paper_id
        self.title = title
        self.abstract = abstract
        self.url = url
        self.pdf_url = pdf_url
        self.full_text = full_text
        self.embedding = embedding
        self.code_url = code_url
        self.meta_info = meta_info or {}

        # Embedding is not stored directly as an attribute
        # but can be added to meta_info if needed temporarily

    def to_dict(self) -> Dict[str, Any]:
        """Convert the paper to a dictionary representation.

        Returns:
            Dictionary with paper attributes
        """
        return {
            "id": self.id,
            "paper_id": self.paper_id,
            "title": self.title,
            "abstract": self.abstract,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "full_text": self.full_text,
            "code_url": self.code_url,
            "meta_info": self.meta_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Paper":
        """Create a Paper object from a dictionary.

        Parameters:
            data: Dictionary with paper attributes

        Returns:
            Paper object
        """
        # Define known attributes
        known_attrs = [
            "id",
            "paper_id",
            "title",
            "abstract",
            "url",
            "pdf_url",
            "code_url",
            "full_text",
            "meta_info",
        ]

        # Extract known attributes
        attrs = {
            attr: data.get(attr, "" if attr in ["title", "abstract"] else None)
            for attr in known_attrs
        }

        # Get existing meta_info or create empty dict
        meta_info = (
            attrs.get("meta_info", {}).copy() if attrs.get("meta_info") else {}
        )

        # Add any remaining keys from data to meta_info
        for key, value in data.items():
            if key not in attrs:
                meta_info[key] = value

        # Update meta_info in attrs
        attrs["meta_info"] = meta_info

        return cls(**attrs)

    def get_text_for_embedding(
        self, construct_fn: Optional[Callable[["Paper"], str]] = None
    ) -> str:
        """Get concatenated text used for creating embeddings.

        Parameters:
        ----------
            construct_fn: Function to construct the text from the paper

        Returns:
            Concatenated text of title and abstract
        """
        if construct_fn is None:
            return f"{self.title}\n\n{self.abstract}"
        else:
            return construct_fn(self)

    def to_json(self) -> str:
        """Convert the paper to a JSON string.

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)

    def __str__(self) -> str:
        """Get string representation of the paper.

        Returns:
            String with basic paper info
        """
        return f"Paper({self.id}): {self.title}"

    def __repr__(self) -> str:
        """Get representation of the paper.

        Returns:
            String representation
        """
        return f"Paper(paper_id='{self.paper_id}', title='{self.title}')"

    def set_embedding(self, embedding: List[float]):
        """Set the embedding of the paper.

        Parameters:
            embedding: Embedding of the paper
        """
        self.embedding = embedding

    @classmethod
    def load_paper_from_path(cls, json_path: str) -> "Paper":
        """Load a paper from a JSON file.

        Parameters:
            json_path: Path to the JSON file containing paper data

        Returns:
            Paper object
        """
        with open(json_path, "r", encoding="utf-8") as f:
            paper_data = json.load(f)

        if not isinstance(paper_data, dict):
            raise ValueError("JSON file must be a dictionary")

        return cls.from_dict(paper_data)

    @classmethod
    def load_paper_from_paths(
        cls, json_path_list: List[Union[str, Path]]
    ) -> List["Paper"]:
        """Create a Paper object from a JSON file.

        Parameters:
            json_path: Path to the JSON file containing paper data
        """
        papers = []
        for json_path in json_path_list:
            papers.append(cls.load_paper_from_path(json_path))
        return papers

    @classmethod
    def from_json_list(cls, json_str: str) -> List["Paper"]:
        """Create multiple Paper objects from a JSON string containing a list.

        Parameters:
            json_str: JSON string containing a list of paper data

        Returns:
            List of Paper objects
        """
        papers_data = json.loads(json_str)

        if not isinstance(papers_data, list):
            raise ValueError("JSON string must contain a list of paper objects")

        return [cls.from_dict(paper_data) for paper_data in papers_data]

    @classmethod
    def to_json_list(cls, papers: List["Paper"]) -> str:
        """Convert a list of Paper objects to a JSON string.

        Parameters:
            papers: List of Paper objects

        Returns:
            JSON string representation of the list
        """
        papers_data = [paper.to_dict() for paper in papers]
        return json.dumps(papers_data, ensure_ascii=False, indent=2)
