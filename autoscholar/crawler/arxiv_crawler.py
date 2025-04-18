import os
import json
import arxiv
import logging
import datetime
import requests
from typing import Dict, List, Any, Optional

from base_crawler import BaseCrawler


class ArxivCrawler(BaseCrawler):
    """Crawler for fetching papers from arXiv.

    This crawler uses the arXiv API to fetch papers based on queries
    and saves the data in a structured format.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the ArXiv crawler.

        Parameters:
        ----------
        config : Dict[str, Any]
            Dictionary containing configuration settings.
        """
        super().__init__(config)
        # ArXiv-specific constants
        self.ARXIV_URL = "http://arxiv.org/"
        self.BASE_URL = "https://arxiv.paperswithcode.com/api/v0/papers/"

    def get_authors(self, authors, partial_author: bool = False) -> str:
        """Retrieve a formatted string of authors.

        Parameters:
        ----------
        authors : list
            List of author names.
        partial_author : bool, optional
            If True, return only the first three authors.

        Returns:
        -------
        str
            String of author names.
        """
        if not partial_author:
            return ", ".join(str(author) for author in authors)
        else:
            return ", ".join(str(author) for author in authors[:3])

    def fetch_data(
        self, topic: str, query: str, max_results: int = 10
    ) -> Dict[str, Any]:
        """Fetch papers from arXiv based on the query.

        Parameters:
        ----------
        topic : str
            Topic name for categorization.
        query : str
            Search query string.
        max_results : int, optional
            Maximum number of papers to fetch.

        Returns:
        -------
        Dict[str, Any]
            Dictionary containing the fetched papers.
        """
        content = {}

        # Create folder structure based on current month and topic
        today_month = datetime.date.today().strftime("%Y-%m")
        folder_path = os.path.join(self.output_dir, today_month)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        topic_folder_path = os.path.join(folder_path, topic)
        if not os.path.exists(topic_folder_path):
            os.makedirs(topic_folder_path)

        search_engine = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
        )

        for result in search_engine.results():
            paper_id = result.get_short_id()
            paper_title = result.title
            paper_url = result.entry_id
            code_url = self.BASE_URL + paper_id  # API endpoint for code link
            paper_abstract = result.summary.replace("\n", " ")
            paper_authors = self.get_authors(result.authors)
            paper_first_author = self.get_authors(
                result.authors, partial_author=True
            )
            primary_category = result.primary_category
            publish_time = result.published.date()
            update_time = result.updated.date()
            comments = (
                result.comment.replace("\n", " ")
                if result.comment is not None
                else ""
            )

            logging.info(
                f"Time = {update_time} title = {paper_title} author = {paper_first_author}"
            )

            # Remove version from arXiv ID (e.g., 2108.09112v1 -> 2108.09112)
            ver_pos = paper_id.find("v")
            if ver_pos == -1:
                paper_key = paper_id
            else:
                paper_key = paper_id[0:ver_pos]
            paper_url = self.ARXIV_URL + "abs/" + paper_key

            # Download the PDF file if enabled in config
            if self.config.get("download_pdf", False):
                pdf_url = result.pdf_url
                pdf_response = requests.get(pdf_url)
                pdf_filename = os.path.join(
                    topic_folder_path, f"{paper_key}.pdf"
                )
                with open(pdf_filename, "wb") as pdf_file:
                    pdf_file.write(pdf_response.content)
                logging.info(
                    f"Downloaded PDF for {paper_title} to {pdf_filename}"
                )

            # Try to get code repository URL
            repo_url = None
            try:
                r = requests.get(code_url).json()
                if "official" in r and r["official"]:
                    repo_url = r["official"]["url"]
            except Exception as e:
                logging.error(f"Exception: {e} with id: {paper_key}")

            # Format the paper data
            if repo_url is not None:
                content[paper_key] = (
                    "|**{}**|**{}**|{} et.al.|[{}]({})|**[link]({})**|{}|{}|\n".format(
                        update_time,
                        paper_title,
                        paper_first_author,
                        paper_key,
                        paper_url,
                        repo_url,
                        comments,
                        paper_abstract,
                    )
                )
            else:
                content[paper_key] = (
                    "|**{}**|**{}**|{} et.al.|[{}]({})|null|{}|{}|\n".format(
                        update_time,
                        paper_title,
                        paper_first_author,
                        paper_key,
                        paper_url,
                        comments,
                        paper_abstract,
                    )
                )

        return {topic: content}

    def process_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the fetched paper data.

        For arXiv papers, we just pass through the data without additional processing.

        Parameters:
        ----------
        data : Dict[str, Any]
            Raw data fetched from arXiv.

        Returns:
        -------
        Dict[str, Any]
            Processed data.
        """
        return data

    def save_data(self, data: Dict[str, Any], output_path: str) -> None:
        """Save the paper data to a JSON file.

        Parameters:
        ----------
        data : Dict[str, Any]
            Processed paper data to save.
        output_path : str
            Path to save the data.
        """
        # Check if file exists
        if not os.path.exists(output_path):
            with open(output_path, "w") as f:
                f.write("{}")

        # Load existing data
        with open(output_path, "r") as f:
            content = f.read()
            if not content:
                existing_data = {}
            else:
                existing_data = json.loads(content)

        # Update with new data
        for topic, papers in data.items():
            if topic in existing_data:
                existing_data[topic].update(papers)
            else:
                existing_data[topic] = papers

        # Write back to file
        with open(output_path, "w") as f:
            json.dump(existing_data, f)

        logging.info(f"Saved paper data to {output_path}")
