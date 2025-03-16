import os
import re
import json
import arxiv
import yaml
import logging
import argparse
import datetime
import requests


# Configure logging
logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Base URLs for various APIs
BASE_URL = "https://arxiv.paperswithcode.com/api/v0/papers/"
GITHUB_URL = "https://api.github.com/search/repositories"
ARXIV_URL = "http://arxiv.org/"


def load_config(config_file: str) -> dict:
    """Load configuration from a YAML file.

    Parameters:
    ----------
    config_file : str
        Path to the configuration file.

    Returns:
    -------
    dict
        Dictionary containing configuration settings.
    """

    def pretty_filters(**config) -> dict:
        """Process and format filter keywords from the configuration.

        Parameters:
        ----------
        config : dict
            Configuration dictionary with a 'keywords' key.

        Returns:
        -------
        dict
            Dictionary with formatted keyword filters.
        """
        keywords = dict()
        EXCAPE = '"'
        QUOTA = ""  # Not used currently
        OR = " OR "  # String to join multiple filters

        def parse_filters(filters: list) -> str:
            """Format a list of filter strings into a single query string.

            Parameters:
            ----------
            filters : list
                List of filter strings.

            Returns:
            -------
            str
                Formatted filter string.
            """
            ret = ""
            for idx in range(0, len(filters)):
                current_filter = filters[idx]
                if len(current_filter.split()) > 1:
                    ret += EXCAPE + current_filter + EXCAPE
                else:
                    ret += QUOTA + current_filter + QUOTA
                if idx != len(filters) - 1:
                    ret += OR
            return ret

        for k, v in config["keywords"].items():
            keywords[k] = parse_filters(v["filters"])
        return keywords

    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["kv"] = pretty_filters(**config)
        logging.info(f"config = {config}")
    return config


def get_authors(authors, partial_author: bool = False) -> str:
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


def sort_papers(papers: dict) -> dict:
    """Sort papers in descending order by their keys.

    Parameters:
    ----------
    papers : dict
        Dictionary of papers.

    Returns:
    -------
    dict
        Sorted dictionary of papers.
    """
    output = dict()
    keys = list(papers.keys())
    keys.sort(reverse=True)
    for key in keys:
        output[key] = papers[key]
    return output


def get_code_link(qword: str) -> str:
    """Retrieve the GitHub repository link corresponding to the query.

    Parameters:
    ----------
    qword : str
        Query string (e.g., arXiv ID or paper title).

    Returns:
    -------
    str
        Repository URL if found; otherwise, None.
    """
    query = f"{qword}"
    params = {"q": query, "sort": "stars", "order": "desc"}
    r = requests.get(GITHUB_URL, params=params)
    results = r.json()
    code_link = None
    if results["total_count"] > 0:
        code_link = results["items"][0]["html_url"]
    return code_link


def get_daily_papers(topic: str, query="quantitative finance", max_results=2):
    """Retrieve daily papers based on a topic and search query.

    Downloads the PDF for each paper and attempts to retrieve
    the corresponding code repository URL.

    Parameters:
    ----------
    topic : str
        Topic name used for folder naming.
    query : str, optional
        Search query for papers.
    max_results : int, optional
        Maximum number of papers to retrieve.

    Returns:
    -------
    tuple
        Two dictionaries; one for standard content and one for web content.
    """
    # Create folder structure based on current month and topic
    today_month = datetime.date.today().strftime("%Y-%m")
    folder_path = os.path.join(os.getcwd(), "papers", today_month)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    query_folder_path = os.path.join(folder_path, topic)
    if not os.path.exists(query_folder_path):
        os.makedirs(query_folder_path)

    content = dict()
    content_to_web = dict()
    search_engine = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    for result in search_engine.results():
        paper_id = result.get_short_id()
        paper_title = result.title
        paper_url = result.entry_id
        code_url = BASE_URL + paper_id  # API endpoint for code link
        paper_abstract = result.summary.replace("\n", " ")
        paper_authors = get_authors(result.authors)
        paper_first_author = get_authors(result.authors, partial_author=True)
        primary_category = result.primary_category
        publish_time = result.published.date()
        update_time = result.updated.date()
        comments = (
            result.comment.replace("\n", " ")
            if result.comment is not None
            else ""
        )
        paper_summary = ""

        logging.info(
            f"Time = {update_time} title = {paper_title} author = {paper_first_author}"
        )

        # Remove version from arXiv ID (e.g., 2108.09112v1 -> 2108.09112)
        ver_pos = paper_id.find("v")
        if ver_pos == -1:
            paper_key = paper_id
        else:
            paper_key = paper_id[0:ver_pos]
        paper_url = ARXIV_URL + "abs/" + paper_key

        # Download the PDF file
        pdf_url = result.pdf_url
        pdf_response = requests.get(pdf_url)
        pdf_filename = os.path.join(query_folder_path, f"{paper_key}.pdf")
        with open(pdf_filename, "wb") as pdf_file:
            pdf_file.write(pdf_response.content)
        logging.info(f"Downloaded PDF for {paper_title} to {pdf_filename}")

        paper_summary = paper_abstract

        try:
            # Retrieve repository link from the code API
            r = requests.get(code_url).json()
            repo_url = None
            if "official" in r and r["official"]:
                repo_url = r["official"]["url"]
            # TODO: If repository URL is not found, attempt additional queries
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
                        paper_summary,
                    )
                )
                content_to_web[paper_key] = (
                    "- {}, **{}**, {} et.al., Paper: [{}]({}), Code: **[{}]({})**, Comment:{}, Summary: {}\n".format(
                        update_time,
                        paper_title,
                        paper_first_author,
                        paper_url,
                        paper_url,
                        repo_url,
                        repo_url,
                        comments,
                        paper_summary,
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
                        paper_summary,
                    )
                )
                content_to_web[paper_key] = (
                    "- {}, **{}**, {} et.al., Paper: [{}]({}), Comment: {}, Abstract: {}\n".format(
                        update_time,
                        paper_title,
                        paper_first_author,
                        paper_url,
                        paper_url,
                        comments,
                        paper_summary,
                    )
                )

            # Append comments if available (currently not used)
            if comments is not None:
                content_to_web[paper_key] += f", {comments}\n"
            else:
                content_to_web[paper_key] += "\n"

        except Exception as e:
            logging.error(f"Exception: {e} with id: {paper_key}")

    data = {topic: content}
    data_web = {topic: content_to_web}
    return data, data_web


def update_paper_links(filename: str):
    """Update paper links in the JSON file on a weekly basis.

    Parameters:
    ----------
    filename : str
        Path to the JSON file.
    """

    def parse_arxiv_string(s: str):
        """Parse a string from the JSON file to extract paper metadata.

        Parameters:
        ----------
        s : str
            A string representing a paper's details.

        Returns:
        -------
        tuple
            Date, title, authors, arXiv ID, code link, comment, and summary.
        """
        parts = s.split("|")
        date = parts[1].strip()
        title = parts[2].strip()
        authors = parts[3].strip()
        arxiv_id = parts[4].strip()
        code = parts[5].strip()
        comment = parts[6].strip()
        summary = parts[7].strip() if len(parts) > 7 else "No summary available"
        arxiv_id = re.sub(r"v\d+", "", arxiv_id)
        return date, title, authors, arxiv_id, code, comment, summary

    with open(filename, "r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

    json_data = m.copy()

    for keywords, v in json_data.items():
        logging.info(f"keywords = {keywords}")
        for paper_id, contents in v.items():
            contents = str(contents)
            (
                update_time,
                paper_title,
                paper_first_author,
                paper_url,
                code_url,
                paper_comment,
                paper_summary,
            ) = parse_arxiv_string(contents)
            contents = "|{}|{}|{}|{}|{}|{}|{}|\n".format(
                update_time,
                paper_title,
                paper_first_author,
                paper_url,
                code_url,
                paper_comment,
                paper_summary,
            )
            json_data[keywords][paper_id] = str(contents)
            logging.info(f"paper_id = {paper_id}, contents = {contents}")

            valid_link = False if "|null|" in contents else True
            if valid_link:
                continue
            try:
                code_url = BASE_URL + paper_id  # API endpoint for code link
                r = requests.get(code_url).json()
                repo_url = None
                if "official" in r and r["official"]:
                    repo_url = r["official"]["url"]
                    if repo_url is not None:
                        new_cont = contents.replace(
                            "|null|", f"|**[link]({repo_url})**|"
                        )
                        logging.info(f"ID = {paper_id}, contents = {new_cont}")
                        json_data[keywords][paper_id] = str(new_cont)
            except Exception as e:
                logging.error(f"Exception: {e} with id: {paper_id}")
    # Write updated data back to the JSON file
    with open(filename, "w") as f:
        json.dump(json_data, f)


def update_json_file(filename: str, data_dict):
    """Daily update of the JSON file using the new data dictionary.

    Parameters:
    ----------
    filename : str
        Path to the JSON file.
    data_dict : list
        List of dictionaries containing new paper data.
    """
    if not os.path.exists(filename):
        with open(filename, "w") as f:
            f.write("{}")

    with open(filename, "r") as f:
        content = f.read()
        if not content:
            m = {}
        else:
            m = json.loads(content)

    json_data = m.copy()

    # Update papers for each keyword
    for data in data_dict:
        for keyword in data.keys():
            papers = data[keyword]
            if keyword in json_data.keys():
                json_data[keyword].update(papers)
            else:
                json_data[keyword] = papers

    with open(filename, "w") as f:
        json.dump(json_data, f)


def json_to_md(
    filename: str,
    md_filename: str,
    task: str = "",
    to_web: bool = False,
    use_title: bool = True,
    use_tc: bool = True,
    show_badge: bool = True,
    use_b2t: bool = True,
):
    """Convert JSON data to a Markdown file with configurable formatting.

    Parameters:
    ----------
    filename : str
        Path to the input JSON file.
    md_filename : str
        Path to the output Markdown file.
    task : str, optional
        Description of the task for logging.
    to_web : bool, optional
        Flag indicating if formatting is for web publishing.
    use_title : bool, optional
        Flag to include a title section.
    use_tc : bool, optional
        Flag to include a table of contents.
    show_badge : bool, optional
        Flag to indicate if badges are to be shown.
    use_b2t : bool, optional
        Flag to include a back-to-top link.
    """

    def pretty_math(s: str) -> str:
        """Format math expressions for proper spacing.

        Parameters:
        ----------
        s : str
            String that may contain math expressions.

        Returns:
        -------
        str
            Formatted string.
        """
        ret = ""
        match = re.search(r"\$.*\$", s)
        if match is None:
            return s
        math_start, math_end = match.span()
        space_trail = space_leading = ""
        if s[:math_start][-1] != " " and s[:math_start][-1] != "*":
            space_trail = " "
        if s[math_end:][0] != " " and s[math_end:][0] != "*":
            space_leading = " "
        ret += s[:math_start]
        ret += f"{space_trail}${match.group()[1:-1].strip()}${space_leading}"
        ret += s[math_end:]
        return ret

    def format_abstract(abstract: str) -> str:
        """Format the abstract field with math formatting and wrap in HTML.

        Parameters:
        ----------
        abstract : str
            The raw abstract text.

        Returns:
        -------
        str
            Formatted abstract wrapped in a scrollable container.
        """
        # First, format any math expressions within the abstract.
        formatted = pretty_math(abstract)
        # Wrap in an HTML div with inline styles to limit height and add a scrollbar.
        return f"<details><summary>Abstract (click to expand)</summary>{formatted}</details>"

    def parse_markdown_row(row: str) -> dict:
        """Parse a Markdown table row into a dictionary.

        Parameters:
        ----------
        row : str
            A Markdown table row as a string.

        Returns:
        -------
        dict
            Dictionary containing parsed values.
        """
        columns = [col.strip() for col in row.strip().strip("|").split("|")]

        if len(columns) != 7:
            print(f"Warning: Row does not contain exactly 6 columns: {row}")
            return {}

        return {
            "publish_date": columns[0].strip("**"),  # 去掉 Markdown 加粗符号 **
            "title": columns[1].strip("**"),
            "authors": columns[2],
            "pdf": columns[3],
            "code": columns[4] if columns[4] != "null" else "",
            "comments": columns[5] if columns[5] != "null" else "",
            "abstract": columns[6],
        }

    def generate_table_row(v: str, use_title: bool, to_web: bool) -> str:
        """Convert a Markdown row string into a properly formatted row.

        Parameters:
        ----------
        v : str
            A Markdown table row string.
        use_title : bool
            Whether to format for web title.
        to_web : bool
            Whether to format for web display.

        Returns:
        -------
        str
            A formatted Markdown table row string.
        """
        v_dict = parse_markdown_row(v)

        if not v_dict:
            return ""

        publish_date = v_dict["publish_date"]
        title = v_dict["title"]
        authors = v_dict["authors"]
        pdf = v_dict["pdf"]
        code = v_dict["code"]
        comments = v_dict["comments"]
        abstract_raw = v_dict["abstract"]

        abstract = format_abstract(abstract_raw)

        return f"| {publish_date} | {title} | {authors} | {pdf} | {code} | {comments} | {abstract} |\n"

    # Format current date
    DateNow = datetime.date.today()
    DateNow = str(DateNow)
    DateNow = DateNow.replace("-", ".")

    with open(filename, "r") as f:
        content = f.read()
        if not content:
            data = {}
        else:
            data = json.loads(content)

    # Clear or create the Markdown file
    with open(md_filename, "w+") as f:
        pass

    # Write data into the Markdown file
    with open(md_filename, "a+") as f:
        if use_title and to_web:
            f.write("---\nlayout: default\n---\n\n")

        if use_title:
            f.write(
                '<p align="center"><h1 align="center">🌟 QUANT-SCHOLAR 🌟</h1><h2 align="center">Automatically Quantitative Finance Papers List</h2></p> \n'
            )
            f.write(
                '<p align="center"><img src="https://raw.githubusercontent.com/LLMQuant/quant-scholar/main/asset/icon.png" width="180"></p>'
            )
            f.write(f"\n \n")
            f.write(f"## 🚩 Updated on {DateNow} \n")
        else:
            f.write(f"> 🚩 Updated on {DateNow}\n")

        # Contents
        if use_tc:
            f.write("<details>\n")
            f.write("  <summary><strong>📜 Contents</strong></summary>\n")
            f.write("  <ol>\n")
            for keyword in data.keys():
                day_content = data[keyword]
                if not day_content:
                    continue
                kw = keyword.replace(" ", "-").lower()
                f.write(f"    <li><a href=#-{kw}>📌 {keyword}</a></li>\n")
            f.write("  </ol>\n")
            f.write("</details>\n\n")

        # Write each keyword section with its papers
        for keyword in data.keys():
            day_content = data[keyword]
            if not day_content:
                continue

            f.write(f"## 📌 {keyword}\n\n")

            if use_title:
                f.write(
                    "| 📅 Publish Date | 📖 Title | 👨‍💻 Authors | 🔗 PDF | 💻 Code | 💬 Comment | 📜 Abstract |\n"
                )
                f.write(
                    "|:--------------:|:----------------------------|:------------------|:------:|:------:|:-------:|:--------|\n"
                )
            # Sort papers by date and write each entry
            day_content = sort_papers(day_content)
            for _, v in day_content.items():
                if v is not None:
                    f.write(generate_table_row(v, use_title, to_web))
            f.write("\n")

            # Add a back-to-top link if enabled
            if use_b2t:
                top_info = f"#-Updated on {DateNow}"
                top_info = top_info.replace(" ", "-").replace(".", "")
                f.write(
                    f"<p align=right>(<a href={top_info.lower()}>back to top</a>)</p>\n\n"
                )

    logging.info(f"{task} finished")


def demo(**config):
    """Main function to collect paper data and update Markdown files.

    Parameters
    ----------
    config : dict
        Dictionary containing configuration settings.
    """
    data_collector = []
    data_collector_web = []

    keywords = config["kv"]
    max_results = config["max_results"]
    publish_readme = config["publish_readme"]
    show_badge = config["show_badge"]

    b_update = config["update_paper_links"]
    logging.info(f"Update Paper Link = {b_update}")
    if not config["update_paper_links"]:
        logging.info("GET daily papers begin")
        for topic, keyword in keywords.items():
            logging.info(f"Keyword: {topic}")
            data, data_web = get_daily_papers(
                topic, query=keyword, max_results=max_results
            )
            data_collector.append(data)
            data_collector_web.append(data_web)
            print("\n")
        logging.info("GET daily papers end")

    if publish_readme:
        json_file = config["paper_list_json_path"]
        md_file = config["paper_list_path"]
        if config["update_paper_links"]:
            update_paper_links(json_file)
        else:
            update_json_file(json_file, data_collector)
        json_to_md(
            json_file, md_file, task="Update Readme", show_badge=show_badge
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default="config.yaml",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--update_paper_links",
        default=False,
        action="store_true",
        help="Flag to update paper links",
    )
    args = parser.parse_args()
    config = load_config(args.config_path)
    config = {**config, "update_paper_links": args.update_paper_links}
    demo(**config)
