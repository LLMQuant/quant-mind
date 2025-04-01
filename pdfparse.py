from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import os
import logging
import subprocess


class ParseTool(ABC):
    """Base abstract class for PDF parsing tools."""

    @abstractmethod
    def parse(self, file_path: str) -> Union[str, Dict]:
        """Parse PDF file into structured format.

        Args:
            file_path: Path to the PDF file

        Returns:
            Parsed content in the appropriate format (markdown string or JSON dict)
        """
        pass

    @abstractmethod
    def get_format(self) -> str:
        """Return the output format of this parser."""
        pass


class PDF2MarkdownTool(ParseTool):
    """Tool for converting PDFs to Markdown format using marker library."""

    def __init__(
        self,
        cleanup: bool = True,
        extract_images: bool = False,
        output_dir: Optional[str] = None,
    ):
        """
        初始化 PDF 转 Markdown 工具

        Args:
            cleanup: 是否清理转换后的 markdown 内容
            extract_images: 是否提取图片
            output_dir: 输出目录，如果不指定则使用默认目录
        """
        self.cleanup = cleanup
        self.extract_images = extract_images
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)

    def parse(self, file_path: str) -> str:
        """Convert PDF to markdown using marker library.

        Args:
            file_path: Path to the PDF file

        Returns:
            String containing markdown representation of the PDF
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        self.logger.info(f"Converting {file_path} to markdown")

        try:
            # 构建 marker 命令
            cmd = ["marker_single", file_path]

            # 添加可选参数
            if self.output_dir:
                cmd.extend(["--output_dir", self.output_dir])
            if self.extract_images:
                cmd.append("--extract_images")

            # 执行转换命令
            result = subprocess.run(cmd, capture_output=True, text=True)

            # 检查是否成功
            if result.returncode != 0:
                raise Exception(f"转换失败: {result.stderr}")

            # 获取输出文件路径
            output_path = os.path.splitext(file_path)[0] + ".md"
            if self.output_dir:
                output_path = os.path.join(self.output_dir, os.path.basename(output_path))

            # 读取转换后的内容
            with open(output_path, "r", encoding="utf-8") as f:
                markdown_content = f.read()

            # 清理内容
            if self.cleanup:
                markdown_content = self._clean_markdown(markdown_content)

            self.logger.info("PDF 转换完成")
            return markdown_content

        except Exception as e:
            self.logger.error(f"转换过程中出错: {str(e)}")
            raise

    def get_format(self) -> str:
        return "markdown"

    def _clean_markdown(self, content: str) -> str:
        """Clean up the markdown content."""
        # 删除多余的空行
        content = "\n".join(line for line in content.split("\n") if line.strip())

        # 确保标题前后有空行
        content = content.replace("\n#", "\n\n#")
        content = content.replace("\n##", "\n\n##")
        content = content.replace("\n###", "\n\n###")

        return content.strip()


class PDF2JSONTool(ParseTool):
    """Tool for converting PDFs to structured JSON format."""

    def __init__(self, include_metadata: bool = True, extract_references: bool = True):
        self.include_metadata = include_metadata
        self.extract_references = extract_references
        self.logger = logging.getLogger(__name__)

    def parse(self, file_path: str) -> Dict:
        """Convert PDF to JSON structure.

        Args:
            file_path: Path to the PDF file

        Returns:
            Dict containing structured representation of the PDF
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        self.logger.info(f"Converting {file_path} to JSON structure")

        # Extract document structure
        document_structure = self._extract_document_structure(file_path)

        # Extract metadata if requested
        if self.include_metadata:
            metadata = self._extract_metadata(file_path)
            document_structure["metadata"] = metadata

        # Extract references if requested
        if self.extract_references:
            references = self._extract_references(file_path)
            document_structure["references"] = references

        return document_structure

    def get_format(self) -> str:
        return "json"

    def _extract_document_structure(self, file_path: str) -> Dict:
        """Extract document structure from PDF."""
        # Actual implementation would go here
        return {
            "title": "Sample Paper Title",
            "sections": [
                {
                    "heading": "Introduction",
                    "level": 1,
                    "content": "This is the introduction content...",
                    "subsections": [],
                },
                # More sections...
            ],
        }

    def _extract_metadata(self, file_path: str) -> Dict:
        """Extract document metadata."""
        return {
            "authors": ["Author 1", "Author 2"],
            "year": 2023,
            "doi": "10.1234/5678",
            "journal": "Journal of Sample Science",
        }

    def _extract_references(self, file_path: str) -> List[Dict]:
        """Extract references from the document."""
        return [
            {
                "authors": ["Author A", "Author B"],
                "year": 2020,
                "title": "A referenced paper",
                "journal": "Journal of References",
                "doi": "10.5678/1234",
            },
            # More references...
        ]


class Paper:
    """Class representing an academic paper with PDF parsing capabilities."""

    def __init__(self, path: Optional[str] = None, title: Optional[str] = None):
        self.path = path
        self.title = title
        self.content = None
        self.content_format = None
        self.metadata = {}
        self.logger = logging.getLogger(__name__)

    def parse_pdf(self, parse_tool: ParseTool) -> None:
        """Parse the PDF using the provided parsing tool.

        Args:
            parse_tool: An instance of a class implementing the ParseTool interface
        """
        if not self.path:
            raise ValueError("Paper path not set. Set paper.path before parsing.")

        try:
            self.logger.info(f"Parsing PDF using {parse_tool.__class__.__name__}")
            self.content = parse_tool.parse(self.path)
            self.content_format = parse_tool.get_format()

            # Extract title if not already set
            if not self.title and self.content_format == "json":
                self.title = self.content.get("title", None)

            self.logger.info(f"Successfully parsed PDF to {self.content_format} format")

        except Exception as e:
            self.logger.error(f"Error parsing PDF: {str(e)}")
            raise

    def get_content(self) -> Union[str, Dict]:
        """Get the parsed content."""
        if self.content is None:
            raise ValueError("Paper has not been parsed yet. Call parse_pdf first.")
        return self.content

    def save_parsed_content(self, output_path: Optional[str] = None) -> str:
        """Save the parsed content to a file.

        Args:
            output_path: Custom path to save the file. If not provided, will use the PDF name.

        Returns:
            Path to the saved file
        """
        if self.content is None:
            raise ValueError("No content to save. Parse the PDF first.")

        if output_path is None:
            # Generate output path based on original PDF path
            base_name = os.path.splitext(os.path.basename(self.path))[0]
            ext = ".md" if self.content_format == "markdown" else ".json"
            output_path = os.path.join(os.path.dirname(self.path), base_name + ext)

        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if self.content_format == "json":
                    import json

                    json.dump(self.content, f, indent=2, ensure_ascii=False)
                else:
                    f.write(self.content)

            self.logger.info(f"Saved parsed content to {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Error saving content: {str(e)}")
            raise


# Example usage
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Initialize parsing tools
    pdf2md = PDF2MarkdownTool(cleanup=True, extract_images=True)
    pdf2json = PDF2JSONTool(include_metadata=True, extract_references=True)

    # Create and parse a paper with markdown
    paper1 = Paper(path="example_paper.pdf")
    paper1.parse_pdf(pdf2md)
    markdown_content = paper1.get_content()
    paper1.save_parsed_content("example_paper.md")

    # Create and parse a paper with JSON
    paper2 = Paper(path="example_paper.pdf", title="Manual Title Override")
    paper2.parse_pdf(pdf2json)
    json_structure = paper2.get_content()
    paper2.save_parsed_content("example_paper.json")
