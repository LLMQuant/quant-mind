# tests/test_pdf2markdown.py
import unittest
from pdfparse import PDF2MarkdownTool
import os
import tempfile


class TestPDF2MarkdownTool(unittest.TestCase):
    def setUp(self):
        self.tool = PDF2MarkdownTool()
        self.test_pdf = "tests/data/test.pdf"  # 需要准备测试用的PDF文件
        self.temp_dir = tempfile.mkdtemp()

    def test_convert_pdf(self):
        result = self.tool.parse(self.test_pdf)
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)

    def test_cleanup(self):
        result = self.tool.parse(self.test_pdf)
        self.assertNotIn("\n\n\n", result)  # 检查多余空行

    def test_output_dir(self):
        tool = PDF2MarkdownTool(output_dir=self.temp_dir)
        result = tool.parse(self.test_pdf)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "test.md")))
