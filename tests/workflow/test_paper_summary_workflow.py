"""Tests for PaperSummaryWorkflow."""

import unittest
from unittest.mock import Mock, patch, MagicMock
from quantmind.models.paper import Paper
from quantmind.workflow.paper_summary_workflow import PaperSummaryWorkflow
from quantmind.config.workflows import PaperSummaryWorkflowConfig


class TestPaperSummaryWorkflow(unittest.TestCase):
    """Test cases for PaperSummaryWorkflow."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = PaperSummaryWorkflowConfig(
            llm_type="openai",
            llm_name="gpt-4o",
            api_key="test-key",
            summary_type="comprehensive",
            max_summary_length=600,
            output_format="narrative"
        )
        
        self.paper = Paper(
            title="Test Paper",
            authors=["Test Author"],
            abstract="This is a test abstract for a quantitative finance paper.",
            categories=["q-fin.TR"],
            tags=["test", "finance"],
            full_text="This is the full text content of the test paper.",
            source="test",
            url="https://test.com"
        )

    def test_workflow_initialization(self):
        """Test workflow initialization."""
        workflow = PaperSummaryWorkflow(self.config)
        self.assertEqual(workflow.config, self.config)
        self.assertEqual(workflow.config.summary_type, "comprehensive")

    def test_build_prompt(self):
        """Test prompt building."""
        workflow = PaperSummaryWorkflow(self.config)
        prompt = workflow.build_prompt(self.paper)
        
        # Check that prompt contains paper information
        self.assertIn("Test Paper", prompt)
        self.assertIn("Test Author", prompt)
        self.assertIn("This is a test abstract", prompt)
        self.assertIn("comprehensive", prompt)
        self.assertIn("600", prompt)

    def test_build_prompt_with_custom_instructions(self):
        """Test prompt building with custom instructions."""
        config = PaperSummaryWorkflowConfig(
            llm_type="openai",
            llm_name="gpt-4o",
            api_key="test-key",
            custom_instructions="Focus on trading strategies"
        )
        
        workflow = PaperSummaryWorkflow(config)
        prompt = workflow.build_prompt(self.paper)
        
        self.assertIn("Focus on trading strategies", prompt)

    @patch('quantmind.workflow.paper_summary_workflow.PaperSummaryWorkflow._call_llm')
    def test_execute_success(self, mock_call_llm):
        """Test successful execution."""
        mock_call_llm.return_value = "This is a test summary of the paper."
        
        workflow = PaperSummaryWorkflow(self.config)
        result = workflow.execute(self.paper)
        
        self.assertFalse(result.get("error"))
        self.assertIn("summary", result)
        self.assertEqual(result["paper_title"], "Test Paper")
        self.assertEqual(result["summary_type"], "comprehensive")

    @patch('quantmind.workflow.paper_summary_workflow.PaperSummaryWorkflow._call_llm')
    def test_execute_no_response(self, mock_call_llm):
        """Test execution with no LLM response."""
        mock_call_llm.return_value = None
        
        workflow = PaperSummaryWorkflow(self.config)
        result = workflow.execute(self.paper)
        
        self.assertTrue(result.get("error"))
        self.assertIn("Failed to generate summary", result["error_message"])

    def test_parse_structured_response(self):
        """Test parsing structured JSON response."""
        workflow = PaperSummaryWorkflow(self.config)
        
        json_response = '''
        {
            "summary": "Test summary",
            "key_findings": "Key findings here",
            "methodology": "Methodology description",
            "results": "Results summary"
        }
        '''
        
        result = workflow._parse_structured_response(json_response)
        
        self.assertEqual(result["summary"], "Test summary")
        self.assertEqual(result["key_findings"], "Key findings here")
        self.assertEqual(result["methodology"], "Methodology description")
        self.assertEqual(result["results"], "Results summary")

    def test_parse_structured_response_with_code_blocks(self):
        """Test parsing structured response with code blocks."""
        workflow = PaperSummaryWorkflow(self.config)
        
        response_with_blocks = '''
        Here is the response:
        ```json
        {
            "summary": "Test summary",
            "key_findings": "Key findings"
        }
        ```
        '''
        
        result = workflow._parse_structured_response(response_with_blocks)
        
        self.assertEqual(result["summary"], "Test summary")
        self.assertEqual(result["key_findings"], "Key findings")

    def test_parse_narrative_response(self):
        """Test parsing narrative response."""
        workflow = PaperSummaryWorkflow(self.config)
        
        narrative_response = """
        This is the main summary.
        
        Key Findings: These are the key findings.
        
        Methodology: This describes the methodology.
        
        Results: These are the results.
        """
        
        result = workflow._parse_narrative_response(narrative_response)
        
        self.assertIn("main summary", result["summary"])
        self.assertIn("key findings", result["key_findings"])
        self.assertIn("methodology", result["methodology"])
        self.assertIn("results", result["results"])

    def test_parse_bullet_points_response(self):
        """Test parsing bullet points response."""
        workflow = PaperSummaryWorkflow(self.config)
        
        bullet_response = """
        Main summary here.
        
        - First bullet point
        - Second bullet point
        • Third bullet point
        * Fourth bullet point
        """
        
        result = workflow._parse_bullet_points_response(bullet_response)
        
        self.assertIn("Main summary", result["summary"])
        self.assertIn("First bullet point", result["bullet_points"])
        self.assertIn("Second bullet point", result["bullet_points"])
        self.assertIn("Third bullet point", result["bullet_points"])
        self.assertIn("Fourth bullet point", result["bullet_points"])

    def test_create_error_result(self):
        """Test error result creation."""
        workflow = PaperSummaryWorkflow(self.config)
        
        error_result = workflow._create_error_result("Test error message")
        
        self.assertTrue(error_result["error"])
        self.assertEqual(error_result["error_message"], "Test error message")
        self.assertEqual(error_result["summary"], "")

    @patch('quantmind.workflow.paper_summary_workflow.PaperSummaryWorkflow._call_llm')
    def test_generate_brief_summary(self, mock_call_llm):
        """Test brief summary generation."""
        mock_call_llm.return_value = "Brief summary of the paper."
        
        workflow = PaperSummaryWorkflow(self.config)
        brief_summary = workflow.generate_brief_summary(self.paper)
        
        self.assertEqual(brief_summary, "Brief summary of the paper.")

    @patch('quantmind.workflow.paper_summary_workflow.PaperSummaryWorkflow._call_llm')
    def test_generate_executive_summary(self, mock_call_llm):
        """Test executive summary generation."""
        mock_call_llm.return_value = "Executive summary of the paper."
        
        workflow = PaperSummaryWorkflow(self.config)
        exec_summary = workflow.generate_executive_summary(self.paper)
        
        self.assertFalse(exec_summary.get("error"))
        self.assertIn("summary", exec_summary)

    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid config
        valid_config = PaperSummaryWorkflowConfig(
            llm_type="openai",
            llm_name="gpt-4o",
            api_key="test-key",
            summary_type="comprehensive",
            max_summary_length=600
        )
        
        workflow = PaperSummaryWorkflow(valid_config)
        self.assertEqual(workflow.config.summary_type, "comprehensive")
        
        # Test invalid summary type (should still work as it's just a string)
        config_with_custom_type = PaperSummaryWorkflowConfig(
            llm_type="openai",
            llm_name="gpt-4o",
            api_key="test-key",
            summary_type="custom_type",
            max_summary_length=600
        )
        
        workflow = PaperSummaryWorkflow(config_with_custom_type)
        self.assertEqual(workflow.config.summary_type, "custom_type")

    def test_paper_with_minimal_content(self):
        """Test workflow with paper having minimal content."""
        minimal_paper = Paper(
            title="Minimal Paper",
            abstract="Short abstract",
            source="test"
        )
        
        workflow = PaperSummaryWorkflow(self.config)
        prompt = workflow.build_prompt(minimal_paper)
        
        self.assertIn("Minimal Paper", prompt)
        self.assertIn("Short abstract", prompt)
        self.assertIn("Unknown", prompt)  # For missing authors

    def test_output_format_handling(self):
        """Test different output format handling."""
        config_structured = PaperSummaryWorkflowConfig(
            llm_type="openai",
            llm_name="gpt-4o",
            api_key="test-key",
            output_format="structured"
        )
        
        config_narrative = PaperSummaryWorkflowConfig(
            llm_type="openai",
            llm_name="gpt-4o",
            api_key="test-key",
            output_format="narrative"
        )
        
        config_bullet = PaperSummaryWorkflowConfig(
            llm_type="openai",
            llm_name="gpt-4o",
            api_key="test-key",
            output_format="bullet_points"
        )
        
        workflow_structured = PaperSummaryWorkflow(config_structured)
        workflow_narrative = PaperSummaryWorkflow(config_narrative)
        workflow_bullet = PaperSummaryWorkflow(config_bullet)
        
        self.assertEqual(workflow_structured.config.output_format, "structured")
        self.assertEqual(workflow_narrative.config.output_format, "narrative")
        self.assertEqual(workflow_bullet.config.output_format, "bullet_points")


if __name__ == "__main__":
    unittest.main() 