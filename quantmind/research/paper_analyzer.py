"""Main paper analyzer for comprehensive research analysis."""

import time
from pathlib import Path
from typing import Optional, Union

from quantmind.models.paper import Paper
from quantmind.research.models import PaperAnalysis, AnalysisConfig
from quantmind.research.tag_analyzer import LLMTagAnalyzer
from quantmind.research.qa_generator import LLMQAGenerator
from quantmind.utils.logger import get_logger

logger = get_logger(__name__)


class PaperAnalyzer:
    """Comprehensive paper analyzer that integrates LLM-based research analysis capabilities.

    Provides:
    1. LLM-based hierarchical tag analysis (market, frequency, algorithms, applications)
    2. LLM-based deep Q&A generation with insights
    3. Comprehensive analysis summary
    """

    def __init__(self, config: Optional[AnalysisConfig] = None):
        """Initialize paper analyzer.

        Args:
            config: Analysis configuration
        """
        self.config = config or AnalysisConfig()

        # Initialize LLM-based components
        self.tag_analyzer = LLMTagAnalyzer(self.config)
        self.qa_generator = LLMQAGenerator(self.config)

        logger.info("PaperAnalyzer initialized with LLM-based components")

    def analyze_paper(self, paper: Paper) -> PaperAnalysis:
        """Perform comprehensive analysis of a research paper using LLM.

        Args:
            paper: Paper object to analyze

        Returns:
            Comprehensive analysis results
        """
        logger.info(
            f"Starting comprehensive LLM analysis of paper: {paper.title}"
        )
        start_time = time.time()

        # Initialize analysis result
        analysis = PaperAnalysis(paper_id=paper.get_primary_id())

        try:
            # 1. LLM-based Tag Analysis
            if self.config.enable_tag_analysis:
                logger.info("Performing LLM-based tag analysis...")
                primary_tags, secondary_tags = self.tag_analyzer.analyze_paper(
                    paper
                )
                analysis.primary_tags = primary_tags
                analysis.secondary_tags = secondary_tags

                # Generate tag summary
                tag_summary = self.tag_analyzer.generate_tag_summary(
                    primary_tags, secondary_tags
                )
                analysis.key_insights.append(f"LLM Tag Analysis: {tag_summary}")

            # 2. Visual Element Extraction (SKIPPED - not implemented yet)
            if self.config.enable_visual_extraction:
                logger.info("Visual extraction skipped - not implemented yet")
                analysis.key_insights.append(
                    "Visual extraction: Skipped - not implemented yet"
                )

            # 3. LLM-based Q&A Generation
            if self.config.enable_qa_generation:
                logger.info("Generating LLM-based Q&A pairs...")
                qa_pairs = self.qa_generator.generate_questions_answers(
                    paper, analysis.primary_tags, analysis.secondary_tags
                )
                analysis.questions_answers = qa_pairs

                # Generate Q&A summary
                qa_summary = self.qa_generator.generate_qa_summary(qa_pairs)
                analysis.key_insights.append(
                    f"LLM Q&A Generation: {qa_summary}"
                )

            # 4. Generate methodology and results summaries
            analysis.methodology_summary = self._generate_methodology_summary(
                paper
            )
            analysis.results_summary = self._generate_results_summary(paper)

            # Calculate analysis duration
            analysis.analysis_duration = time.time() - start_time

            logger.info(
                f"LLM analysis completed in {analysis.analysis_duration:.2f} seconds"
            )

        except Exception as e:
            logger.error(f"Error during LLM analysis: {str(e)}")
            analysis.key_insights.append(f"Analysis Error: {str(e)}")
            # Ensure duration is always set
            analysis.analysis_duration = time.time() - start_time

        return analysis

    def analyze_papers(self, papers: list[Paper]) -> list[PaperAnalysis]:
        """Analyze multiple papers using LLM.

        Args:
            papers: List of paper objects to analyze

        Returns:
            List of analysis results
        """
        logger.info(f"Starting LLM analysis of {len(papers)} papers")

        analyses = []
        for i, paper in enumerate(papers, 1):
            logger.info(f"Analyzing paper {i}/{len(papers)}: {paper.title}")
            analysis = self.analyze_paper(paper)
            analyses.append(analysis)

        logger.info(f"Completed LLM analysis of {len(papers)} papers")
        return analyses

    def _generate_methodology_summary(self, paper: Paper) -> Optional[str]:
        """Generate methodology summary from paper content.

        Args:
            paper: Paper object

        Returns:
            Methodology summary
        """
        if not paper.full_text:
            return None

        # Look for methodology section
        text_lower = paper.full_text.lower()

        # Find methodology-related sections
        methodology_keywords = [
            "method",
            "methodology",
            "approach",
            "framework",
            "model",
        ]
        methodology_sections = []

        lines = paper.full_text.split("\n")
        in_methodology = False
        current_section = []

        for line in lines:
            line_lower = line.lower().strip()

            # Check if this is a methodology section header
            if any(keyword in line_lower for keyword in methodology_keywords):
                if (
                    line_lower.startswith(("##", "###", "#"))
                    or line_lower.isupper()
                ):
                    if current_section:
                        methodology_sections.append("\n".join(current_section))
                    current_section = [line]
                    in_methodology = True
                    continue

            if in_methodology:
                # Stop at next major section
                if (
                    line_lower.startswith(("##", "###", "#"))
                    and len(line.strip()) < 100
                ):
                    break
                current_section.append(line)

        if current_section:
            methodology_sections.append("\n".join(current_section))

        if methodology_sections:
            # Combine and summarize
            methodology_text = "\n\n".join(methodology_sections)
            return (
                methodology_text[:500] + "..."
                if len(methodology_text) > 500
                else methodology_text
            )

        return None

    def _generate_results_summary(self, paper: Paper) -> Optional[str]:
        """Generate results summary from paper content.

        Args:
            paper: Paper object

        Returns:
            Results summary
        """
        if not paper.full_text:
            return None

        # Look for results section
        text_lower = paper.full_text.lower()

        # Find results-related sections
        results_keywords = [
            "result",
            "experiment",
            "evaluation",
            "performance",
            "comparison",
        ]
        results_sections = []

        lines = paper.full_text.split("\n")
        in_results = False
        current_section = []

        for line in lines:
            line_lower = line.lower().strip()

            # Check if this is a results section header
            if any(keyword in line_lower for keyword in results_keywords):
                if (
                    line_lower.startswith(("##", "###", "#"))
                    or line_lower.isupper()
                ):
                    if current_section:
                        results_sections.append("\n".join(current_section))
                    current_section = [line]
                    in_results = True
                    continue

            if in_results:
                # Stop at next major section
                if (
                    line_lower.startswith(("##", "###", "#"))
                    and len(line.strip()) < 100
                ):
                    break
                current_section.append(line)

        if current_section:
            results_sections.append("\n".join(current_section))

        if results_sections:
            # Combine and summarize
            results_text = "\n\n".join(results_sections)
            return (
                results_text[:500] + "..."
                if len(results_text) > 500
                else results_text
            )

        return None

    def generate_analysis_report(self, analysis: PaperAnalysis) -> str:
        """Generate a comprehensive analysis report.

        Args:
            analysis: Analysis results

        Returns:
            Formatted analysis report
        """
        report_parts = []

        # Header
        report_parts.append("# Paper Analysis Report (LLM-based)")
        report_parts.append(f"**Paper ID:** {analysis.paper_id}")
        report_parts.append(f"**Analysis ID:** {analysis.analysis_id}")
        report_parts.append(f"**Analysis Date:** {analysis.analysis_timestamp}")
        report_parts.append(
            f"**Analysis Duration:** {analysis.analysis_duration:.2f} seconds"
        )
        report_parts.append("")

        # Key Insights
        if analysis.key_insights:
            report_parts.append("## Key Insights")
            for insight in analysis.key_insights:
                report_parts.append(f"- {insight}")
            report_parts.append("")

        # Tags
        if analysis.primary_tags or analysis.secondary_tags:
            report_parts.append("## LLM-Generated Tags")

            if analysis.primary_tags:
                report_parts.append("### Primary Tags (Level 1)")
                for tag in analysis.primary_tags:
                    report_parts.append(
                        f"- **{tag.tag}** ({tag.value}, confidence: {tag.confidence:.2f})"
                    )

            if analysis.secondary_tags:
                report_parts.append("### Secondary Tags (Level 2)")
                for tag in analysis.secondary_tags:
                    report_parts.append(
                        f"- **{tag.tag}** ({tag.value}, confidence: {tag.confidence:.2f})"
                    )
            report_parts.append("")

        # Q&A
        if analysis.questions_answers:
            report_parts.append("## LLM-Generated Questions & Answers")

            # Group by category
            categories = {}
            for qa in analysis.questions_answers:
                if qa.category not in categories:
                    categories[qa.category] = []
                categories[qa.category].append(qa)

            for category, qa_list in categories.items():
                report_parts.append(f"### {category.title()}")
                for i, qa in enumerate(qa_list, 1):
                    report_parts.append(
                        f"**Q{i} ({qa.difficulty_level}):** {qa.question}"
                    )
                    report_parts.append(f"**A{i}:** {qa.answer}")
                    report_parts.append("")

        # Methodology Summary
        if analysis.methodology_summary:
            report_parts.append("## Methodology Summary")
            report_parts.append(analysis.methodology_summary)
            report_parts.append("")

        # Results Summary
        if analysis.results_summary:
            report_parts.append("## Results Summary")
            report_parts.append(analysis.results_summary)
            report_parts.append("")

        return "\n".join(report_parts)

    def save_analysis(
        self, analysis: PaperAnalysis, output_path: Union[str, Path]
    ) -> None:
        """Save analysis results to file.

        Args:
            analysis: Analysis results
            output_path: Output file path
        """
        import json

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as JSON
        json_path = output_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(analysis.model_dump_json(indent=2))

        # Save as Markdown report
        md_path = output_path.with_suffix(".md")
        report = self.generate_analysis_report(analysis)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(report)

        logger.info(f"Analysis saved to {json_path} and {md_path}")

    def load_analysis(self, file_path: Union[str, Path]) -> PaperAnalysis:
        """Load analysis results from file.

        Args:
            file_path: Path to analysis file

        Returns:
            Loaded analysis results
        """
        import json

        file_path = Path(file_path)

        if file_path.suffix == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return PaperAnalysis(**data)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def get_analysis_statistics(self, analyses: list[PaperAnalysis]) -> dict:
        """Get statistics from multiple analyses.

        Args:
            analyses: List of analysis results

        Returns:
            Statistics dictionary
        """
        if not analyses:
            return {}

        stats = {
            "total_papers": len(analyses),
            "avg_analysis_duration": sum(
                a.analysis_duration or 0 for a in analyses
            )
            / len(analyses),
            "tag_statistics": {},
            "qa_statistics": {},
        }

        # Tag statistics
        all_primary_tags = []
        all_secondary_tags = []
        for analysis in analyses:
            all_primary_tags.extend(analysis.primary_tags)
            all_secondary_tags.extend(analysis.secondary_tags)

        stats["tag_statistics"] = {
            "total_primary_tags": len(all_primary_tags),
            "total_secondary_tags": len(all_secondary_tags),
            "avg_primary_tags_per_paper": len(all_primary_tags) / len(analyses),
            "avg_secondary_tags_per_paper": len(all_secondary_tags)
            / len(analyses),
        }

        # Q&A statistics
        total_qa_pairs = sum(len(a.questions_answers) for a in analyses)
        difficulty_counts = {}
        category_counts = {}

        for analysis in analyses:
            for qa in analysis.questions_answers:
                difficulty_counts[qa.difficulty_level] = (
                    difficulty_counts.get(qa.difficulty_level, 0) + 1
                )
                category_counts[qa.category] = (
                    category_counts.get(qa.category, 0) + 1
                )

        stats["qa_statistics"] = {
            "total_qa_pairs": total_qa_pairs,
            "avg_qa_pairs_per_paper": total_qa_pairs / len(analyses),
            "difficulty_distribution": difficulty_counts,
            "category_distribution": category_counts,
        }

        return stats
