"""Example usage of LLM-based research analysis."""

import os
from pathlib import Path

from quantmind.models.paper import Paper
from quantmind.research import PaperAnalyzer, AnalysisConfig


def create_sample_paper() -> Paper:
    """Create a sample paper for testing."""
    
    paper = Paper(
        title="Deep Learning for High-Frequency Trading: A Comprehensive Framework",
        abstract="""
        This paper presents a novel deep learning framework for high-frequency trading 
        that combines LSTM networks with attention mechanisms to predict short-term 
        price movements in equity markets. Our approach leverages real-time market 
        data including price, volume, and order book information to generate trading 
        signals with high accuracy. We evaluate our framework on historical data from 
        the S&P 500 index and demonstrate significant improvements over traditional 
        methods, achieving a Sharpe ratio of 2.1 and annual returns of 15.3%. 
        The framework incorporates risk management strategies and transaction cost 
        considerations to ensure practical applicability in real-world trading scenarios.
        """,
        authors=["John Smith", "Jane Doe", "Bob Johnson"],
        source="arxiv",
        full_text="""
        # Deep Learning for High-Frequency Trading: A Comprehensive Framework

        ## Abstract
        This paper presents a novel deep learning framework for high-frequency trading 
        that combines LSTM networks with attention mechanisms to predict short-term 
        price movements in equity markets.

        ## Introduction
        High-frequency trading (HFT) has become increasingly important in modern 
        financial markets, requiring sophisticated algorithms to process vast amounts 
        of data in real-time. Traditional approaches based on statistical methods 
        often fail to capture the complex patterns present in market data.

        ## Methodology
        Our framework consists of three main components:
        1. Data preprocessing module for handling real-time market data
        2. LSTM-based prediction model with attention mechanisms
        3. Risk management and execution module

        The LSTM network processes sequential market data including:
        - Price movements
        - Volume data
        - Order book information
        - Technical indicators

        ## Results
        We evaluated our framework on historical S&P 500 data from 2018-2023:
        - Sharpe Ratio: 2.1
        - Annual Returns: 15.3%
        - Maximum Drawdown: 8.2%
        - Win Rate: 67.4%

        ## Conclusion
        Our deep learning framework demonstrates significant improvements over 
        traditional HFT methods and provides a practical solution for real-world 
        trading applications.
        """,
        categories=["Machine Learning", "Finance"],
        tags=["deep learning", "trading", "lstm"]
    )
    
    return paper


def main():
    """Main example function."""
    
    print("=== QuantMind LLM Research Analysis Example ===\n")
    
    # Create sample paper
    paper = create_sample_paper()
    print(f"Paper: {paper.title}")
    print(f"Authors: {', '.join(paper.authors)}")
    print(f"Abstract: {paper.abstract[:250]}...\n")
    
    # Configure analysis
    config = AnalysisConfig(
        enable_tag_analysis=True,
        enable_qa_generation=True,
        enable_visual_extraction=False,  # Skipped as requested
        num_questions=5,
        tag_confidence_threshold=0.6,
        llm_model="gpt-4o",
        temperature=0.3,
        max_tokens=4096
    )
    
    # Initialize analyzer
    analyzer = PaperAnalyzer(config)
    
    print("Starting LLM-based analysis...")
    
    # Perform analysis
    analysis = analyzer.analyze_paper(paper)
    
    print(f"\n=== Analysis Results ===")
    print(f"Analysis ID: {analysis.analysis_id}")
    print(f"Duration: {analysis.analysis_duration:.2f} seconds")
    
    # Display tags
    print(f"\n--- LLM-Generated Tags ---")
    if analysis.primary_tags:
        print("Primary Tags:")
        for tag in analysis.primary_tags:
            print(f"  - {tag.tag} ({tag.value}, confidence: {tag.confidence:.2f})")
    
    if analysis.secondary_tags:
        print("\nSecondary Tags:")
        for tag in analysis.secondary_tags:
            print(f"  - {tag.tag} ({tag.value}, confidence: {tag.confidence:.2f})")
    
    # Display Q&A
    print(f"\n--- LLM-Generated Q&A ---")
    if analysis.questions_answers:
        for i, qa in enumerate(analysis.questions_answers, 1):
            print(f"\nQ{i} ({qa.difficulty_level} - {qa.category}):")
            print(f"  {qa.question}")
            print(f"A{i}:")
            print(f"  {qa.answer}")
    
    # Display insights (from Q&A)
    print(f"\n--- Key Insights ---")
    for insight in analysis.key_insights:
        print(f"  - {insight}")
    
    # Save results
    output_dir = Path("analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    analyzer.save_analysis(analysis, output_dir / "sample_paper_analysis")
    print(f"\nResults saved to {output_dir}/")
    
    # Generate and display report
    report = analyzer.generate_analysis_report(analysis)
    print(f"\n=== Generated Report ===")
    print(report[:1000] + "..." if len(report) > 1000 else report)


if __name__ == "__main__":
    main() 