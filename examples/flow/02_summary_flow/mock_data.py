"""Mock financial research papers for demonstration."""

from quantmind.models.content import KnowledgeItem


def get_sample_papers():
    """Get sample financial research papers for testing."""
    # Large paper that will be chunked
    large_paper = KnowledgeItem(
        title="Machine Learning Applications in Algorithmic Trading: A Comprehensive Study",
        abstract=(
            "This paper presents a comprehensive analysis of machine learning techniques "
            "applied to algorithmic trading strategies. We evaluate various ML models "
            "including random forests, gradient boosting, and deep neural networks on "
            "high-frequency trading data from major equity markets."
        ),
        content=(
            "Introduction:\n"
            "The financial markets have undergone significant transformation with the advent "
            "of algorithmic trading. Machine learning techniques offer unprecedented "
            "opportunities to identify complex patterns in market data that traditional "
            "statistical methods might miss. This study focuses on the practical application "
            "of ML algorithms in creating profitable trading strategies.\n\n"
            "Literature Review:\n"
            "Previous research in this domain has shown mixed results. Early studies by "
            "Johnson et al. (2018) demonstrated modest improvements using linear models, "
            "while more recent work by Chen and Liu (2021) achieved breakthrough results "
            "with deep learning architectures. However, most studies lack comprehensive "
            "evaluation across different market conditions and asset classes.\n\n"
            "Methodology:\n"
            "Our experimental setup involves training multiple ML models on 5 years of "
            "high-frequency data from S&P 500 constituents. We employ a rolling window "
            "approach with 252-day training periods and 63-day testing periods. Feature "
            "engineering includes technical indicators (RSI, MACD, Bollinger Bands), "
            "fundamental metrics (P/E ratios, earnings growth), and market microstructure "
            "variables (bid-ask spreads, order book depth).\n\n"
            "Model Architecture:\n"
            "We implement three primary model types: (1) Random Forest with 1000 trees "
            "and maximum depth of 10, (2) Gradient Boosting with learning rate 0.01 and "
            "500 estimators, and (3) LSTM neural network with 128 hidden units and dropout "
            "regularization. Each model is optimized using grid search cross-validation.\n\n"
            "Results:\n"
            "The experimental results demonstrate significant improvements over baseline "
            "strategies. The ensemble approach combining all three models achieved 67% "
            "directional accuracy compared to 52% for random walk baseline. Risk-adjusted "
            "returns measured by Sharpe ratio improved from 1.2 to 1.8. Maximum drawdown "
            "was reduced from 15% to 8%, indicating better risk management.\n\n"
            "Statistical Analysis:\n"
            "We conducted rigorous statistical testing to ensure result significance. "
            "Bootstrap confidence intervals show 95% confidence that true accuracy lies "
            "between 64-70%. Paired t-tests confirm statistical significance (p < 0.001) "
            "for performance improvements. Out-of-sample testing on 2023 data validates "
            "model robustness across different market regimes.\n\n"
            "Risk Management:\n"
            "Implementation includes comprehensive risk controls: position sizing based on "
            "volatility targeting, correlation limits to prevent over-concentration, and "
            "dynamic stop-loss levels adjusted for market volatility. These controls "
            "contributed significantly to the improved risk-adjusted performance.\n\n"
            "Conclusion:\n"
            "This study demonstrates that machine learning techniques can significantly "
            "enhance algorithmic trading performance when properly implemented with "
            "appropriate risk controls. The key success factors include comprehensive "
            "feature engineering, ensemble modeling approaches, and robust validation "
            "methodologies. Future research should explore alternative data sources "
            "and investigate model interpretability for regulatory compliance."
        ),
        authors=["Dr. Sarah Chen", "Prof. Michael Rodriguez", "Dr. Alex Kim"],
        categories=["q-fin.TR", "q-fin.ST", "cs.AI"],
        tags=[
            "machine learning",
            "algorithmic trading",
            "quantitative finance",
            "risk management",
        ],
        source="demo_data",
    )

    # Small paper that won't be chunked
    small_paper = KnowledgeItem(
        title="High-Frequency Trading Impact on Market Liquidity",
        abstract=(
            "This study examines the impact of high-frequency trading on market liquidity "
            "using transaction-level data from NYSE and NASDAQ."
        ),
        content=(
            "This research analyzes high-frequency trading effects on market quality metrics. "
            "Using millisecond-level data, we find that HFT improves bid-ask spreads by "
            "12% on average but increases volatility during stress periods. The net effect "
            "on market welfare depends on trading volume and market conditions."
        ),
        authors=["Dr. Jennifer Wang"],
        categories=["q-fin.TR"],
        tags=[
            "high-frequency trading",
            "market liquidity",
            "market microstructure",
        ],
        source="demo_data",
    )

    return [large_paper, small_paper]


def get_custom_chunking_example():
    """Get example with custom chunking strategy."""

    def paragraph_chunker(text: str):
        """Custom chunker that splits by paragraphs."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        return paragraphs

    paper = KnowledgeItem(
        title="Custom Chunking Strategy Demo",
        abstract="Demonstrates paragraph-based chunking instead of size-based.",
        content=(
            "First paragraph discusses the introduction to the topic.\n\n"
            "Second paragraph covers the methodology used in the research.\n\n"
            "Third paragraph presents the main results and findings.\n\n"
            "Fourth paragraph concludes with implications and future work."
        ),
        authors=["Demo Author"],
        categories=["demo"],
        tags=["custom chunking"],
        source="demo_data",
    )

    return paper, paragraph_chunker
