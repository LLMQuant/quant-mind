"""Intent-oriented collection flow for public company news."""

from quantmind.configs import NewsFlowCfg, NewsWindow
from quantmind.preprocess.news import (
    NewsArtifact,
    NewsBatch,
    NewsDocument,
    NewsFailure,
)
from quantmind.preprocess.pr_newswire import _collect_pr_newswire

__all__ = [
    "NewsArtifact",
    "NewsBatch",
    "NewsDocument",
    "NewsFailure",
    "news_flow",
]


async def news_flow(
    input: NewsWindow,
    *,
    cfg: NewsFlowCfg | None = None,
) -> NewsBatch:
    """Collect one replayable news window without exposing source mechanics."""
    cfg = cfg or NewsFlowCfg()
    return await _collect_pr_newswire(
        start=input.start,
        end=input.end,
        retain_raw_html=cfg.retain_raw_html,
    )
