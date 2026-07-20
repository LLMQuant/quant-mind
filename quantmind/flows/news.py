"""Intent-oriented collection operation for public company news."""

from typing_extensions import assert_never

from quantmind.configs import NewsCollectionCfg, NewsWindow
from quantmind.preprocess import NewsBatch
from quantmind.preprocess.pr_newswire import _collect_pr_newswire

__all__ = ["collect_news"]


async def collect_news(
    input: NewsWindow,
    *,
    cfg: NewsCollectionCfg | None = None,
) -> NewsBatch:
    """Collect one replayable window without exposing source mechanics.

    Args:
        input: Source and half-open time window to collect.
        cfg: Options that change the returned collection evidence.

    Returns:
        Successfully collected documents, recoverable failures, and whether
        source discovery proved complete coverage of the requested window.
        Article failures do not make discovery incomplete.
    """
    cfg = cfg or NewsCollectionCfg()
    source = input.source
    if source == "pr-newswire":
        return await _collect_pr_newswire(
            start=input.start,
            end=input.end,
            retain_raw_html=cfg.retain_raw_html,
        )

    assert_never(source)
