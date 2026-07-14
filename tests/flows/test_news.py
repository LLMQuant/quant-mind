"""Tests for the agent-facing news collection operation."""

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from quantmind.configs import NewsCollectionCfg, NewsWindow
from quantmind.flows import collect_news
from quantmind.magic import _introspect_flow_signature
from quantmind.preprocess import NewsBatch


class CollectNewsTests(unittest.IsolatedAsyncioTestCase):
    async def test_dispatches_pr_newswire_window(
        self,
    ) -> None:
        window = NewsWindow(
            source="pr-newswire",
            start=datetime(2026, 7, 13, tzinfo=timezone.utc),
            end=datetime(2026, 7, 14, tzinfo=timezone.utc),
        )
        expected = NewsBatch(observed_count=3, complete=True)

        with patch(
            "quantmind.flows.news._collect_pr_newswire",
            new=AsyncMock(return_value=expected),
        ) as collect:
            result = await collect_news(
                window,
                cfg=NewsCollectionCfg(retain_raw_html=True),
            )

        self.assertIs(result, expected)
        collect.assert_awaited_once_with(
            start=window.start,
            end=window.end,
            retain_raw_html=True,
        )

    async def test_uses_agent_safe_retention_default(self) -> None:
        window = NewsWindow(
            source="pr-newswire",
            start=datetime(2026, 7, 13, tzinfo=timezone.utc),
            end=datetime(2026, 7, 14, tzinfo=timezone.utc),
        )

        with patch(
            "quantmind.flows.news._collect_pr_newswire",
            new=AsyncMock(return_value=NewsBatch(complete=True)),
        ) as collect:
            await collect_news(window)

        self.assertFalse(collect.await_args.kwargs["retain_raw_html"])

    def test_signature_is_magic_input_compatible(self) -> None:
        input_type, cfg_type = _introspect_flow_signature(collect_news)

        self.assertIs(input_type, NewsWindow)
        self.assertIs(cfg_type, NewsCollectionCfg)


if __name__ == "__main__":
    unittest.main()
