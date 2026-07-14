"""Tests for the agent-facing news flow."""

import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

from quantmind.configs import NewsFlowCfg, NewsWindow
from quantmind.flows import news_flow
from quantmind.magic import _introspect_flow_signature
from quantmind.preprocess.news import NewsBatch


class NewsFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_forwards_window_and_retention_to_source_collector(
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
            result = await news_flow(
                window,
                cfg=NewsFlowCfg(retain_raw_html=True),
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
            await news_flow(window)

        self.assertFalse(collect.await_args.kwargs["retain_raw_html"])

    def test_signature_is_magic_input_compatible(self) -> None:
        input_type, cfg_type = _introspect_flow_signature(news_flow)

        self.assertIs(input_type, NewsWindow)
        self.assertIs(cfg_type, NewsFlowCfg)


if __name__ == "__main__":
    unittest.main()
