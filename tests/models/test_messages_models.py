import unittest

from quantmind.models.messages import (
    ChatMessage,
    ChatMessageStreamDelta,
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    ChatMessageToolCallStreamDelta,
    MessageRole,
    agglomerate_stream_deltas,
    get_clean_message_list,
)
from quantmind.utils.monitoring import TokenUsage


class MessagesModelsTestCase(unittest.TestCase):
    """Test messages models functionality."""

    def test_chat_message_from_dict_parses_tool_calls(self):
        data = {
            "role": MessageRole.ASSISTANT,
            "content": "Computation done",
            "tool_calls": [
                {
                    "function": {"name": "compute", "arguments": {"x": 1}},
                    "id": "call-1",
                    "type": "function",
                }
            ],
        }
        msg = ChatMessage.from_dict(data)

        self.assertIsInstance(msg.tool_calls[0], ChatMessageToolCall)
        self.assertEqual(msg.tool_calls[0].function.name, "compute")
        self.assertEqual(msg.tool_calls[0].function.arguments, {"x": 1})

    def test_agglomerate_stream_deltas_merges_content_and_tokens(self):
        deltas = [
            ChatMessageStreamDelta(
                content="First chunk ",
                token_usage=TokenUsage(input_tokens=1, output_tokens=2),
            ),
            ChatMessageStreamDelta(
                content="second chunk",
                token_usage=TokenUsage(input_tokens=3, output_tokens=1),
            ),
            ChatMessageStreamDelta(
                tool_calls=[
                    ChatMessageToolCallStreamDelta(
                        index=0,
                        id="call-2",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name="analyse",
                            arguments='{"param": ',
                        ),
                    )
                ]
            ),
            ChatMessageStreamDelta(
                tool_calls=[
                    ChatMessageToolCallStreamDelta(
                        index=0,
                        function=ChatMessageToolCallFunction(
                            name="",
                            arguments='"value"}',
                        ),
                    )
                ]
            ),
        ]

        message = agglomerate_stream_deltas(deltas)

        self.assertEqual(message.content, "First chunk second chunk")
        self.assertEqual(message.token_usage.total_tokens, 7)
        self.assertEqual(len(message.tool_calls), 1)
        self.assertEqual(message.tool_calls[0].id, "call-2")
        self.assertEqual(message.tool_calls[0].function.name, "analyse")
        self.assertEqual(
            message.tool_calls[0].function.arguments, '{"param": "value"}'
        )

    def test_get_clean_message_list_merges_consecutive_roles(self):
        messages = [
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": "Line one"}],
            ),
            ChatMessage(
                role=MessageRole.USER,
                content=[{"type": "text", "text": "Line two"}],
            ),
        ]

        result = get_clean_message_list(messages)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["role"], MessageRole.USER)
        self.assertEqual(result[0]["content"][0]["text"], "Line one\nLine two")


if __name__ == "__main__":
    unittest.main()
