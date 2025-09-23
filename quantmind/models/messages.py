# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import logging
import re
import uuid
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

from quantmind.tools import Tool
from quantmind.utils.agentic_ext import (
    encode_image_base64,
    make_image_url,
    parse_json_blob,
)
from quantmind.utils.monitoring import TokenUsage

logger = logging.getLogger(__name__)

STRUCTURED_GENERATION_PROVIDERS = ["cerebras", "fireworks-ai"]


def get_dict_from_nested_dataclasses(obj, ignore_key=None):
    """Convert a nested dataclass to a dictionary."""

    def convert(obj):
        if hasattr(obj, "__dataclass_fields__"):
            return {
                k: convert(v) for k, v in asdict(obj).items() if k != ignore_key
            }
        return obj

    return convert(obj)


@dataclass
class ChatMessageToolCallFunction:
    """Function for a tool call."""

    arguments: Any
    name: str
    description: str | None = None


@dataclass
class ChatMessageToolCall:
    """Tool call for a chat message."""

    function: ChatMessageToolCallFunction
    id: str
    type: str

    def __str__(self) -> str:
        return f"Call: {self.id}: Calling {str(self.function.name)} with arguments: {str(self.function.arguments)}"


class MessageRole(str, Enum):
    """Message role."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL_CALL = "tool-call"
    TOOL_RESPONSE = "tool-response"

    @classmethod
    def roles(cls):
        return [r.value for r in cls]


@dataclass
class ChatMessage:
    """Chat message."""

    role: MessageRole
    content: str | list[dict[str, Any]] | None = None
    tool_calls: list[ChatMessageToolCall] | None = None
    raw: Any | None = None  # Stores the raw output from the API
    token_usage: TokenUsage | None = None

    def model_dump_json(self):
        return json.dumps(
            get_dict_from_nested_dataclasses(self, ignore_key="raw")
        )

    @classmethod
    def from_dict(
        cls,
        data: dict,
        raw: Any | None = None,
        token_usage: TokenUsage | None = None,
    ) -> "ChatMessage":
        if data.get("tool_calls"):
            tool_calls = [
                ChatMessageToolCall(
                    function=ChatMessageToolCallFunction(**tc["function"]),
                    id=tc["id"],
                    type=tc["type"],
                )
                for tc in data["tool_calls"]
            ]
            data["tool_calls"] = tool_calls
        return cls(
            role=data["role"],
            content=data.get("content"),
            tool_calls=data.get("tool_calls"),
            raw=raw,
            token_usage=token_usage,
        )

    def dict(self):
        return get_dict_from_nested_dataclasses(self)

    def render_as_markdown(self) -> str:
        rendered = str(self.content) or ""
        if self.tool_calls:
            rendered += "\n".join(
                [
                    json.dumps(
                        {
                            "tool": tool.function.name,
                            "arguments": tool.function.arguments,
                        }
                    )
                    for tool in self.tool_calls
                ]
            )
        return rendered


def parse_json_if_needed(arguments: str | dict) -> str | dict:
    """Parse a JSON string if needed."""
    if isinstance(arguments, dict):
        return arguments
    else:
        try:
            return json.loads(arguments)
        except Exception:
            return arguments


@dataclass
class ChatMessageToolCallStreamDelta:
    """Represents a streaming delta for tool calls during generation."""

    index: int | None = None
    id: str | None = None
    type: str | None = None
    function: ChatMessageToolCallFunction | None = None


@dataclass
class ChatMessageStreamDelta:
    """Represents a streaming delta for a chat message."""

    content: str | None = None
    tool_calls: list[ChatMessageToolCallStreamDelta] | None = None
    token_usage: TokenUsage | None = None


def agglomerate_stream_deltas(
    stream_deltas: list[ChatMessageStreamDelta],
    role: MessageRole = MessageRole.ASSISTANT,
) -> ChatMessage:
    """Agglomerate a list of stream deltas into a single stream delta.

    Args:
        stream_deltas (`list[ChatMessageStreamDelta]`): List of chat message stream deltas.
        role (`MessageRole`, *optional*): Role of the chat message.

    Returns:
        `ChatMessage`: Agglomerated chat message.
    """
    accumulated_tool_calls: dict[int, ChatMessageToolCallStreamDelta] = {}
    accumulated_content = ""
    total_input_tokens = 0
    total_output_tokens = 0
    for stream_delta in stream_deltas:
        if stream_delta.token_usage:
            total_input_tokens += stream_delta.token_usage.input_tokens
            total_output_tokens += stream_delta.token_usage.output_tokens
        if stream_delta.content:
            accumulated_content += stream_delta.content
        if stream_delta.tool_calls:
            for tool_call_delta in (
                stream_delta.tool_calls
            ):  # ?ormally there should be only one call at a time
                # Extend accumulated_tool_calls list to accommodate the new tool call if needed
                if tool_call_delta.index is not None:
                    if tool_call_delta.index not in accumulated_tool_calls:
                        accumulated_tool_calls[tool_call_delta.index] = (
                            ChatMessageToolCallStreamDelta(
                                id=tool_call_delta.id,
                                type=tool_call_delta.type,
                                function=ChatMessageToolCallFunction(
                                    name="", arguments=""
                                ),
                            )
                        )
                    # Update the tool call at the specific index
                    tool_call = accumulated_tool_calls[tool_call_delta.index]
                    if tool_call_delta.id:
                        tool_call.id = tool_call_delta.id
                    if tool_call_delta.type:
                        tool_call.type = tool_call_delta.type
                    if tool_call_delta.function:
                        if (
                            tool_call_delta.function.name
                            and len(tool_call_delta.function.name) > 0
                        ):
                            tool_call.function.name = (
                                tool_call_delta.function.name
                            )
                        if tool_call_delta.function.arguments:
                            tool_call.function.arguments += (
                                tool_call_delta.function.arguments
                            )
                else:
                    raise ValueError(
                        f"Tool call index is not provided in tool delta: {tool_call_delta}"
                    )

    return ChatMessage(
        role=role,
        content=accumulated_content,
        tool_calls=[
            ChatMessageToolCall(
                function=ChatMessageToolCallFunction(
                    name=tool_call_stream_delta.function.name,
                    arguments=tool_call_stream_delta.function.arguments,
                ),
                id=tool_call_stream_delta.id or "",
                type="function",
            )
            for tool_call_stream_delta in accumulated_tool_calls.values()
            if tool_call_stream_delta.function
        ],
        token_usage=TokenUsage(
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
        ),
    )


tool_role_conversions = {
    MessageRole.TOOL_CALL: MessageRole.ASSISTANT,
    MessageRole.TOOL_RESPONSE: MessageRole.USER,
}


def get_tool_json_schema(tool: Tool) -> dict:
    """Get a JSON schema for a tool."""
    properties = deepcopy(tool.inputs)
    required = []
    for key, value in properties.items():
        if value["type"] == "any":
            value["type"] = "string"
        if not ("nullable" in value and value["nullable"]):
            required.append(key)
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def remove_stop_sequences(content: str, stop_sequences: list[str]) -> str:
    """Remove stop sequences from a content."""
    for stop_seq in stop_sequences:
        if content[-len(stop_seq) :] == stop_seq:
            content = content[: -len(stop_seq)]
    return content


def get_clean_message_list(
    message_list: list[ChatMessage | dict],
    role_conversions: dict[MessageRole, MessageRole] | dict[str, str] = {},
    convert_images_to_image_urls: bool = False,
    flatten_messages_as_text: bool = False,
) -> list[dict[str, Any]]:
    """Get a clean message list.

    Creates a list of messages to give as input to the LLM.
    These messages are dictionaries and chat
    template compatible with transformers LLM chat template.
    Subsequent messages with the same role will be concatenated to a single message.

    Args:
        message_list (`list[ChatMessage | dict]`): List of chat messages. Mixed types are allowed.
        role_conversions (`dict[MessageRole, MessageRole]`, *optional* ): Mapping to convert roles.
        convert_images_to_image_urls (`bool`, default `False`):
            Whether to convert images to imageURLs.
        flatten_messages_as_text (`bool`, default `False`): Whether to flatten messages as text.
    """
    output_message_list: list[dict[str, Any]] = []
    message_list = deepcopy(message_list)  # Avoid modifying the original list
    for message in message_list:
        if isinstance(message, dict):
            message = ChatMessage.from_dict(message)
        role = message.role
        if role not in MessageRole.roles():
            raise ValueError(
                f"Incorrect role {role}, only {MessageRole.roles()} are supported for now."
            )

        if role in role_conversions:
            message.role = role_conversions[role]  # type: ignore
        # encode images if needed
        if isinstance(message.content, list):
            for element in message.content:
                assert isinstance(element, dict), (
                    "Error: this element should be a dict:" + str(element)
                )
                if element["type"] == "image":
                    assert (
                        not flatten_messages_as_text
                    ), f"Cannot use images with {flatten_messages_as_text=}"
                    if convert_images_to_image_urls:
                        element.update(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": make_image_url(
                                        encode_image_base64(
                                            element.pop("image")
                                        )
                                    )
                                },
                            }
                        )
                    else:
                        element["image"] = encode_image_base64(element["image"])

        if (
            len(output_message_list) > 0
            and message.role == output_message_list[-1]["role"]
        ):
            assert isinstance(message.content, list), (
                "Error: wrong content:" + str(message.content)
            )
            if flatten_messages_as_text:
                output_message_list[-1]["content"] += (
                    "\n" + message.content[0]["text"]
                )
            else:
                for el in message.content:
                    if (
                        el["type"] == "text"
                        and output_message_list[-1]["content"][-1]["type"]
                        == "text"
                    ):
                        # Merge consecutive text messages rather than creating new ones
                        output_message_list[-1]["content"][-1]["text"] += (
                            "\n" + el["text"]
                        )
                    else:
                        output_message_list[-1]["content"].append(el)
        else:
            if flatten_messages_as_text:
                content = message.content[0]["text"]
            else:
                content = message.content
            output_message_list.append(
                {
                    "role": message.role,
                    "content": content,
                }
            )
    return output_message_list


def get_tool_call_from_text(
    text: str, tool_name_key: str, tool_arguments_key: str
) -> ChatMessageToolCall:
    """Get a tool call from a text."""
    tool_call_dictionary, _ = parse_json_blob(text)
    try:
        tool_name = tool_call_dictionary[tool_name_key]
    except Exception as e:
        raise ValueError(
            f"Tool call needs to have a key '{tool_name_key}'. Got keys: {list(tool_call_dictionary.keys())} instead"
        ) from e
    tool_arguments = tool_call_dictionary.get(tool_arguments_key, None)
    if isinstance(tool_arguments, str):
        tool_arguments = parse_json_if_needed(tool_arguments)
    return ChatMessageToolCall(
        id=str(uuid.uuid4()),
        type="function",
        function=ChatMessageToolCallFunction(
            name=tool_name, arguments=tool_arguments
        ),
    )


def supports_stop_parameter(model_id: str) -> bool:
    """Check if the model supports the `stop` parameter.

    Not supported with reasoning models openai/o3, openai/o4-mini, and
    the openai/gpt-5 series (and their versioned variants).

    Args:
        model_id (`str`): Model identifier (e.g. "openai/o3", "o4-mini-2025-04-16")

    Returns:
        bool: True if the model supports the stop parameter, False otherwise
    """
    model_name = model_id.split("/")[-1]
    # o3, o4-mini, grok-3-mini, grok-4, grok-code-fast and the gpt-5 series
    # (including versioned variants, o3-2025-04-16) don't support stop parameter
    openai_model_pattern = r"(o3[-\d]*|o4-mini[-\d]*|gpt-5(-mini|-nano)?[-\d]*)"
    grok_model_pattern = (
        r"([a-zA-Z]+\.)?(grok-3-mini|grok-4|grok-code-fast)(-[A-Za-z0-9]*)?"
    )
    pattern = rf"^({openai_model_pattern}|{grok_model_pattern})$"

    return not re.match(pattern, model_name)


class _ParameterRemove:
    """Sentinel value to indicate a parameter should be removed."""

    def __repr__(self):
        return "REMOVE_PARAMETER"


# Singleton instance for removing parameters
REMOVE_PARAMETER = _ParameterRemove()
