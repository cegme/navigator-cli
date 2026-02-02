"""Tests for the MCP client module."""

import json
from unittest.mock import MagicMock, patch

import pytest

from mcp_client import (
    _call_navigator_api,
    _extract_mcp_result_text,
    mcp_tools_to_openai_format,
)


class TestMcpToolsToOpenaiFormat:
    """Tests for converting MCP tool schemas to OpenAI format."""

    def test_converts_single_tool(self):
        """Test conversion of a single MCP tool definition."""
        mock_tool = MagicMock()
        mock_tool.name = "read_csv_file"
        mock_tool.description = "Read a CSV file."
        mock_tool.inputSchema = {
            "type": "object",
            "properties": {"file_path": {"type": "string"}},
            "required": ["file_path"],
        }

        result = mcp_tools_to_openai_format([mock_tool])

        assert len(result) == 1
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "read_csv_file"
        assert result[0]["function"]["description"] == "Read a CSV file."
        assert "file_path" in result[0]["function"]["parameters"]["properties"]

    def test_converts_multiple_tools(self):
        """Test conversion of multiple MCP tool definitions."""
        tools = []
        for name in ["tool_a", "tool_b", "tool_c"]:
            t = MagicMock()
            t.name = name
            t.description = f"Description for {name}"
            t.inputSchema = {"type": "object", "properties": {}}
            tools.append(t)

        result = mcp_tools_to_openai_format(tools)

        assert len(result) == 3
        names = [r["function"]["name"] for r in result]
        assert names == ["tool_a", "tool_b", "tool_c"]

    def test_handles_missing_schema(self):
        """Test that a tool with no inputSchema gets a default schema."""
        mock_tool = MagicMock()
        mock_tool.name = "simple_tool"
        mock_tool.description = "A simple tool."
        mock_tool.inputSchema = None

        result = mcp_tools_to_openai_format([mock_tool])

        assert result[0]["function"]["parameters"] == {
            "type": "object",
            "properties": {},
        }

    def test_handles_missing_description(self):
        """Test that a tool with no description gets an empty string."""
        mock_tool = MagicMock()
        mock_tool.name = "no_desc"
        mock_tool.description = None
        mock_tool.inputSchema = {"type": "object", "properties": {}}

        result = mcp_tools_to_openai_format([mock_tool])

        assert result[0]["function"]["description"] == ""


class TestCallNavigatorApi:
    """Tests for the _call_navigator_api helper."""

    @patch("mcp_client.requests.post")
    def test_basic_call_without_tools(self, mock_post):
        """Test a basic API call with no tools."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hi"}]
        result = _call_navigator_api(messages, api_key="test-key")

        assert result["choices"][0]["message"]["content"] == "Hello"
        call_kwargs = mock_post.call_args.kwargs
        assert "tools" not in call_kwargs["json"]

    @patch("mcp_client.requests.post")
    def test_call_with_tools(self, mock_post):
        """Test that tools are included in the payload when provided."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Using tool"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        messages = [{"role": "user", "content": "Use the tool"}]
        _call_navigator_api(messages, api_key="test-key", tools=tools)

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["tools"] == tools

    @patch("mcp_client.requests.post")
    def test_uses_bearer_auth(self, mock_post):
        """Test that the API key is sent as a Bearer token."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": ""}}]}
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        _call_navigator_api(
            [{"role": "user", "content": "test"}],
            api_key="my-secret-key",
        )

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["headers"]["Authorization"] == "Bearer my-secret-key"

    @patch("mcp_client.requests.post")
    def test_api_error_propagates(self, mock_post):
        """Test that HTTP errors are raised."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = (
            requests.exceptions.HTTPError("500 Server Error")
        )
        mock_post.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            _call_navigator_api(
                [{"role": "user", "content": "fail"}],
                api_key="test-key",
            )


class TestExtractMcpResultText:
    """Tests for the _extract_mcp_result_text helper."""

    def test_extracts_text_content(self):
        """Test extracting text from a result with text blocks."""
        block = MagicMock()
        block.text = "The file has 5 rows."

        result = MagicMock()
        result.content = [block]

        assert _extract_mcp_result_text(result) == "The file has 5 rows."

    def test_joins_multiple_text_blocks(self):
        """Test that multiple text blocks are joined with newlines."""
        block1 = MagicMock()
        block1.text = "Line 1"
        block2 = MagicMock()
        block2.text = "Line 2"

        result = MagicMock()
        result.content = [block1, block2]

        assert _extract_mcp_result_text(result) == "Line 1\nLine 2"

    def test_falls_back_to_str_for_non_text(self):
        """Test fallback when content blocks have no text attribute."""
        block = MagicMock(spec=[])  # No attributes
        result = MagicMock()
        result.content = [block]

        output = _extract_mcp_result_text(result)
        assert isinstance(output, str)
