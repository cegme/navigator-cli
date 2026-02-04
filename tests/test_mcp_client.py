"""Tests for the MCP client module."""

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_client import (
    _call_navigator_api,
    _extract_mcp_result_text,
    mcp_tools_to_openai_format,
    run_with_tools,
)

MODELS = ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet"]


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


class TestCallNavigatorApiMultiModel:
    """Tests that _call_navigator_api correctly propagates different models."""

    @pytest.mark.parametrize("model", MODELS)
    @patch("mcp_client.requests.post")
    def test_model_passed_in_payload(self, mock_post, model):
        """Test that the specified model appears in the API request payload."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": f"Response from {model}"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello"}]
        result = _call_navigator_api(messages, api_key="test-key", model=model)

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["model"] == model
        assert result["choices"][0]["message"]["content"] == f"Response from {model}"

    @pytest.mark.parametrize("model", MODELS)
    @patch("mcp_client.requests.post")
    def test_model_with_tools_in_payload(self, mock_post, model):
        """Test that model and tools are both sent correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "tool response"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        tools = [{"type": "function", "function": {"name": "read_csv"}}]
        messages = [{"role": "user", "content": "Use tool"}]
        _call_navigator_api(messages, api_key="test-key", model=model, tools=tools)

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["model"] == model
        assert call_kwargs["json"]["tools"] == tools

    @pytest.mark.parametrize("model", MODELS)
    @patch("mcp_client.requests.post")
    def test_model_with_custom_temperature(self, mock_post, model):
        """Test that model and temperature are both sent correctly."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "ok"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "test"}]
        _call_navigator_api(
            messages, api_key="test-key", model=model, temperature=0.3,
        )

        call_kwargs = mock_post.call_args.kwargs
        assert call_kwargs["json"]["model"] == model
        assert call_kwargs["json"]["temperature"] == 0.3


class TestRunWithToolsMultiModel:
    """Tests that run_with_tools propagates models through the tool-calling loop."""

    def _make_mock_session(self, tool_names):
        """Create a mock MCP ClientSession with named tools.

        Args:
            tool_names: List of tool name strings to register.

        Returns:
            A MagicMock configured as an async MCP session.
        """
        tools = []
        for name in tool_names:
            t = MagicMock()
            t.name = name
            t.description = f"Description for {name}"
            t.inputSchema = {"type": "object", "properties": {}}
            tools.append(t)

        mock_tools_result = MagicMock()
        mock_tools_result.tools = tools

        session = AsyncMock()
        session.initialize = AsyncMock()
        session.list_tools = AsyncMock(return_value=mock_tools_result)

        return session

    @pytest.mark.parametrize("model", MODELS)
    @patch("mcp_client._call_navigator_api")
    @patch("mcp_client.stdio_client")
    def test_model_propagated_no_tool_calls(self, mock_stdio, mock_api, model):
        """Test that the model is passed to the API when no tools are called."""
        session = self._make_mock_session(["read_csv"])

        # Configure stdio_client as an async context manager
        mock_read_write = (MagicMock(), MagicMock())
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_read_write)
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

        # Patch ClientSession
        with patch("mcp_client.ClientSession") as mock_cs_cls:
            mock_cs_cls.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_cs_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            # LLM returns a text response (no tool calls)
            mock_api.return_value = {
                "choices": [{"message": {"content": f"Answer from {model}"}}]
            }

            result = asyncio.run(run_with_tools(
                server_script="fake_server.py",
                prompt="What is the data?",
                api_key="test-key",
                model=model,
            ))

        assert result == f"Answer from {model}"
        call_kwargs = mock_api.call_args.kwargs
        assert call_kwargs["model"] == model

    @pytest.mark.parametrize("model", MODELS)
    @patch("mcp_client._call_navigator_api")
    @patch("mcp_client.stdio_client")
    def test_model_propagated_with_tool_calls(self, mock_stdio, mock_api, model):
        """Test model propagation through a tool-calling round trip."""
        session = self._make_mock_session(["csv_stats"])

        # Configure the tool result
        tool_result_block = MagicMock()
        tool_result_block.text = "mean: 42.0"
        tool_result = MagicMock()
        tool_result.content = [tool_result_block]
        session.call_tool = AsyncMock(return_value=tool_result)

        mock_read_write = (MagicMock(), MagicMock())
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_read_write)
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("mcp_client.ClientSession") as mock_cs_cls:
            mock_cs_cls.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_cs_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            # First call: LLM requests a tool call
            # Second call: LLM returns final text
            mock_api.side_effect = [
                {
                    "choices": [{
                        "message": {
                            "content": None,
                            "tool_calls": [{
                                "id": "call_1",
                                "function": {
                                    "name": "csv_stats",
                                    "arguments": json.dumps({"file_path": "data.csv", "column": "score"}),
                                },
                            }],
                        },
                    }],
                },
                {
                    "choices": [{"message": {"content": f"The mean is 42 ({model})"}}],
                },
            ]

            result = asyncio.run(run_with_tools(
                server_script="fake_server.py",
                prompt="What is the average score?",
                api_key="test-key",
                model=model,
            ))

        assert f"The mean is 42 ({model})" == result

        # Both API calls should use the correct model
        assert mock_api.call_count == 2
        for call in mock_api.call_args_list:
            assert call.kwargs["model"] == model

    @pytest.mark.parametrize("model", MODELS)
    @patch("mcp_client._call_navigator_api")
    @patch("mcp_client.stdio_client")
    def test_model_with_system_prompt(self, mock_stdio, mock_api, model):
        """Test model and system prompt are both propagated correctly."""
        session = self._make_mock_session(["read_csv"])

        mock_read_write = (MagicMock(), MagicMock())
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_read_write)
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("mcp_client.ClientSession") as mock_cs_cls:
            mock_cs_cls.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_cs_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            mock_api.return_value = {
                "choices": [{"message": {"content": "Expert answer"}}]
            }

            asyncio.run(run_with_tools(
                server_script="fake_server.py",
                prompt="Analyze data",
                api_key="test-key",
                model=model,
                system_prompt="You are a data analyst",
            ))

        call_kwargs = mock_api.call_args.kwargs
        assert call_kwargs["model"] == model
        messages = call_kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a data analyst"

    @pytest.mark.parametrize("model", MODELS)
    @patch("mcp_client._call_navigator_api")
    @patch("mcp_client.stdio_client")
    def test_model_propagated_at_max_rounds(self, mock_stdio, mock_api, model):
        """Test that the model is used in the final API call after max rounds."""
        session = self._make_mock_session(["count_rows"])

        tool_result_block = MagicMock()
        tool_result_block.text = "5"
        tool_result = MagicMock()
        tool_result.content = [tool_result_block]
        session.call_tool = AsyncMock(return_value=tool_result)

        mock_read_write = (MagicMock(), MagicMock())
        mock_stdio.return_value.__aenter__ = AsyncMock(return_value=mock_read_write)
        mock_stdio.return_value.__aexit__ = AsyncMock(return_value=False)

        with patch("mcp_client.ClientSession") as mock_cs_cls:
            mock_cs_cls.return_value.__aenter__ = AsyncMock(return_value=session)
            mock_cs_cls.return_value.__aexit__ = AsyncMock(return_value=False)

            # Every round requests a tool call, forcing max_tool_rounds exhaustion
            tool_call_response = {
                "choices": [{
                    "message": {
                        "content": None,
                        "tool_calls": [{
                            "id": "call_n",
                            "function": {
                                "name": "count_rows",
                                "arguments": json.dumps({"file_path": "data.csv"}),
                            },
                        }],
                    },
                }],
            }
            final_response = {
                "choices": [{"message": {"content": f"Done ({model})"}}],
            }

            # 2 rounds of tool calls + 1 final call (with max_tool_rounds=2)
            mock_api.side_effect = [
                tool_call_response,
                tool_call_response,
                final_response,
            ]

            result = asyncio.run(run_with_tools(
                server_script="fake_server.py",
                prompt="Count rows",
                api_key="test-key",
                model=model,
                max_tool_rounds=2,
            ))

        assert result == f"Done ({model})"
        # All 3 API calls should use the correct model
        assert mock_api.call_count == 3
        for call in mock_api.call_args_list:
            assert call.kwargs["model"] == model


class TestRunWithToolsCLIMultiModel:
    """Tests that the CLI --mcp-server flag propagates models correctly."""

    @pytest.mark.parametrize("model", MODELS)
    @patch("navigator_cli.asyncio.run")
    @patch("navigator_cli.os.getenv")
    def test_cli_mcp_server_passes_model(self, mock_getenv, mock_asyncio_run, model, capsys):
        """Test that CLI passes the model to run_with_tools via --mcp-server."""
        from navigator_cli import main

        mock_getenv.return_value = "test-api-key"
        mock_asyncio_run.return_value = f"MCP answer from {model}"

        with patch("sys.argv", [
            "navigator_cli.py",
            "--mcp-server", "mcp_servers/csv_tools.py",
            "--model", model,
            "--no-log",
            "What is the average score?",
        ]):
            main()

        # Verify asyncio.run was called and the run_with_tools coroutine
        # received the correct model
        mock_asyncio_run.assert_called_once()
        captured = capsys.readouterr()
        assert f"MCP answer from {model}" in captured.out

    @pytest.mark.parametrize("model", MODELS)
    @patch("navigator_cli.asyncio.run")
    @patch("navigator_cli.os.getenv")
    def test_cli_mcp_with_cot_and_model(self, mock_getenv, mock_asyncio_run, model, capsys):
        """Test that CLI passes model and CoT prompt through MCP path."""
        from navigator_cli import main

        mock_getenv.return_value = "test-api-key"
        mock_asyncio_run.return_value = "Step-by-step answer"

        with patch("sys.argv", [
            "navigator_cli.py",
            "--mcp-server", "mcp_servers/csv_tools.py",
            "--model", model,
            "--cot",
            "--no-log",
            "Analyze the data",
        ]):
            main()

        mock_asyncio_run.assert_called_once()

    @pytest.mark.parametrize("model", MODELS)
    @patch("navigator_cli.asyncio.run")
    @patch("navigator_cli.os.getenv")
    def test_cli_mcp_with_system_prompt_and_model(
        self, mock_getenv, mock_asyncio_run, model, capsys,
    ):
        """Test that CLI passes model and system prompt through MCP path."""
        from navigator_cli import main

        mock_getenv.return_value = "test-api-key"
        mock_asyncio_run.return_value = "Expert analysis"

        with patch("sys.argv", [
            "navigator_cli.py",
            "--mcp-server", "mcp_servers/csv_tools.py",
            "--model", model,
            "--system", "You are a data scientist",
            "--no-log",
            "Summarize the CSV",
        ]):
            main()

        mock_asyncio_run.assert_called_once()
