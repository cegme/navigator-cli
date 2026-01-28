"""Tests for navigator_cli with mocked API calls."""

import pytest
from unittest.mock import patch, MagicMock

from navigator_cli import query_llm, list_models, print_models, main


class TestQueryLLM:
    """Tests for the query_llm function."""

    @patch("navigator_cli.requests.post")
    def test_basic_query(self, mock_post):
        """Test a basic query returns the expected response."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Paris is the capital of France."}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = query_llm(
            prompt="What is the capital of France?",
            api_key="test-api-key",
        )

        assert result == "Paris is the capital of France."
        mock_post.assert_called_once()

    @patch("navigator_cli.requests.post")
    def test_query_with_system_prompt(self, mock_post):
        """Test that system prompt is included in the request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Use pandas.read_csv()"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = query_llm(
            prompt="How do I read a CSV?",
            api_key="test-api-key",
            system_prompt="You are a Python expert.",
        )

        assert result == "Use pandas.read_csv()"

        # Verify system prompt was included in the request
        call_args = mock_post.call_args
        messages = call_args.kwargs["json"]["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a Python expert."
        assert messages[1]["role"] == "user"

    @patch("navigator_cli.requests.post")
    def test_chain_of_thought_prompting(self, mock_post):
        """Test that --cot appends 'Let's think step by step.'"""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Step 1: Start with 5 balls..."}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        result = query_llm(
            prompt="How many balls does Roger have?",
            api_key="test-api-key",
            use_cot=True,
        )

        # Verify CoT phrase was appended
        call_args = mock_post.call_args
        messages = call_args.kwargs["json"]["messages"]
        user_message = messages[0]["content"]
        assert "Let's think step by step." in user_message

    @patch("navigator_cli.requests.post")
    def test_custom_model(self, mock_post):
        """Test that custom model is passed to the API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response from mini model"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        query_llm(
            prompt="Quick question",
            api_key="test-api-key",
            model="gpt-4o-mini",
        )

        call_args = mock_post.call_args
        assert call_args.kwargs["json"]["model"] == "gpt-4o-mini"

    @patch("navigator_cli.requests.post")
    def test_custom_temperature(self, mock_post):
        """Test that custom temperature is passed to the API."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Creative response"}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        query_llm(
            prompt="Be creative",
            api_key="test-api-key",
            temperature=1.5,
        )

        call_args = mock_post.call_args
        assert call_args.kwargs["json"]["temperature"] == 1.5

    @patch("navigator_cli.requests.post")
    def test_api_error_handling(self, mock_post):
        """Test that HTTP errors are raised properly."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "401 Unauthorized"
        )
        mock_post.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            query_llm(
                prompt="This should fail",
                api_key="invalid-key",
            )


class TestMainCLI:
    """Tests for the main CLI entry point."""

    @patch("navigator_cli.query_llm")
    @patch("navigator_cli.os.getenv")
    def test_main_with_prompt_argument(self, mock_getenv, mock_query, capsys):
        """Test CLI with prompt as argument."""
        mock_getenv.return_value = "test-api-key"
        mock_query.return_value = "The answer is 42."

        with patch("sys.argv", ["navigator_cli.py", "What is the meaning of life?"]):
            main()

        captured = capsys.readouterr()
        assert "The answer is 42." in captured.out

    @patch("navigator_cli.query_llm")
    @patch("navigator_cli.os.getenv")
    def test_main_with_cot_flag(self, mock_getenv, mock_query, capsys):
        """Test CLI with --cot flag."""
        mock_getenv.return_value = "test-api-key"
        mock_query.return_value = "Step by step answer."

        with patch("sys.argv", ["navigator_cli.py", "--cot", "Math problem"]):
            main()

        mock_query.assert_called_once()
        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["use_cot"] is True

    @patch("navigator_cli.query_llm")
    @patch("navigator_cli.os.getenv")
    def test_main_with_system_prompt(self, mock_getenv, mock_query, capsys):
        """Test CLI with --system flag."""
        mock_getenv.return_value = "test-api-key"
        mock_query.return_value = "Expert response."

        with patch(
            "sys.argv",
            ["navigator_cli.py", "--system", "You are an expert", "Question"],
        ):
            main()

        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["system_prompt"] == "You are an expert"

    @patch("navigator_cli.os.getenv")
    def test_main_missing_api_key(self, mock_getenv, capsys):
        """Test CLI exits with error when API key is missing."""
        mock_getenv.return_value = None

        with patch("sys.argv", ["navigator_cli.py", "Some prompt"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "NAVIGATOR_API_KEY" in captured.err

    @patch("navigator_cli.query_llm")
    @patch("navigator_cli.os.getenv")
    def test_main_with_stdin(self, mock_getenv, mock_query, capsys, monkeypatch):
        """Test CLI reading from stdin."""
        mock_getenv.return_value = "test-api-key"
        mock_query.return_value = "Response to stdin input."

        # Mock stdin
        monkeypatch.setattr("sys.stdin.read", lambda: "Prompt from stdin\n")

        with patch("sys.argv", ["navigator_cli.py", "--stdin"]):
            main()

        call_kwargs = mock_query.call_args.kwargs
        assert call_kwargs["prompt"] == "Prompt from stdin"

    @patch("navigator_cli.os.getenv")
    def test_main_no_prompt_shows_help(self, mock_getenv, capsys):
        """Test CLI with no prompt shows help and exits."""
        mock_getenv.return_value = "test-api-key"

        with patch("sys.argv", ["navigator_cli.py"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1


class TestChainOfThoughtBehavior:
    """Tests specifically for Chain-of-Thought prompting behavior."""

    @patch("navigator_cli.requests.post")
    def test_cot_prompt_format(self, mock_post):
        """Verify exact format of CoT prompt."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Reasoning..."}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        original_prompt = "What is 5 + 3?"
        query_llm(
            prompt=original_prompt,
            api_key="test-key",
            use_cot=True,
        )

        call_args = mock_post.call_args
        sent_prompt = call_args.kwargs["json"]["messages"][0]["content"]

        # Verify the original prompt is preserved
        assert original_prompt in sent_prompt
        # Verify CoT phrase is at the end
        assert sent_prompt.endswith("Let's think step by step.")

    @patch("navigator_cli.requests.post")
    def test_cot_with_system_prompt(self, mock_post):
        """Test CoT works correctly with system prompt."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Analysis..."}}]
        }
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response

        query_llm(
            prompt="Analyze this data",
            api_key="test-key",
            system_prompt="You are a data analyst",
            use_cot=True,
        )

        call_args = mock_post.call_args
        messages = call_args.kwargs["json"]["messages"]

        # System prompt should be first
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are a data analyst"

        # User prompt should have CoT appended
        assert messages[1]["role"] == "user"
        assert "Let's think step by step." in messages[1]["content"]


class TestListModels:
    """Tests for the list_models function."""

    @patch("navigator_cli.requests.get")
    def test_list_models_returns_model_list(self, mock_get):
        """Test that list_models returns a list of models."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": [
                {"id": "gpt-4o", "owned_by": "openai", "created": 1700000000},
                {"id": "gpt-4o-mini", "owned_by": "openai", "created": 1700000001},
                {"id": "claude-3-5-sonnet", "owned_by": "anthropic", "created": 1700000002},
            ]
        }
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = list_models(api_key="test-api-key")

        assert len(models) == 3
        assert models[0]["id"] == "gpt-4o"
        assert models[1]["id"] == "gpt-4o-mini"
        assert models[2]["id"] == "claude-3-5-sonnet"
        mock_get.assert_called_once()

    @patch("navigator_cli.requests.get")
    def test_list_models_empty_response(self, mock_get):
        """Test that list_models handles empty model list."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        models = list_models(api_key="test-api-key")

        assert models == []

    @patch("navigator_cli.requests.get")
    def test_list_models_api_error(self, mock_get):
        """Test that list_models raises on API error."""
        import requests

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            "401 Unauthorized"
        )
        mock_get.return_value = mock_response

        with pytest.raises(requests.exceptions.HTTPError):
            list_models(api_key="invalid-key")

    @patch("navigator_cli.requests.get")
    def test_list_models_uses_correct_endpoint(self, mock_get):
        """Test that list_models calls the correct API endpoint."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": []}
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        list_models(api_key="test-api-key")

        call_args = mock_get.call_args
        assert "models" in call_args.args[0]
        assert "Bearer test-api-key" in call_args.kwargs["headers"]["Authorization"]


class TestPrintModels:
    """Tests for the print_models function."""

    def test_print_models_formats_output(self, capsys):
        """Test that print_models formats output correctly."""
        models = [
            {"id": "gpt-4o", "owned_by": "openai", "created": 1700000000},
            {"id": "claude-3-5-sonnet", "owned_by": "anthropic", "created": 1700000001},
        ]

        print_models(models)

        captured = capsys.readouterr()
        assert "gpt-4o" in captured.out
        assert "claude-3-5-sonnet" in captured.out
        assert "openai" in captured.out
        assert "anthropic" in captured.out
        assert "Total: 2 models" in captured.out

    def test_print_models_empty_list(self, capsys):
        """Test that print_models handles empty list."""
        print_models([])

        captured = capsys.readouterr()
        assert "No models available" in captured.out

    def test_print_models_missing_fields(self, capsys):
        """Test that print_models handles missing fields gracefully."""
        models = [
            {"id": "model-1"},  # Missing owned_by and created
            {"id": "model-2", "owned_by": "test"},  # Missing created
        ]

        print_models(models)

        captured = capsys.readouterr()
        assert "model-1" in captured.out
        assert "model-2" in captured.out
        assert "Total: 2 models" in captured.out


class TestListModelsCLI:
    """Tests for --list-models CLI flag."""

    @patch("navigator_cli.list_models")
    @patch("navigator_cli.os.getenv")
    def test_list_models_flag(self, mock_getenv, mock_list, capsys):
        """Test CLI with --list-models flag."""
        mock_getenv.return_value = "test-api-key"
        mock_list.return_value = [
            {"id": "gpt-4o", "owned_by": "openai", "created": 1700000000},
        ]

        with patch("sys.argv", ["navigator_cli.py", "--list-models"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        # Should exit with 0 (success)
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "gpt-4o" in captured.out

    @patch("navigator_cli.list_models")
    @patch("navigator_cli.os.getenv")
    def test_list_models_short_flag(self, mock_getenv, mock_list, capsys):
        """Test CLI with -l short flag."""
        mock_getenv.return_value = "test-api-key"
        mock_list.return_value = [
            {"id": "gpt-4o-mini", "owned_by": "openai", "created": 1700000000},
        ]

        with patch("sys.argv", ["navigator_cli.py", "-l"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "gpt-4o-mini" in captured.out

    @patch("navigator_cli.list_models")
    @patch("navigator_cli.os.getenv")
    def test_list_models_api_failure(self, mock_getenv, mock_list, capsys):
        """Test CLI handles API failure for --list-models."""
        import requests

        mock_getenv.return_value = "test-api-key"
        mock_list.side_effect = requests.exceptions.HTTPError("500 Server Error")

        with patch("sys.argv", ["navigator_cli.py", "--list-models"]):
            with pytest.raises(SystemExit) as exc_info:
                main()

        assert exc_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error" in captured.err
