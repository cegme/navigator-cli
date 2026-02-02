#!/usr/bin/env python3
"""
Simple CLI for querying NavigatorAI LLMs.

Usage:
    uv run python navigator_cli.py "What is 2+2?"
    uv run python navigator_cli.py --cot "If I have 5 apples and buy 3 more, how many do I have?"
    uv run python navigator_cli.py --system "You are a data engineer" "How do I clean CSV data?"
    echo "Explain Python decorators" | uv run python navigator_cli.py --stdin
"""

import argparse
import asyncio
import os
import sys
import time

import requests
from dotenv import load_dotenv
from loguru import logger

# Configure loguru to only show warnings and above by default
logger.remove()
logger.add(sys.stderr, level="WARNING")


# ANSI color codes for terminal output
class Colors:
    """ANSI color codes for terminal output."""

    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"

    @classmethod
    def disable(cls):
        """Disable colors (for non-TTY output)."""
        cls.CYAN = ""
        cls.GREEN = ""
        cls.YELLOW = ""
        cls.DIM = ""
        cls.BOLD = ""
        cls.RESET = ""


# Disable colors if stderr is not a TTY
if not sys.stderr.isatty():
    Colors.disable()


# NavigatorAI API endpoints (OpenAI-compatible)
NAVIGATOR_BASE_URL = "https://api.ai.it.ufl.edu/v1"
NAVIGATOR_API_URL = f"{NAVIGATOR_BASE_URL}/chat/completions"
NAVIGATOR_MODELS_URL = f"{NAVIGATOR_BASE_URL}/models"
DEFAULT_MODEL = "gpt-4o"


def query_llm(
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    use_cot: bool = False,
    temperature: float = 0.7,
) -> str:
    """
    Query the NavigatorAI LLM.

    Args:
        prompt: The user's question or prompt.
        api_key: NavigatorAI API key.
        model: Model to use (default: gpt-4o).
        system_prompt: Optional system prompt to set context.
        use_cot: If True, append "Let's think step by step." to trigger CoT.
        temperature: Sampling temperature (0.0-2.0).

    Returns:
        The LLM's response text.

    Raises:
        requests.RequestException: If the API request fails.
    """
    messages = []

    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Apply Chain-of-Thought if requested
    if use_cot:
        prompt = f"{prompt}\n\nLet's think step by step."

    messages.append({"role": "user", "content": prompt})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    logger.debug(f"Sending request to {NAVIGATOR_API_URL}")
    logger.debug(f"Model: {model}, Temperature: {temperature}")
    logger.debug(f"Messages: {messages}")

    response = requests.post(
        NAVIGATOR_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    return data["choices"][0]["message"]["content"]


def list_models(api_key: str) -> list[dict]:
    """
    List available models from NavigatorAI.

    Args:
        api_key: NavigatorAI API key.

    Returns:
        List of model information dictionaries.

    Raises:
        requests.RequestException: If the API request fails.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    logger.debug(f"Fetching models from {NAVIGATOR_MODELS_URL}")

    response = requests.get(
        NAVIGATOR_MODELS_URL,
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()

    data = response.json()
    return data.get("data", [])


def print_models(models: list[dict]) -> None:
    """
    Print available models in a formatted table.

    Args:
        models: List of model dictionaries from the API.
    """
    if not models:
        print("No models available.")
        return

    print(f"{'Model ID':<40} {'Owner':<20} {'Created':<12}")
    print("-" * 72)

    for model in sorted(models, key=lambda m: m.get("id", "")):
        model_id = model.get("id", "unknown")
        owner = model.get("owned_by", "unknown")
        created = model.get("created", "")
        if created:
            from datetime import datetime
            try:
                created = datetime.fromtimestamp(created).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                created = str(created)
        print(f"{model_id:<40} {owner:<20} {created:<12}")

    print(f"\nTotal: {len(models)} models available")


def main():
    """Main entry point for the CLI."""
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Query NavigatorAI LLMs from the command line.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run python navigator_cli.py "What is the capital of France?"
  uv run python navigator_cli.py --cot "What is 15% of 80?"
  uv run python navigator_cli.py --system "You are a Python expert" "How do I read a CSV?"
  uv run python navigator_cli.py --model gpt-4o-mini "Quick question"
  uv run python navigator_cli.py --list-models
  echo "Explain recursion" | uv run python navigator_cli.py --stdin
        """,
    )

    parser.add_argument(
        "prompt",
        nargs="?",
        help="The prompt to send to the LLM",
    )
    parser.add_argument(
        "--stdin",
        action="store_true",
        help="Read prompt from stdin instead of argument",
    )
    parser.add_argument(
        "--cot",
        action="store_true",
        help="Use Chain-of-Thought prompting (adds 'Let's think step by step.')",
    )
    parser.add_argument(
        "--system", "-s",
        type=str,
        default=None,
        help="System prompt to set context (e.g., 'You are a data engineer')",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--temperature", "-t",
        type=float,
        default=0.7,
        help="Sampling temperature 0.0-2.0 (default: 0.7)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging to stdout",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Only output the LLM response (suppress all other output)",
    )
    parser.add_argument(
        "--list-models", "-l",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--mcp-server",
        type=str,
        default=None,
        help="Path to an MCP server script to enable tool use",
    )

    args = parser.parse_args()

    # Configure logging based on verbosity flags
    if args.quiet:
        # Suppress all logging in quiet mode
        logger.remove()
    elif args.verbose:
        # Verbose mode: DEBUG level to stdout
        logger.remove()
        logger.add(sys.stdout, level="DEBUG")

    # Get API key from environment
    api_key = os.getenv("NAVIGATOR_API_KEY")
    if not api_key:
        logger.error("NAVIGATOR_API_KEY not set. Create a .env file or export it.")
        if not args.quiet:
            print("Error: NAVIGATOR_API_KEY not set.", file=sys.stderr)
            print("Create a .env file with: NAVIGATOR_API_KEY=your-key-here", file=sys.stderr)
            print("Or export it: export NAVIGATOR_API_KEY=your-key-here", file=sys.stderr)
        sys.exit(1)

    # Handle --list-models
    if args.list_models:
        try:
            models = list_models(api_key)
            print_models(models)
            sys.exit(0)
        except requests.exceptions.HTTPError as e:
            logger.error(f"Failed to fetch models: {e}")
            if not args.quiet:
                print(f"Error: Failed to fetch models - {e}", file=sys.stderr)
            sys.exit(1)
        except requests.exceptions.ConnectionError:
            logger.error("Could not connect to NavigatorAI API")
            if not args.quiet:
                print("Error: Could not connect to NavigatorAI API.", file=sys.stderr)
            sys.exit(1)

    # Get prompt from stdin or argument
    if args.stdin:
        prompt = sys.stdin.read().strip()
    elif args.prompt:
        prompt = args.prompt
    else:
        parser.print_help()
        sys.exit(1)

    if not prompt:
        if not args.quiet:
            print("Error: No prompt provided.", file=sys.stderr)
        sys.exit(1)

    try:
        start_time = time.time()

        if args.mcp_server:
            from mcp_client import run_with_tools

            if args.cot:
                prompt = f"{prompt}\n\nLet's think step by step."

            response = asyncio.run(run_with_tools(
                server_script=args.mcp_server,
                prompt=prompt,
                api_key=api_key,
                model=args.model,
                system_prompt=args.system,
                temperature=args.temperature,
            ))
        else:
            response = query_llm(
                prompt=prompt,
                api_key=api_key,
                model=args.model,
                system_prompt=args.system,
                use_cot=args.cot,
                temperature=args.temperature,
            )

        elapsed_time = time.time() - start_time

        # Output model name and time to stderr (unless quiet mode)
        if not args.quiet:
            model_str = f"{Colors.CYAN}{args.model}{Colors.RESET}"
            time_str = f"{Colors.GREEN}{elapsed_time:.2f}s{Colors.RESET}"
            print(f"{Colors.DIM}Model:{Colors.RESET} {model_str} {Colors.DIM}|{Colors.RESET} {Colors.DIM}Time:{Colors.RESET} {time_str}", file=sys.stderr)

        # Log additional info in verbose mode
        if args.verbose:
            logger.info(f"Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
            logger.info(f"Temperature: {args.temperature}")
            if args.system:
                logger.info(f"System prompt: {args.system[:50]}{'...' if len(args.system) > 50 else ''}")
            if args.cot:
                logger.info("Chain-of-Thought enabled")

        print(response)
    except requests.exceptions.HTTPError as e:
        logger.error(f"API request failed: {e}")
        if not args.quiet:
            print(f"Error: API request failed - {e}", file=sys.stderr)
        sys.exit(1)
    except requests.exceptions.ConnectionError:
        logger.error("Could not connect to NavigatorAI API")
        if not args.quiet:
            print("Error: Could not connect to NavigatorAI API.", file=sys.stderr)
            print("Check your internet connection and API endpoint.", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        logger.error(f"Unexpected API response format: {e}")
        if not args.quiet:
            print(f"Error: Unexpected API response format - {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
