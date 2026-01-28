# NavigatorAI CLI

A simple command-line tool for querying NavigatorAI LLMs using Python and uv.

## Setup

```bash
# Navigate to this directory
cd lecture-outlines/day7-prompting/navigator-cli

# Copy the environment template and add your API key
cp .env.example .env
# Edit .env and add your NAVIGATOR_API_KEY

# Sync dependencies (uv will create a virtual environment)
uv sync
```

## Get Your API Key

1. Go to https://ai.it.ufl.edu
2. Log in with your GatorLink credentials
3. Navigate to Navigator Toolkit → API Keys
4. Create a new API key and copy it to your `.env` file

## Usage

You can run the CLI in two ways:

```bash
# Direct script execution
uv run python navigator_cli.py "What is the capital of France?"

# As a module (using __main__.py)
uv run python -m navigator_cli "What is the capital of France?"
```

### Basic Query

```bash
uv run python -m navigator_cli "What is the capital of France?"
```

### Chain-of-Thought Prompting

Use `--cot` to automatically append "Let's think step by step.":

```bash
uv run python navigator_cli.py --cot "If I have 5 apples and buy 3 more bags with 4 apples each, how many apples do I have?"
```

### Custom System Prompt

```bash
uv run python navigator_cli.py --system "You are a data engineer" "How do I clean messy CSV data?"
```

### Different Models

```bash
# Use GPT-4o-mini for faster/cheaper responses
uv run python navigator_cli.py --model gpt-4o-mini "Quick question"

# Use Claude
uv run python navigator_cli.py --model claude-3-5-sonnet "Explain Python decorators"
```

### Read from Stdin

```bash
echo "Explain recursion in simple terms" | uv run python navigator_cli.py --stdin

# Or pipe a file
cat prompt.txt | uv run python navigator_cli.py --stdin
```

### Verbose Mode

```bash
uv run python navigator_cli.py -v "Debug this query"
```

## Options

| Option | Short | Description |
|--------|-------|-------------|
| `--cot` | | Enable Chain-of-Thought prompting |
| `--system` | `-s` | Set a system prompt |
| `--model` | `-m` | Choose model (default: gpt-4o) |
| `--temperature` | `-t` | Set temperature 0.0-2.0 (default: 0.7) |
| `--stdin` | | Read prompt from stdin |
| `--verbose` | `-v` | Enable debug logging |

## Examples for Class Demo

### Compare Zero-Shot vs Chain-of-Thought

```bash
# Zero-shot (direct answer)
uv run python navigator_cli.py "Roger has 5 tennis balls. He buys 2 cans with 3 balls each. How many balls does he have?"

# Chain-of-Thought (step by step)
uv run python navigator_cli.py --cot "Roger has 5 tennis balls. He buys 2 cans with 3 balls each. How many balls does he have?"
```

### Data Engineering Example

```bash
uv run python navigator_cli.py --system "You are a data quality analyst" --cot "
Given this record, identify all validation errors:
Schema: email (valid format), age (0-120), date (YYYY-MM-DD, not future)
Record: {\"email\": \"john@\", \"age\": \"25\", \"date\": \"2030-01-15\"}
"
```

## Running Tests

```bash
# Install dev dependencies
uv add --dev pytest

# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run a specific test class
uv run pytest tests/test_navigator_cli.py::TestQueryLLM -v
```

All tests mock the API calls, so no API key is needed to run them.

## Troubleshooting

**Error: NAVIGATOR_API_KEY not set**
- Make sure you have a `.env` file with your API key
- Or export it: `export NAVIGATOR_API_KEY=your-key`

**Error: Could not connect to NavigatorAI API**
- Check your internet connection
- Verify you're on the UF network or VPN

**Error: 401 Unauthorized**
- Your API key may be invalid or expired
- Generate a new key at https://ai.it.ufl.edu
