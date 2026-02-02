"""MCP client integration for NavigatorAI CLI.

Provides functions to spawn an MCP server, discover its tools, and run a
tool-calling loop against the NavigatorAI API (OpenAI-compatible).

Usage:
    import asyncio
    from mcp_client import run_with_tools

    answer = asyncio.run(run_with_tools(
        server_script="mcp_servers/csv_tools.py",
        prompt="What is the average score in sample_data.csv?",
        api_key="your-key",
    ))
"""

import sys

import requests
from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

NAVIGATOR_API_URL = "https://api.ai.it.ufl.edu/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o"


def mcp_tools_to_openai_format(tools: list) -> list[dict]:
    """Convert MCP tool definitions to the OpenAI function-calling format.

    Args:
        tools: List of MCP Tool objects from session.list_tools().

    Returns:
        A list of tool definitions in OpenAI's tool-calling schema.
    """
    openai_tools = []
    for tool in tools:
        schema = tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}}
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema,
            },
        })
    return openai_tools


def _call_navigator_api(
    messages: list[dict],
    api_key: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
    tools: list[dict] | None = None,
) -> dict:
    """Send a chat completion request to the NavigatorAI API.

    Args:
        messages: The conversation messages.
        api_key: NavigatorAI API key.
        model: Model to use.
        temperature: Sampling temperature.
        tools: OpenAI-format tool definitions (optional).

    Returns:
        The full JSON response from the API.

    Raises:
        requests.RequestException: If the API request fails.
    """
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if tools:
        payload["tools"] = tools

    logger.debug(f"Sending request to {NAVIGATOR_API_URL}")
    logger.debug(f"Model: {model}, Tools: {len(tools) if tools else 0}")

    response = requests.post(
        NAVIGATOR_API_URL,
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _extract_mcp_result_text(result) -> str:
    """Extract plain text from an MCP CallToolResult.

    Args:
        result: A CallToolResult from session.call_tool().

    Returns:
        The concatenated text content from the result.
    """
    parts = []
    for block in result.content:
        if hasattr(block, "text"):
            parts.append(block.text)
    return "\n".join(parts) if parts else str(result.content)


async def run_with_tools(
    server_script: str,
    prompt: str,
    api_key: str,
    model: str = DEFAULT_MODEL,
    system_prompt: str | None = None,
    temperature: float = 0.7,
    max_tool_rounds: int = 5,
) -> str:
    """Run a prompt with MCP tool support.

    Spawns an MCP server as a subprocess, discovers its tools, and enters a
    tool-calling loop with the NavigatorAI API until the LLM produces a final
    text response or the round limit is reached.

    Args:
        server_script: Path to the MCP server Python script.
        prompt: The user's question or prompt.
        api_key: NavigatorAI API key.
        model: Model to use (default: gpt-4o).
        system_prompt: Optional system prompt for context.
        temperature: Sampling temperature (0.0-2.0).
        max_tool_rounds: Maximum number of tool-calling rounds.

    Returns:
        The final text response from the LLM.
    """
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            mcp_tools = await session.list_tools()
            openai_tools = mcp_tools_to_openai_format(mcp_tools.tools)

            logger.debug(f"Discovered {len(openai_tools)} MCP tools")
            for t in openai_tools:
                logger.debug(f"  Tool: {t['function']['name']}")

            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            for round_num in range(max_tool_rounds):
                data = _call_navigator_api(
                    messages=messages,
                    api_key=api_key,
                    model=model,
                    temperature=temperature,
                    tools=openai_tools,
                )

                choice = data["choices"][0]
                assistant_message = choice["message"]
                messages.append(assistant_message)

                tool_calls = assistant_message.get("tool_calls")
                if not tool_calls:
                    return assistant_message.get("content", "")

                logger.debug(
                    f"Round {round_num + 1}: LLM requested "
                    f"{len(tool_calls)} tool call(s)"
                )

                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    func_args = tc["function"]["arguments"]

                    if isinstance(func_args, str):
                        import json
                        func_args = json.loads(func_args)

                    logger.debug(f"  Calling tool: {func_name}({func_args})")

                    result = await session.call_tool(func_name, func_args)
                    result_text = _extract_mcp_result_text(result)

                    logger.debug(f"  Tool result: {result_text[:200]}")

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result_text,
                    })

            final = _call_navigator_api(
                messages=messages,
                api_key=api_key,
                model=model,
                temperature=temperature,
            )
            return final["choices"][0]["message"].get("content", "")
