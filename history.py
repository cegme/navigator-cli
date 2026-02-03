"""
DuckDB-based query history storage for navigator-cli.

Provides optional persistent logging of LLM queries and responses,
with search and history retrieval capabilities.

Requires: duckdb (install via `uv add "navigator-cli[analytics]"`)
"""

import os
from datetime import datetime
from pathlib import Path

from loguru import logger

try:
    import duckdb

    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

DEFAULT_DB_DIR = Path.home() / ".navigator"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "history.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS query_history (
    id INTEGER PRIMARY KEY DEFAULT nextval('query_history_id_seq'),
    timestamp TIMESTAMP NOT NULL,
    model VARCHAR NOT NULL,
    prompt TEXT NOT NULL,
    system_prompt TEXT,
    response TEXT NOT NULL,
    elapsed_time DOUBLE NOT NULL,
    temperature DOUBLE NOT NULL,
    cot_enabled BOOLEAN NOT NULL
)
"""

CREATE_SEQUENCE_SQL = "CREATE SEQUENCE IF NOT EXISTS query_history_id_seq START 1"


def get_db_path() -> Path:
    """Return the database path, creating the directory if needed.

    Returns:
        Path to the DuckDB database file.
    """
    db_dir = Path(os.environ.get("NAVIGATOR_DB_DIR", str(DEFAULT_DB_DIR)))
    db_dir.mkdir(parents=True, exist_ok=True)
    return db_dir / "history.db"


def init_db(db_path: Path | None = None) -> "duckdb.DuckDBPyConnection":
    """Initialize the DuckDB database and return a connection.

    Args:
        db_path: Optional path to the database file. Defaults to
            ~/.navigator/history.db.

    Returns:
        An open DuckDB connection with the schema initialized.

    Raises:
        ImportError: If duckdb is not installed.
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError(
            "duckdb is not installed. Install it with: "
            'uv add "navigator-cli[analytics]"'
        )

    if db_path is None:
        db_path = get_db_path()

    conn = duckdb.connect(str(db_path))
    conn.execute(CREATE_SEQUENCE_SQL)
    conn.execute(CREATE_TABLE_SQL)
    return conn


def log_query(
    model: str,
    prompt: str,
    response: str,
    elapsed_time: float,
    temperature: float,
    cot_enabled: bool,
    system_prompt: str | None = None,
    db_path: Path | None = None,
) -> None:
    """Log a query and response to the history database.

    Args:
        model: The model used for the query.
        prompt: The user prompt sent to the LLM.
        response: The LLM response text.
        elapsed_time: Time taken for the query in seconds.
        temperature: Sampling temperature used.
        cot_enabled: Whether Chain-of-Thought was enabled.
        system_prompt: Optional system prompt used.
        db_path: Optional path to the database file.
    """
    if not DUCKDB_AVAILABLE:
        logger.debug("duckdb not installed, skipping query logging")
        return

    try:
        conn = init_db(db_path)
        conn.execute(
            """
            INSERT INTO query_history
                (timestamp, model, prompt, system_prompt, response,
                 elapsed_time, temperature, cot_enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                datetime.now(),
                model,
                prompt,
                system_prompt,
                response,
                elapsed_time,
                temperature,
                cot_enabled,
            ],
        )
        conn.close()
        logger.debug("Query logged to history database")
    except Exception as e:
        logger.warning(f"Failed to log query to history: {e}")


def get_history(
    limit: int = 20, db_path: Path | None = None
) -> list[dict]:
    """Retrieve recent query history.

    Args:
        limit: Maximum number of records to return.
        db_path: Optional path to the database file.

    Returns:
        List of query history records as dictionaries.

    Raises:
        ImportError: If duckdb is not installed.
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError(
            "duckdb is not installed. Install it with: "
            'uv add "navigator-cli[analytics]"'
        )

    conn = init_db(db_path)
    results = conn.execute(
        """
        SELECT id, timestamp, model, prompt, system_prompt,
               response, elapsed_time, temperature, cot_enabled
        FROM query_history
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        [limit],
    ).fetchall()
    conn.close()

    columns = [
        "id", "timestamp", "model", "prompt", "system_prompt",
        "response", "elapsed_time", "temperature", "cot_enabled",
    ]
    return [dict(zip(columns, row)) for row in results]


def search_history(
    query: str, limit: int = 20, db_path: Path | None = None
) -> list[dict]:
    """Search query history by prompt or response content.

    Args:
        query: Search term to match against prompts and responses.
        limit: Maximum number of records to return.
        db_path: Optional path to the database file.

    Returns:
        List of matching query history records as dictionaries.

    Raises:
        ImportError: If duckdb is not installed.
    """
    if not DUCKDB_AVAILABLE:
        raise ImportError(
            "duckdb is not installed. Install it with: "
            'uv add "navigator-cli[analytics]"'
        )

    conn = init_db(db_path)
    search_pattern = f"%{query}%"
    results = conn.execute(
        """
        SELECT id, timestamp, model, prompt, system_prompt,
               response, elapsed_time, temperature, cot_enabled
        FROM query_history
        WHERE prompt ILIKE ? OR response ILIKE ?
        ORDER BY timestamp DESC
        LIMIT ?
        """,
        [search_pattern, search_pattern, limit],
    ).fetchall()
    conn.close()

    columns = [
        "id", "timestamp", "model", "prompt", "system_prompt",
        "response", "elapsed_time", "temperature", "cot_enabled",
    ]
    return [dict(zip(columns, row)) for row in results]
