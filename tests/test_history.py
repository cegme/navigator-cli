"""Tests for the DuckDB history module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from history import (
    get_db_path,
    init_db,
    log_query,
    get_history,
    search_history,
    DUCKDB_AVAILABLE,
)


# Skip all tests if duckdb is not installed
pytestmark = pytest.mark.skipif(
    not DUCKDB_AVAILABLE, reason="duckdb not installed"
)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database path for testing."""
    return tmp_path / "test_history.db"


class TestInitDb:
    """Tests for database initialization."""

    def test_creates_table(self, tmp_db):
        """Test that init_db creates the query_history table."""
        conn = init_db(tmp_db)
        tables = conn.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'query_history'"
        ).fetchall()
        conn.close()
        assert len(tables) == 1

    def test_idempotent(self, tmp_db):
        """Test that init_db can be called multiple times safely."""
        conn1 = init_db(tmp_db)
        conn1.close()
        conn2 = init_db(tmp_db)
        tables = conn2.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_name = 'query_history'"
        ).fetchall()
        conn2.close()
        assert len(tables) == 1


class TestLogQuery:
    """Tests for logging queries."""

    def test_log_and_retrieve(self, tmp_db):
        """Test that a logged query can be retrieved."""
        log_query(
            model="gpt-4o",
            prompt="What is 2+2?",
            response="4",
            elapsed_time=1.5,
            temperature=0.7,
            cot_enabled=False,
            db_path=tmp_db,
        )

        records = get_history(db_path=tmp_db)
        assert len(records) == 1
        assert records[0]["model"] == "gpt-4o"
        assert records[0]["prompt"] == "What is 2+2?"
        assert records[0]["response"] == "4"
        assert records[0]["elapsed_time"] == 1.5
        assert records[0]["cot_enabled"] is False

    def test_log_with_system_prompt(self, tmp_db):
        """Test logging a query with a system prompt."""
        log_query(
            model="gpt-4o",
            prompt="Help me",
            response="Sure",
            elapsed_time=0.5,
            temperature=0.7,
            cot_enabled=False,
            system_prompt="You are helpful",
            db_path=tmp_db,
        )

        records = get_history(db_path=tmp_db)
        assert records[0]["system_prompt"] == "You are helpful"

    def test_log_multiple_queries(self, tmp_db):
        """Test logging multiple queries."""
        for i in range(5):
            log_query(
                model="gpt-4o",
                prompt=f"Question {i}",
                response=f"Answer {i}",
                elapsed_time=float(i),
                temperature=0.7,
                cot_enabled=False,
                db_path=tmp_db,
            )

        records = get_history(limit=10, db_path=tmp_db)
        assert len(records) == 5


class TestGetHistory:
    """Tests for retrieving history."""

    def test_empty_history(self, tmp_db):
        """Test getting history from empty database."""
        records = get_history(db_path=tmp_db)
        assert records == []

    def test_limit(self, tmp_db):
        """Test that limit parameter works."""
        for i in range(10):
            log_query(
                model="gpt-4o",
                prompt=f"Q{i}",
                response=f"A{i}",
                elapsed_time=1.0,
                temperature=0.7,
                cot_enabled=False,
                db_path=tmp_db,
            )

        records = get_history(limit=3, db_path=tmp_db)
        assert len(records) == 3

    def test_ordered_by_most_recent(self, tmp_db):
        """Test that results are ordered most recent first."""
        log_query(
            model="gpt-4o",
            prompt="First",
            response="A",
            elapsed_time=1.0,
            temperature=0.7,
            cot_enabled=False,
            db_path=tmp_db,
        )
        log_query(
            model="gpt-4o",
            prompt="Second",
            response="B",
            elapsed_time=1.0,
            temperature=0.7,
            cot_enabled=False,
            db_path=tmp_db,
        )

        records = get_history(db_path=tmp_db)
        assert records[0]["prompt"] == "Second"
        assert records[1]["prompt"] == "First"


class TestSearchHistory:
    """Tests for searching history."""

    def test_search_by_prompt(self, tmp_db):
        """Test searching history by prompt content."""
        log_query(
            model="gpt-4o",
            prompt="How do I use Python decorators?",
            response="Decorators are...",
            elapsed_time=1.0,
            temperature=0.7,
            cot_enabled=False,
            db_path=tmp_db,
        )
        log_query(
            model="gpt-4o",
            prompt="What is JavaScript?",
            response="JS is...",
            elapsed_time=1.0,
            temperature=0.7,
            cot_enabled=False,
            db_path=tmp_db,
        )

        results = search_history("Python", db_path=tmp_db)
        assert len(results) == 1
        assert "Python" in results[0]["prompt"]

    def test_search_by_response(self, tmp_db):
        """Test searching history by response content."""
        log_query(
            model="gpt-4o",
            prompt="Tell me something",
            response="DuckDB is an embedded database",
            elapsed_time=1.0,
            temperature=0.7,
            cot_enabled=False,
            db_path=tmp_db,
        )

        results = search_history("DuckDB", db_path=tmp_db)
        assert len(results) == 1

    def test_search_case_insensitive(self, tmp_db):
        """Test that search is case insensitive."""
        log_query(
            model="gpt-4o",
            prompt="UPPERCASE prompt",
            response="response",
            elapsed_time=1.0,
            temperature=0.7,
            cot_enabled=False,
            db_path=tmp_db,
        )

        results = search_history("uppercase", db_path=tmp_db)
        assert len(results) == 1

    def test_search_no_results(self, tmp_db):
        """Test search with no matching results."""
        log_query(
            model="gpt-4o",
            prompt="Hello",
            response="World",
            elapsed_time=1.0,
            temperature=0.7,
            cot_enabled=False,
            db_path=tmp_db,
        )

        results = search_history("nonexistent", db_path=tmp_db)
        assert results == []


class TestGetDbPath:
    """Tests for database path resolution."""

    def test_default_path(self):
        """Test that default path is under ~/.navigator."""
        path = get_db_path()
        assert path.name == "history.db"
        assert ".navigator" in str(path)

    def test_custom_path_via_env(self, tmp_path, monkeypatch):
        """Test that NAVIGATOR_DB_DIR overrides default path."""
        monkeypatch.setenv("NAVIGATOR_DB_DIR", str(tmp_path))
        path = get_db_path()
        assert path.parent == tmp_path


class TestHistoryCLI:
    """Tests for history CLI flags in navigator_cli.main."""

    @patch("navigator_cli.os.getenv")
    def test_history_flag(self, mock_getenv, tmp_path, capsys):
        """Test --history flag shows past queries."""
        mock_getenv.return_value = "test-api-key"

        db_path = tmp_path / "test.db"
        log_query(
            model="gpt-4o",
            prompt="Test prompt",
            response="Test response",
            elapsed_time=1.23,
            temperature=0.7,
            cot_enabled=False,
            db_path=db_path,
        )

        with patch("history.get_db_path", return_value=db_path):
            with patch("sys.argv", ["navigator_cli.py", "--history"]):
                from navigator_cli import main

                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "gpt-4o" in captured.out
        assert "Test prompt" in captured.out

    @patch("navigator_cli.os.getenv")
    def test_search_flag(self, mock_getenv, tmp_path, capsys):
        """Test --search flag finds matching queries."""
        mock_getenv.return_value = "test-api-key"

        db_path = tmp_path / "test.db"
        log_query(
            model="gpt-4o",
            prompt="How to use pandas",
            response="Import pandas as pd",
            elapsed_time=2.0,
            temperature=0.7,
            cot_enabled=False,
            db_path=db_path,
        )

        with patch("history.get_db_path", return_value=db_path):
            with patch("sys.argv", ["navigator_cli.py", "--search", "pandas"]):
                from navigator_cli import main

                with pytest.raises(SystemExit) as exc_info:
                    main()

        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "pandas" in captured.out
