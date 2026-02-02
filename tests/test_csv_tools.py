"""Tests for the MCP CSV tools server."""

import os
import tempfile

from mcp_servers.csv_tools import count_rows, csv_stats, read_csv_file

SAMPLE_CSV = os.path.join(
    os.path.dirname(__file__), "..", "mcp_servers", "sample_data.csv"
)


class TestReadCsvFile:
    """Tests for the read_csv_file tool."""

    def test_read_sample_data(self):
        """Test reading the bundled sample CSV file."""
        result = read_csv_file(SAMPLE_CSV)
        assert "name" in result
        assert "Alice" in result
        assert "Eve" in result

    def test_read_returns_all_rows(self):
        """Test that all data rows appear in the output."""
        result = read_csv_file(SAMPLE_CSV)
        lines = result.strip().split("\n")
        # Header + 5 data rows
        assert len(lines) == 6

    def test_read_empty_csv(self, tmp_path):
        """Test reading a CSV with only headers (no data rows)."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("name,age\n")
        result = read_csv_file(str(csv_file))
        assert result == "The CSV file is empty."


class TestCsvStats:
    """Tests for the csv_stats tool."""

    def test_stats_for_age_column(self):
        """Test computing stats for the age column."""
        result = csv_stats(SAMPLE_CSV, "age")
        assert "mean" in result
        assert "median" in result
        assert "count" in result
        assert "5" in result  # 5 rows

    def test_stats_for_score_column(self):
        """Test computing stats for the score column."""
        result = csv_stats(SAMPLE_CSV, "score")
        assert "mean" in result
        # Mean of 85, 92, 78, 95, 88 = 87.6
        assert "87.6" in result

    def test_stats_nonexistent_column(self):
        """Test requesting stats for a column that does not exist."""
        result = csv_stats(SAMPLE_CSV, "nonexistent")
        assert "not found" in result
        assert "name" in result  # Should suggest available columns

    def test_stats_non_numeric_column(self):
        """Test stats for a non-numeric column returns appropriate message."""
        result = csv_stats(SAMPLE_CSV, "name")
        assert "No numeric values" in result

    def test_stats_empty_csv(self, tmp_path):
        """Test stats on an empty CSV."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("x,y\n")
        result = csv_stats(str(csv_file), "x")
        assert "empty" in result


class TestCountRows:
    """Tests for the count_rows tool."""

    def test_count_sample_data(self):
        """Test counting rows in the sample CSV."""
        result = count_rows(SAMPLE_CSV)
        assert "5" in result

    def test_count_empty_csv(self, tmp_path):
        """Test counting rows in an empty CSV."""
        csv_file = tmp_path / "empty.csv"
        csv_file.write_text("a,b,c\n")
        result = count_rows(str(csv_file))
        assert "0" in result
