"""Fixtures for sector pipeline tests."""

# TODO: review

from pathlib import Path

import pytest


@pytest.fixture
def company_ticker_payload() -> dict:
    """Return a minimal SEC company ticker table payload."""
    return {
        "0": {"cik_str": "1", "ticker": "AAA", "title": "Alpha Company"},
        "1": {"cik_str": "2", "ticker": "BBB", "title": "Beta Company"},
    }


@pytest.fixture
def submissions_payloads() -> dict:
    """Return submission payloads keyed by central index key."""
    return {1: {"sic": 1000}, 2: {"sic": 2000}}


@pytest.fixture
def universe_file(tmp_path: Path) -> Path:
    """Create a text file listing a small ticker universe."""
    file_path = tmp_path / "universe.txt"
    file_path.write_text("AAA\nBBB\n", encoding="utf-8")
    return file_path


@pytest.fixture
def ff_mapping_file(tmp_path: Path) -> Path:
    """Create a CSV file representing a SIC to Fama-French mapping."""
    csv_content = (
        "sic_start,sic_end,ff12,ff48,ff49,label\n"
        "1000,1999,1,10,100,Alpha\n"
        "2000,2999,2,20,200,Beta\n"
    )
    file_path = tmp_path / "mapping.csv"
    file_path.write_text(csv_content, encoding="utf-8")
    return file_path


@pytest.fixture
def ff_mapping_csv_text() -> str:
    """Return CSV text used for mocking HTTP responses."""
    return (
        "sic_start,sic_end,ff12,ff48,ff49,label\n"
        "1000,1999,1,10,100,Alpha\n"
    )
