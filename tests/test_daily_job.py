import datetime
from pathlib import Path

import pytest

from stock_indicator import daily_job


def test_run_daily_job_writes_log_file(tmp_path, monkeypatch):
    """run_daily_job should create a dated log file with signals."""

    def fake_run_daily_tasks_from_argument(
        argument_line: str,
        start_date: str,
        end_date: str,
        symbol_list=None,
        data_download_function=None,
        data_directory: Path | None = None,
    ):
        return {"entry_signals": ["AAA"], "exit_signals": ["BBB"]}

    monkeypatch.setattr(
        daily_job.cron,
        "run_daily_tasks_from_argument",
        fake_run_daily_tasks_from_argument,
    )

    log_directory = tmp_path / "logs"
    data_directory = tmp_path / "data"
    current_date = datetime.date(2024, 1, 10)

    log_file_path = daily_job.run_daily_job(
        "dollar_volume>1 ema_sma_cross ema_sma_cross",
        data_directory=data_directory,
        log_directory=log_directory,
        current_date=current_date,
    )

    expected_log_path = log_directory / "2024-01-10.log"
    assert log_file_path == expected_log_path
    assert log_file_path.read_text(encoding="utf-8") == (
        "entry_signals: AAA\nexit_signals: BBB\n"
    )


def test_run_daily_job_uses_oldest_data_date(tmp_path, monkeypatch):
    """run_daily_job should start from the oldest available data date."""

    captured_start_date: dict[str, str] = {}

    def fake_run_daily_tasks_from_argument(
        argument_line: str,
        start_date: str,
        end_date: str,
        symbol_list=None,
        data_download_function=None,
        data_directory: Path | None = None,
    ):
        captured_start_date["value"] = start_date
        return {"entry_signals": [], "exit_signals": []}

    monkeypatch.setattr(
        daily_job.cron, "run_daily_tasks_from_argument", fake_run_daily_tasks_from_argument
    )

    data_directory = tmp_path / "data"
    data_directory.mkdir()
    (data_directory / "AAA.csv").write_text(
        "Date,open,close\n2018-06-01,1,1\n2018-06-02,1,1\n", encoding="utf-8"
    )
    (data_directory / "BBB.csv").write_text(
        "Date,open,close\n2019-01-01,1,1\n2019-01-02,1,1\n", encoding="utf-8"
    )
    log_directory = tmp_path / "logs"
    current_date = datetime.date(2024, 1, 10)

    daily_job.run_daily_job(
        "dollar_volume>1 ema_sma_cross ema_sma_cross",
        data_directory=data_directory,
        log_directory=log_directory,
        current_date=current_date,
    )

    assert captured_start_date["value"] == "2018-06-01"

def test_find_signal_returns_expected_symbols(tmp_path):
    """find_signal should return symbol lists stored in the log file."""

    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    sample_log_path = log_directory / "2024-01-10.log"
    sample_log_path.write_text(
        "entry_signals: AAA, BBB\nexit_signals: CCC, DDD\n", encoding="utf-8"
    )

    signal_dictionary = daily_job.find_signal(
        "2024-01-10", log_directory=log_directory
    )

    assert signal_dictionary == {
        "entry_signals": ["AAA", "BBB"],
        "exit_signals": ["CCC", "DDD"],
    }


def test_find_signal_raises_file_not_found_error(tmp_path):
    """find_signal should raise FileNotFoundError when the log is absent."""

    log_directory = tmp_path / "logs"
    log_directory.mkdir()

    with pytest.raises(FileNotFoundError):
        daily_job.find_signal("2024-01-10", log_directory=log_directory)
