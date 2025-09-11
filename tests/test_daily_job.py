import datetime
import logging
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas
import pytest
import yfinance.exceptions as yfinance_exceptions

from stock_indicator import cron, daily_job


def test_find_history_signal_returns_cron_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """find_history_signal should return the values from cron."""

    expected_result = {"entry_signals": ["AAA"], "exit_signals": ["BBB"]}

    def fake_run_daily_tasks(*args, **kwargs):
        return expected_result

    monkeypatch.setattr(daily_job, "run_daily_tasks", fake_run_daily_tasks)
    csv_file_path = tmp_path / "AAA.csv"
    csv_file_path.write_text(
        "Date,open,close\n2023-08-01,1,1\n2024-01-10,1,1\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(daily_job, "STOCK_DATA_DIRECTORY", tmp_path)

    signal_dictionary = daily_job.find_history_signal(
        "2024-01-10",
        "dollar_volume>1",
        "ema_sma_cross",
        "ema_sma_cross",
        1.0,
    )

    assert signal_dictionary["entry_signals"] == expected_result["entry_signals"]
    assert signal_dictionary["exit_signals"] == expected_result["exit_signals"]


def test_find_history_signal_detects_same_day_crossover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """find_history_signal should detect signals on the crossover day."""

    data_directory = tmp_path
    csv_lines = ["Date,open,close,volume\n"]
    start_day = datetime.date(2024, 1, 1)
    for day_index in range(51):
        current_day = start_day + datetime.timedelta(days=day_index)
        price_value = 1.0 if day_index < 50 else 2.0
        csv_lines.append(
            f"{current_day.isoformat()},{price_value},{price_value},1000000\n"
        )
    (data_directory / "AAA.csv").write_text("".join(csv_lines), encoding="utf-8")

    monkeypatch.setattr(daily_job, "STOCK_DATA_DIRECTORY", data_directory)
    monkeypatch.setattr(cron, "update_symbol_cache", lambda: None)
    monkeypatch.setattr(cron, "load_symbols", lambda: ["AAA"])

    original_run = daily_job.run_daily_tasks

    def fake_download_history(
        symbol: str, start: str, end: str, cache_path: Path | None = None
    ):
        return pandas.read_csv(cache_path, parse_dates=["Date"], index_col="Date")

    def patched_run_daily_tasks(
        buy_strategy_name: str,
        sell_strategy_name: str,
        start_date: str,
        end_date: str,
        symbol_list=None,
        data_download_function=None,
        data_directory: Path | None = None,
        minimum_average_dollar_volume=None,
        top_dollar_volume_rank=None,
        allowed_fama_french_groups=None,
        maximum_symbols_per_group: int = 1,
        use_unshifted_signals: bool = False,
    ):
        return original_run(
            buy_strategy_name,
            sell_strategy_name,
            start_date,
            end_date,
            symbol_list=symbol_list,
            data_download_function=fake_download_history,
            data_directory=data_directory,
            minimum_average_dollar_volume=minimum_average_dollar_volume,
            top_dollar_volume_rank=top_dollar_volume_rank,
            allowed_fama_french_groups=allowed_fama_french_groups,
            maximum_symbols_per_group=maximum_symbols_per_group,
            use_unshifted_signals=use_unshifted_signals,
        )

    monkeypatch.setattr(daily_job, "run_daily_tasks", patched_run_daily_tasks)

    signal_dictionary = daily_job.find_history_signal(
        "2024-02-20",
        "dollar_volume>1",
        "20_50_sma_cross",
        "20_50_sma_cross",
        1.0,
    )

    assert signal_dictionary["entry_signals"] == ["AAA"]


def test_find_history_signal_skips_download_when_cache_covers_range(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """find_history_signal should skip downloading when the cache spans the required range."""

    data_directory = tmp_path
    csv_file_path = data_directory / "AAA.csv"
    csv_file_path.write_text(
        "Date,open,close\n2023-08-01,1,1\n2024-01-10,1,1\n2024-01-11,1,1\n",
        encoding="utf-8",
    )

    download_calls: list[str] = []

    def fake_download_history(
        symbol_name: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        download_calls.append(symbol_name)
        return pandas.DataFrame()

    monkeypatch.setattr(daily_job, "STOCK_DATA_DIRECTORY", data_directory)
    monkeypatch.setattr(daily_job, "download_history", fake_download_history)
    monkeypatch.setattr(
        daily_job,
        "run_daily_tasks",
        lambda *a, **k: {"entry_signals": [], "exit_signals": []},
    )

    daily_job.find_history_signal(
        "2024-01-10",
        "dollar_volume>1",
        "buy",
        "sell",
        1.0,
    )

    assert not download_calls


def test_update_all_data_from_yf_deduplicates_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """update_all_data_from_yf should remove duplicate rows."""

    data_directory = tmp_path
    csv_path = data_directory / "AAA.csv"
    csv_path.write_text(
        "Date,open,close\n2024-01-01,1,1\n2024-01-02,1,1\n",
        encoding="utf-8",
    )

    def fake_download_history(
        symbol_name: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        existing_frame = pandas.read_csv(cache_path, index_col=0, parse_dates=True)
        new_frame = pandas.DataFrame(
            {"open": [1, 1], "close": [1, 1]},
            index=pandas.to_datetime(["2024-01-02", "2024-01-03"]),
        )
        combined_frame = pandas.concat([existing_frame, new_frame])
        combined_frame.to_csv(cache_path)
        return combined_frame

    monkeypatch.setattr(daily_job, "download_history", fake_download_history)
    monkeypatch.setattr(daily_job, "load_symbols", lambda: ["AAA"])
    daily_job.update_all_data_from_yf(
        "2024-01-01", "2024-01-04", data_directory
    )

    result_frame = pandas.read_csv(csv_path, index_col=0, parse_dates=True)
    assert list(result_frame.index.strftime("%Y-%m-%d")) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    assert not result_frame.index.duplicated().any()


def test_update_all_data_from_yf_preserves_existing_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing rows should remain intact after a refresh."""

    data_directory = tmp_path
    csv_file_path = data_directory / "AAA.csv"
    csv_file_path.write_text(
        (
            "Date,open,close\n"
            "2024-01-01,1,1\n"
            "2024-01-02,1,1\n"
            "2024-01-03,1,1\n"
        ),
        encoding="utf-8",
    )

    def fake_download_history(
        symbol_name: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        existing_frame = pandas.read_csv(cache_path, index_col=0, parse_dates=True)
        new_frame = pandas.DataFrame(
            {"open": [1, 1], "close": [1, 1]},
            index=pandas.to_datetime(["2024-01-03", "2024-01-04"]),
        )
        combined_frame = pandas.concat([existing_frame, new_frame])
        combined_frame.to_csv(cache_path)
        return combined_frame

    monkeypatch.setattr(daily_job, "download_history", fake_download_history)
    monkeypatch.setattr(daily_job, "load_symbols", lambda: ["AAA"])

    daily_job.update_all_data_from_yf(
        "2024-01-01", "2024-01-05", data_directory
    )

    result_frame = pandas.read_csv(csv_file_path, index_col=0, parse_dates=True)
    assert list(result_frame.index.strftime("%Y-%m-%d")) == [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
        "2024-01-04",
    ]
    assert not result_frame.index.duplicated().any()


def test_find_history_signal_without_date_uses_latest_trading_day(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Omitting the date should evaluate the latest trading day."""

    task_arguments: dict[str, str] = {}

    def fake_run_daily_tasks(
        buy_strategy_name: str,
        sell_strategy_name: str,
        start_date: str,
        end_date: str,
        symbol_list: list[str] | None,
        data_directory: Path,
        use_unshifted_signals: bool,
        **_: object,
    ) -> dict[str, list[str]]:
        task_arguments["end_date"] = end_date
        return {"entry_signals": [], "exit_signals": []}

    monkeypatch.setattr(daily_job, "run_daily_tasks", fake_run_daily_tasks)
    monkeypatch.setattr(daily_job, "STOCK_DATA_DIRECTORY", tmp_path)
    monkeypatch.setattr(
        daily_job, "determine_latest_trading_date", lambda: datetime.date(2024, 1, 10)
    )

    csv_path = tmp_path / "AAA.csv"
    csv_path.write_text(
        "Date,close\n2024-01-10,1.0\n", encoding="utf-8"
    )

    result = daily_job.find_history_signal(
        None, "dollar_volume>1", "buy", "sell", 1.0
    )

    assert task_arguments["end_date"] == "2024-01-10"
    assert result["entry_signals"] == []
    assert result["exit_signals"] == []


def test_find_history_signal_uses_previous_trading_day_before_market_open(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A pre-market call should evaluate the prior trading day."""

    task_arguments: dict[str, str] = {}

    def fake_run_daily_tasks(
        buy_strategy_name: str,
        sell_strategy_name: str,
        start_date: str,
        end_date: str,
        symbol_list: list[str] | None,
        data_directory: Path,
        use_unshifted_signals: bool,
        **_: object,
    ) -> dict[str, list[str]]:
        task_arguments["end_date"] = end_date
        return {"entry_signals": [], "exit_signals": []}

    pre_open_timestamp = datetime.datetime(
        2025, 9, 10, 1, 47, tzinfo=ZoneInfo("US/Eastern")
    )

    original_helper = daily_job.determine_latest_trading_date

    def fake_determine_latest_trading_date() -> datetime.date:
        return original_helper(pre_open_timestamp)

    class FakeDate(datetime.date):
        @classmethod
        def today(cls) -> datetime.date:
            return datetime.date(2025, 9, 10)

    monkeypatch.setattr(daily_job, "run_daily_tasks", fake_run_daily_tasks)
    monkeypatch.setattr(daily_job, "determine_latest_trading_date", fake_determine_latest_trading_date)
    monkeypatch.setattr(daily_job.datetime, "date", FakeDate)
    monkeypatch.setattr(daily_job, "STOCK_DATA_DIRECTORY", tmp_path)

    csv_path = tmp_path / "AAA.csv"
    csv_path.write_text("Date,close\n2025-09-08,1.0\n", encoding="utf-8")

    daily_job.find_history_signal(None, "dollar_volume>1", "buy", "sell", 1.0)

    assert task_arguments["end_date"] == "2025-09-09"


def test_update_all_data_from_yf_logs_warning_on_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Errors during download should be logged and not raised."""

    def fake_download_history(
        symbol_name: str, start: str, end: str, cache_path: Path | None = None
    ) -> pandas.DataFrame:
        if symbol_name == "BBB":
            raise yfinance_exceptions.YFException("bad symbol")
        frame = pandas.DataFrame({"close": [1.0]}, index=pandas.to_datetime([start]))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(cache_path)
        return frame

    monkeypatch.setattr(daily_job, "download_history", fake_download_history)

    with caplog.at_level(logging.WARNING):
        daily_job.update_all_data_from_yf(
            "2024-01-01", "2024-01-05", tmp_path
        )

    assert any("BBB" in record.message for record in caplog.records)

