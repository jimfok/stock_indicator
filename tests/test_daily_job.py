import datetime
from pathlib import Path

import pytest
import pandas

from stock_indicator import daily_job


def test_run_daily_job_writes_log_file(tmp_path, monkeypatch):
    """run_daily_job should create a dated log file with signals."""

    def fake_find_signal(
        date_string: str,
        dollar_volume_filter: str,
        buy_strategy: str,
        sell_strategy: str,
        stop_loss: float,
        allowed_groups: set[int] | None,
    ):
        return {"entry_signals": ["AAA"], "exit_signals": ["BBB"]}

    monkeypatch.setattr(daily_job, "find_signal", fake_find_signal)

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


def test_run_daily_job_accepts_percentage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_daily_job should parse percentage-based filters."""

    captured_arguments: dict[str, float | int | None] = {}

    def fake_run_daily_tasks(
        buy_strategy_name: str,
        sell_strategy_name: str,
        start_date: str,
        end_date: str,
        symbol_list=None,
        data_download_function=None,
        data_directory: Path | None = None,
        minimum_average_dollar_volume: float | None = None,
        top_dollar_volume_rank: int | None = None,
        allowed_fama_french_groups: set[int] | None = None,
    ):
        captured_arguments["minimum_average_dollar_volume"] = minimum_average_dollar_volume
        captured_arguments["top_dollar_volume_rank"] = top_dollar_volume_rank
        return {"entry_signals": ["AAA"], "exit_signals": ["BBB"]}

    monkeypatch.setattr(daily_job.cron, "run_daily_tasks", fake_run_daily_tasks)

    log_directory = tmp_path / "logs"
    data_directory = tmp_path / "data"
    current_date = datetime.date(2024, 1, 10)

    log_file_path = daily_job.run_daily_job(
        "dollar_volume>2.41% ema_sma_cross ema_sma_cross",
        data_directory=data_directory,
        log_directory=log_directory,
        current_date=current_date,
    )

    assert captured_arguments["minimum_average_dollar_volume"] == pytest.approx(0.0241)
    assert captured_arguments["top_dollar_volume_rank"] is None
    assert log_file_path.read_text(encoding="utf-8") == (
        "entry_signals: AAA\nexit_signals: BBB\n"
    )


def test_run_daily_job_accepts_percentage_and_rank(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """run_daily_job should parse combined percentage and ranking filters."""

    captured_arguments: dict[str, float | int | None] = {}

    def fake_run_daily_tasks(
        buy_strategy_name: str,
        sell_strategy_name: str,
        start_date: str,
        end_date: str,
        symbol_list=None,
        data_download_function=None,
        data_directory: Path | None = None,
        minimum_average_dollar_volume: float | None = None,
        top_dollar_volume_rank: int | None = None,
        allowed_fama_french_groups: set[int] | None = None,
    ):
        captured_arguments["minimum_average_dollar_volume"] = minimum_average_dollar_volume
        captured_arguments["top_dollar_volume_rank"] = top_dollar_volume_rank
        return {"entry_signals": ["AAA"], "exit_signals": ["BBB"]}

    monkeypatch.setattr(daily_job.cron, "run_daily_tasks", fake_run_daily_tasks)

    log_directory = tmp_path / "logs"
    data_directory = tmp_path / "data"
    current_date = datetime.date(2024, 1, 10)

    log_file_path = daily_job.run_daily_job(
        "dollar_volume>2.41%,5th ema_sma_cross ema_sma_cross",
        data_directory=data_directory,
        log_directory=log_directory,
        current_date=current_date,
    )

    assert captured_arguments["minimum_average_dollar_volume"] == pytest.approx(0.0241)
    assert captured_arguments["top_dollar_volume_rank"] == 5
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


def test_run_daily_job_expands_strategy_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_daily_job should replace ``strategy=ID`` with buy and sell names."""

    captured_parameters: dict[str, object] = {}

    def fake_find_signal(
        date_string: str,
        dollar_volume_filter: str,
        buy_strategy: str,
        sell_strategy: str,
        stop_loss: float,
        allowed_groups: set[int] | None,
    ):
        captured_parameters["dollar_volume_filter"] = dollar_volume_filter
        captured_parameters["buy_strategy"] = buy_strategy
        captured_parameters["sell_strategy"] = sell_strategy
        captured_parameters["stop_loss"] = stop_loss
        captured_parameters["allowed_groups"] = allowed_groups
        return {"entry_signals": [], "exit_signals": []}

    monkeypatch.setattr(daily_job, "find_signal", fake_find_signal)
    monkeypatch.setattr(
        daily_job.strategy_sets,
        "load_strategy_set_mapping",
        lambda: {"s1": ("buy_one", "sell_two")},
    )

    log_directory = tmp_path / "logs"
    data_directory = tmp_path / "data"
    current_date = datetime.date(2024, 1, 10)

    daily_job.run_daily_job(
        "group=1,2 dollar_volume>1 strategy=s1 0.5",
        data_directory=data_directory,
        log_directory=log_directory,
        current_date=current_date,
    )

    assert captured_parameters["dollar_volume_filter"] == "dollar_volume>1"
    assert captured_parameters["buy_strategy"] == "buy_one"
    assert captured_parameters["sell_strategy"] == "sell_two"
    assert captured_parameters["stop_loss"] == 0.5
    assert captured_parameters["allowed_groups"] == {1, 2}


def test_run_daily_job_unknown_strategy_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """run_daily_job should raise when strategy id is not found."""

    monkeypatch.setattr(
        daily_job.strategy_sets, "load_strategy_set_mapping", lambda: {}
    )

    log_directory = tmp_path / "logs"
    data_directory = tmp_path / "data"
    current_date = datetime.date(2024, 1, 10)

    with pytest.raises(ValueError, match="unknown strategy id: missing"):
        daily_job.run_daily_job(
            "dollar_volume>1 strategy=missing 0.2",
            data_directory=data_directory,
            log_directory=log_directory,
            current_date=current_date,
        )


def test_find_signal_returns_cron_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """find_signal should return the values from cron."""

    expected_result = {"entry_signals": ["AAA"], "exit_signals": ["BBB"]}

    def fake_run_daily_tasks_from_argument(
        argument_line: str,
        start_date: str,
        end_date: str,
        symbol_list=None,
        data_download_function=None,
        data_directory: Path | None = None,
    ):
        return expected_result

    monkeypatch.setattr(
        daily_job.cron, "run_daily_tasks_from_argument", fake_run_daily_tasks_from_argument
    )

    signal_dictionary = daily_job.find_signal(
        "2024-01-10",
        "dollar_volume>1",
        "ema_sma_cross",
        "ema_sma_cross",
        1.0,
    )

    assert signal_dictionary == expected_result


def test_find_signal_detects_previous_day_crossover(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """find_signal should detect signals when the crossover is one day earlier."""

    data_directory = tmp_path
    csv_lines = ["Date,open,close,volume\n"]
    start_day = datetime.date(2024, 1, 1)
    for day_index in range(52):
        current_day = start_day + datetime.timedelta(days=day_index)
        price_value = 1.0 if day_index < 50 else 2.0
        csv_lines.append(
            f"{current_day.isoformat()},{price_value},{price_value},1000000\n"
        )
    (data_directory / "AAA.csv").write_text("".join(csv_lines), encoding="utf-8")

    monkeypatch.setattr(daily_job, "STOCK_DATA_DIRECTORY", data_directory)
    monkeypatch.setattr(daily_job.cron, "update_symbol_cache", lambda: None)
    monkeypatch.setattr(daily_job.cron, "load_symbols", lambda: ["AAA"])

    original_run = daily_job.cron.run_daily_tasks_from_argument

    def fake_download_history(
        symbol: str, start: str, end: str, cache_path: Path | None = None
    ):
        return pandas.read_csv(cache_path, parse_dates=["Date"], index_col="Date")

    def patched_run_daily_tasks_from_argument(
        argument_line: str,
        start_date: str,
        end_date: str,
        symbol_list=None,
        data_download_function=None,
        data_directory: Path | None = None,
    ):
        return original_run(
            argument_line,
            start_date,
            end_date,
            symbol_list=symbol_list,
            data_download_function=fake_download_history,
            data_directory=data_directory,
        )

    monkeypatch.setattr(
        daily_job.cron, "run_daily_tasks_from_argument", patched_run_daily_tasks_from_argument
    )

    signal_dictionary = daily_job.find_signal(
        "2024-02-21",
        "dollar_volume>1",
        "20_50_sma_cross",
        "20_50_sma_cross",
        1.0,
    )

    assert signal_dictionary["entry_signals"] == ["AAA"]
