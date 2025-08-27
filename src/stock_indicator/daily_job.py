"""Entry point for running the daily cron tasks."""
# TODO: review

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict, List

import pandas

from . import cron

LOGGER = logging.getLogger(__name__)

DEFAULT_START_DATE = "2019-01-01"
DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
LOG_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "logs"


def determine_start_date(data_directory: Path) -> str:
    """Return the oldest date found in ``data_directory``.

    When no CSV files are available, ``DEFAULT_START_DATE`` is returned.
    """
    earliest_date: datetime.date | None = None
    if not data_directory.exists():
        return DEFAULT_START_DATE
    for csv_file_path in data_directory.glob("*.csv"):
        try:
            date_frame = pandas.read_csv(
                csv_file_path, usecols=[0], nrows=1, parse_dates=[0]
            )
        except Exception as read_error:  # noqa: BLE001
            LOGGER.warning("Could not read %s: %s", csv_file_path, read_error)
            continue
        if date_frame.empty:
            continue
        first_date = date_frame.iloc[0, 0].date()
        if earliest_date is None or first_date < earliest_date:
            earliest_date = first_date
    if earliest_date is None:
        return DEFAULT_START_DATE
    return earliest_date.isoformat()


def run_daily_job(
    argument_line: str,
    *,
    data_directory: Path | None = None,
    log_directory: Path | None = None,
    current_date: datetime.date | None = None,
) -> Path:
    """Execute daily tasks and record the signals to a log file.

    Parameters
    ----------
    argument_line: str
        Argument string in the format accepted by
        :func:`cron.run_daily_tasks_from_argument`.
    data_directory: Path | None, optional
        Directory where downloaded price history data is written. Defaults to the
        project's ``data`` directory.
    log_directory: Path | None, optional
        Directory where the log file is stored. Defaults to the ``logs``
        directory in the project root.
    current_date: datetime.date | None, optional
        Date used for data retrieval and log file naming. When ``None`` the
        system date is used.

    Returns
    -------
    Path
        Path to the log file containing the entry and exit signals.
    """
    if current_date is None:
        current_date = datetime.date.today()
    if data_directory is None:
        data_directory = DATA_DIRECTORY
    if log_directory is None:
        log_directory = LOG_DIRECTORY

    current_date_string = current_date.isoformat()
    LOGGER.info("Starting daily tasks for %s", current_date_string)
    start_date_string = determine_start_date(data_directory)
    signal_result: Dict[str, list[str]] = cron.run_daily_tasks_from_argument(
        argument_line,
        start_date=start_date_string,
        end_date=current_date_string,
        data_directory=data_directory,
    )
    log_directory.mkdir(parents=True, exist_ok=True)
    log_file_path = log_directory / f"{current_date_string}.log"
    with log_file_path.open("w", encoding="utf-8") as log_file:
        log_file.write(
            f"entry_signals: {', '.join(signal_result['entry_signals'])}\n",
        )
        log_file.write(
            f"exit_signals: {', '.join(signal_result['exit_signals'])}\n",
        )
    LOGGER.info("Daily tasks completed; results written to %s", log_file_path)
    return log_file_path


def find_signal(
    date_string: str, log_directory: Path | None = None
) -> Dict[str, List[str]]:
    """Return entry and exit signals stored in the log file.

    Parameters
    ----------
    date_string:
        Date string in ISO format representing the log file to parse.
    log_directory: Path | None, optional
        Directory where the log file is stored. When ``None`` the module's
        ``LOG_DIRECTORY`` value is used.

    Returns
    -------
    Dict[str, List[str]]
        Dictionary containing two keys: ``"entry_signals"`` and
        ``"exit_signals"``. Each key maps to the list of symbols parsed from
        the corresponding line in the log file. Missing lines result in empty
        lists.

    Raises
    ------
    FileNotFoundError
        Raised when the log file for ``date_string`` does not exist.
    """
    if log_directory is None:
        log_directory = LOG_DIRECTORY
    log_file_path = log_directory / f"{date_string}.log"
    if not log_file_path.exists():
        error_message = f"Log file {log_file_path} does not exist"
        LOGGER.error(error_message)
        raise FileNotFoundError(error_message)

    entry_signal_list: List[str] = []
    exit_signal_list: List[str] = []
    with log_file_path.open("r", encoding="utf-8") as log_file:
        for line in log_file:
            stripped_line = line.strip()
            if stripped_line.startswith("entry_signals:"):
                symbol_list = stripped_line.split(":", 1)[1]
                entry_signal_list = [
                    symbol.strip()
                    for symbol in symbol_list.split(",")
                    if symbol.strip()
                ]
            elif stripped_line.startswith("exit_signals:"):
                symbol_list = stripped_line.split(":", 1)[1]
                exit_signal_list = [
                    symbol.strip()
                    for symbol in symbol_list.split(",")
                    if symbol.strip()
                ]
    return {
        "entry_signals": entry_signal_list,
        "exit_signals": exit_signal_list,
    }


def main() -> None:
    """Parse command line arguments and run the daily job."""
    parser = argparse.ArgumentParser(description="Run daily cron tasks")
    parser.add_argument(
        "argument_line",
        help=(
            "Task description: 'dollar_volume>NUMBER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', "
            "'dollar_volume=RANKth BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]', or "
            "'dollar_volume>NUMBER,RANKth BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]'"
        ),
    )
    parsed_arguments = parser.parse_args()
    run_daily_job(parsed_arguments.argument_line)


if __name__ == "__main__":
    main()
