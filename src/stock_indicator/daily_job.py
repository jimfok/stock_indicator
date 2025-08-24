"""Entry point for running the daily cron tasks."""
# TODO: review

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path
from typing import Dict

from . import cron

LOGGER = logging.getLogger(__name__)

DEFAULT_START_DATE = "2019-01-01"
DATA_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "data"
LOG_DIRECTORY = Path(__file__).resolve().parent.parent.parent / "logs"


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
    signal_result: Dict[str, list[str]] = cron.run_daily_tasks_from_argument(
        argument_line,
        start_date=DEFAULT_START_DATE,
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


def main() -> None:
    """Parse command line arguments and run the daily job."""
    parser = argparse.ArgumentParser(description="Run daily cron tasks")
    parser.add_argument(
        "argument_line",
        help=(
            "Task description: 'dollar_volume>NUMBER BUY_STRATEGY SELL_STRATEGY [STOP_LOSS]'"
        ),
    )
    parsed_arguments = parser.parse_args()
    run_daily_job(parsed_arguments.argument_line)


if __name__ == "__main__":
    main()
