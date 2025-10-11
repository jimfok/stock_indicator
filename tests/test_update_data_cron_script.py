from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Iterable


def _create_fake_python(executable_path: Path, *, log_path: Path) -> None:
    """Create a fake python executable that logs incoming commands."""
    script_contents = f"""#!/bin/bash
echo "$0 $@" >> "{log_path}"
"""
    executable_path.write_text(script_contents, encoding="utf-8")
    executable_path.chmod(0o755)


def _run_script(script_path: Path, *, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Execute the cron script with the supplied environment."""
    return subprocess.run(
        ["/bin/bash", str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )


def _read_log_lines(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    return log_path.read_text(encoding="utf-8").splitlines()


def _create_stub_environment(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    """Prepare a temporary environment with fake python and log files."""
    scripts_directory = tmp_path / "scripts"
    scripts_directory.mkdir()
    log_directory = tmp_path / "logs"
    log_directory.mkdir()
    cron_script_path = scripts_directory / "update_data_cron.sh"
    source_script_path = Path(__file__).resolve().parents[1] / "scripts" / "update_data_cron.sh"
    cron_script_path.write_text(source_script_path.read_text(encoding="utf-8"), encoding="utf-8")
    cron_script_path.chmod(0o755)
    fake_python_path = tmp_path / "python"
    call_log_path = tmp_path / "python_calls.log"
    _create_fake_python(fake_python_path, log_path=call_log_path)
    log_path = tmp_path / "logs" / "update_data_pipeline.log"
    environment = {
        **os.environ,
        "PATH": f"{tmp_path}:{os.environ.get('PATH', '')}",
        "VIRTUAL_ENV": "",
        "PYTHONHOME": "",
    }
    environment.pop("HISTORICAL_START_DATE", None)
    environment.pop("HISTORICAL_END_DATE", None)
    return call_log_path, log_path, cron_script_path, environment


def _assert_command_sequence(
    command_lines: Iterable[str],
    expected_start: str,
    expected_end: str,
) -> None:
    command_list = list(command_lines)
    assert any("python -m stock_indicator.manage update_sector_data" in line for line in command_list)
    assert any("python -m stock_indicator.manage update_symbols" in line for line in command_list)
    all_data_commands = [
        line for line in command_list if "python -m stock_indicator.manage update_all_data_from_yf" in line
    ]
    assert all_data_commands, "update_all_data_from_yf should be invoked exactly once"
    assert all_data_commands[0].endswith(f" {expected_start} {expected_end}")


def test_cron_script_defaults_to_prior_year(tmp_path: Path) -> None:
    """The cron script should default to the prior calendar year range."""
    call_log_path, log_path, cron_script_path, environment = _create_stub_environment(tmp_path)
    environment.pop("HISTORICAL_START_DATE", None)
    environment.pop("HISTORICAL_END_DATE", None)

    _run_script(cron_script_path, env=environment)

    log_lines = _read_log_lines(log_path)
    assert any("weekly mode: refreshing data from the prior calendar year" in line for line in log_lines)

    current_year = int(subprocess.check_output(["date", "-u", "+%Y"], text=True).strip())
    prior_year_start = f"{current_year - 1}-01-01"
    today = subprocess.check_output(["date", "-u", "+%Y-%m-%d"], text=True).strip()
    _assert_command_sequence(_read_log_lines(call_log_path), expected_start=prior_year_start, expected_end=today)


def test_cron_script_honors_custom_start(tmp_path: Path) -> None:
    """The cron script should respect HISTORICAL_START_DATE overrides."""
    call_log_path, log_path, cron_script_path, environment = _create_stub_environment(tmp_path)
    environment["HISTORICAL_START_DATE"] = "1990-01-01"

    _run_script(cron_script_path, env=environment)

    log_lines = _read_log_lines(log_path)
    assert any("custom range requested starting at 1990-01-01" in line for line in log_lines)

    today = subprocess.check_output(["date", "-u", "+%Y-%m-%d"], text=True).strip()
    _assert_command_sequence(_read_log_lines(call_log_path), expected_start="1990-01-01", expected_end=today)
