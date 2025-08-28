import argparse
import sys
from typing import List

import pandas as pd

from stock_indicator.sector_pipeline.pipeline import (
    build_sector_classification_dataset,
    update_latest_dataset,
    generate_coverage_report,
)
from stock_indicator.sector_pipeline.config import DEFAULT_OUTPUT_PARQUET_PATH


def main(argument_list: List[str] | None = None) -> None:
    """Command-line interface for the SEC→SIC→Fama-French pipeline."""
    parser = argparse.ArgumentParser(description="SEC→SIC→FF pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--ff-map-url", required=True)
    build_parser.add_argument("--out", default=DEFAULT_OUTPUT_PARQUET_PATH)

    subparsers.add_parser("update")
    subparsers.add_parser("verify")

    arguments = parser.parse_args(argument_list)

    if arguments.command == "build":
        data_frame = build_sector_classification_dataset(
            arguments.ff_map_url, arguments.out
        )
        print(generate_coverage_report(data_frame))
    elif arguments.command == "update":
        data_frame = update_latest_dataset()
        print(generate_coverage_report(data_frame))
    elif arguments.command == "verify":
        try:
            data_frame = pd.read_parquet(DEFAULT_OUTPUT_PARQUET_PATH)
            print(generate_coverage_report(data_frame))
        except (OSError, ValueError) as error:
            print(
                f"Failed to load {DEFAULT_OUTPUT_PARQUET_PATH}: {error}",
                file=sys.stderr,
            )
            sys.exit(1)


if __name__ == "__main__":
    main()
