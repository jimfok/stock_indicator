# TODO: review

set shell := ["bash", "-c"]

venv := ".venv/bin/activate"

default:
    @just --list

build:
    source {{venv}} && uv sync

test:
    source {{venv}} && uv run pytest

manage CMD="help":
    source {{venv}} && python -m stock_indicator.manage {{CMD}}

sector-init url output="data/symbols_with_sector.parquet":
    source {{venv}} && python -m stock_indicator.manage update_sector_data --ff-map-url={{url}} {{output}}

sector-refresh:
    ./scripts/update_data_cron.sh

data-full-backfill:
    HISTORICAL_START_DATE=1990-01-01 ./scripts/update_data_cron.sh

lint:
    # TODO: review add lint/format commands here
