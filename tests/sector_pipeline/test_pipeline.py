"""Tests for building and updating the sector classification dataset."""

# TODO: review

from pathlib import Path

import pandas as pd


def test_build_and_update_dataset(monkeypatch, tmp_path, submissions_payloads, ff_mapping_file) -> None:
    """The pipeline should build and refresh classification data."""
    import stock_indicator.sector_pipeline.sec_api as sec_api_module
    import stock_indicator.sector_pipeline.pipeline as pipeline_module

    def fake_load_company_tickers() -> pd.DataFrame:
        return pd.DataFrame([
            {"ticker": "AAA", "cik": 1},
            {"ticker": "BBB", "cik": 2},
        ])

    def fake_fetch_submissions_json(central_index_key: int, use_cache: bool = True) -> dict:
        return submissions_payloads[central_index_key]

    monkeypatch.setattr(sec_api_module, 'load_company_tickers', fake_load_company_tickers)
    monkeypatch.setattr(sec_api_module, 'fetch_submissions_json', fake_fetch_submissions_json)
    monkeypatch.setattr(sec_api_module, 'SUBMISSIONS_DIRECTORY', tmp_path / 'submissions')

    monkeypatch.setattr(pipeline_module, 'load_company_tickers', fake_load_company_tickers)
    monkeypatch.setattr(pipeline_module, 'LAST_RUN_CONFIG_PATH', tmp_path / 'last_run.json')
    monkeypatch.setattr(pipeline_module, 'DEFAULT_OUTPUT_PARQUET_PATH', tmp_path / 'output.parquet')
    monkeypatch.setattr(pipeline_module, 'DEFAULT_OUTPUT_CSV_PATH', tmp_path / 'output.csv')
    monkeypatch.setattr(pipeline_module, 'SUBMISSIONS_DIRECTORY', tmp_path / 'submissions')
    monkeypatch.setattr(pd.DataFrame, 'to_parquet', lambda self, *a, **k: None)

    data_frame = pipeline_module.build_sector_classification_dataset(
        ff_mapping_file, tmp_path / 'output.parquet', tmp_path / 'output.csv'
    )
    assert set(data_frame['ff48']) == {10, 20}
    assert (tmp_path / 'last_run.json').exists()

    updated_data_frame = pipeline_module.update_latest_dataset()
    assert len(updated_data_frame) == 2
