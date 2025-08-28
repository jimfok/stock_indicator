import pandas as pd

from stock_indicator.sector_pipeline.utils import normalize_ticker_symbol
from stock_indicator.sector_pipeline.ff_mapping import (
    build_classification_lookup,
    attach_fama_french_groups,
)


def test_normalize_ticker_symbol_converts_dash_to_dot():
    assert normalize_ticker_symbol("brk-b") == "BRK.B"


def test_build_classification_lookup_expands_ranges():
    mapping_data_frame = pd.DataFrame(
        {
            "sic_start": [1000],
            "sic_end": [1002],
            "ff12": [1],
            "ff48": [2],
            "ff49": [3],
            "label": ["Test"],
        }
    )
    lookup_data_frame = build_classification_lookup(mapping_data_frame)
    assert len(lookup_data_frame) == 3
    assert (
        lookup_data_frame.loc[lookup_data_frame["sic"] == 1001, "ff48"].iloc[0] == 2
    )


def test_attach_fama_french_groups_merges_lookup():
    data_frame = pd.DataFrame({"sic": [1001]})
    lookup_data_frame = pd.DataFrame(
        {
            "sic": [1001],
            "ff12": [1],
            "ff48": [2],
            "ff49": [3],
            "ff_label": ["Label"],
        }
    )
    merged_data_frame = attach_fama_french_groups(data_frame, lookup_data_frame)
    assert merged_data_frame["ff48"].iloc[0] == 2
    assert merged_data_frame["ff_label"].iloc[0] == "Label"
