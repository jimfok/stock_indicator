"""Tests for Fama-French mapping utilities with mocked downloads."""

# TODO: review


def test_load_fama_french_mapping_downloads_csv(monkeypatch, ff_mapping_csv_text) -> None:
    """The loader should fetch and parse a mapping CSV from a URL."""
    import stock_indicator.sector_pipeline.ff_mapping as ff_mapping_module

    class FakeResponse:
        def __init__(self, text: str) -> None:
            self.content = text.encode('utf-8')

        def raise_for_status(self) -> None:
            return None

    def fake_get(url: str, timeout: int = 30) -> 'FakeResponse':
        return FakeResponse(ff_mapping_csv_text)

    monkeypatch.setattr(ff_mapping_module.requests, 'get', fake_get)

    data_frame = ff_mapping_module.load_fama_french_mapping('http://example.com/mapping.csv')
    assert list(data_frame.columns) == ['sic_start', 'sic_end', 'ff12', 'ff48', 'ff49', 'label']
    assert data_frame.iloc[0]['ff48'] == 10
