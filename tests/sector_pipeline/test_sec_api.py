"""Tests for SEC API utilities with mocked HTTP requests."""

# TODO: review


def test_fetch_company_ticker_table(monkeypatch, company_ticker_payload) -> None:
    """The function should parse the company ticker table payload."""
    import stock_indicator.sector_pipeline.sec_api as sec_api_module

    class FakeResponse:
        def __init__(self, data: dict) -> None:
            self.data = data

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.data

    def fake_get(url: str, headers: dict | None = None, timeout: int = 30) -> 'FakeResponse':
        return FakeResponse(company_ticker_payload)

    monkeypatch.setattr(sec_api_module.requests, 'get', fake_get)

    data_frame = sec_api_module.fetch_company_ticker_table()
    assert set(data_frame['ticker']) == {'AAA', 'BBB'}
    assert data_frame.loc[data_frame['ticker'] == 'AAA', 'cik'].iloc[0] == 1


def test_fetch_submissions_json_uses_cache(monkeypatch, tmp_path, submissions_payloads) -> None:
    """Submissions should be cached after the first HTTP request."""
    import stock_indicator.sector_pipeline.sec_api as sec_api_module

    call_counter = {'count': 0}

    class FakeResponse:
        def __init__(self, payload: dict) -> None:
            self.payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return self.payload

    def fake_get(url: str, headers: dict | None = None, timeout: int = 30) -> 'FakeResponse':
        call_counter['count'] += 1
        return FakeResponse(submissions_payloads[1])

    monkeypatch.setattr(sec_api_module.requests, 'get', fake_get)
    monkeypatch.setattr(sec_api_module, 'SUBMISSIONS_DIRECTORY', tmp_path)

    first_result = sec_api_module.fetch_submissions_json(1, use_cache=True)
    second_result = sec_api_module.fetch_submissions_json(1, use_cache=True)

    assert first_result == submissions_payloads[1]
    assert second_result == submissions_payloads[1]
    assert call_counter['count'] == 1
