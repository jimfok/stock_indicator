import os, requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import pandas as pd
from .config import SEC_COMPANY_TICKERS_URL, SEC_SUBMISSIONS_URL_TMPL, SEC_USER_AGENT, SUBMISSIONS_DIR
from .utils import ensure_dir, save_json, load_json, sleep_polite, normalize_ticker

HEADERS={'User-Agent': SEC_USER_AGENT}
def _pad_cik(cik:int)->str: return f"{int(cik):010d}"
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, min=0.5, max=8), retry=retry_if_exception_type((requests.RequestException,)))
def _http_json(url:str):
    r=requests.get(url, headers=HEADERS, timeout=30); r.raise_for_status(); return r.json()
def fetch_company_tickers()->pd.DataFrame:
    data=_http_json(SEC_COMPANY_TICKERS_URL)
    rows=[{'ticker':normalize_ticker(v['ticker']),'cik':int(v['cik_str']),'title':v.get('title')} for v in data.values()]
    return pd.DataFrame(rows)
def _submissions_cache_path(cik:int)->str: return os.path.join(SUBMISSIONS_DIR, f"CIK{_pad_cik(cik)}.json")
def fetch_submissions_json(cik:int, use_cache=True):
    ensure_dir(SUBMISSIONS_DIR); p=_submissions_cache_path(cik)
    if use_cache and os.path.exists(p):
        try: return load_json(p)
        except Exception: pass
    url=SEC_SUBMISSIONS_URL_TMPL.format(cik_padded=_pad_cik(cik))
    data=_http_json(url); save_json(data,p); sleep_polite(); return data
def extract_sic_from_submissions(data)->int|None:
    if not isinstance(data,dict): return None
    sic=data.get('sic');
    try: return int(sic) if sic is not None else None
    except Exception: return None
def build_ticker_cik_sic(universe:pd.DataFrame)->pd.DataFrame:
    sec_map=fetch_company_tickers(); uni=universe.copy(); uni['ticker_norm']=uni['ticker'].map(normalize_ticker)
    sec_map['ticker_norm']=sec_map['ticker']
    df=uni.merge(sec_map[['ticker_norm','cik']], on='ticker_norm', how='left').drop(columns=['ticker_norm']).rename(columns={'ticker':'ticker'})
    unique_ciks=sorted(x for x in df['cik'].dropna().unique())
    rows=[]
    for cik in unique_ciks:
        js=fetch_submissions_json(int(cik), use_cache=True)
        sic=extract_sic_from_submissions(js) if js else None
        rows.append({'cik':int(cik),'sic':int(sic) if sic is not None else None})
    cik_sic=pd.DataFrame(rows)
    return df.merge(cik_sic, on='cik', how='left')
