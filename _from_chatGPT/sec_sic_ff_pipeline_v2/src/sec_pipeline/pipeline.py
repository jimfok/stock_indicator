import pandas as pd
import requests
from .config import CACHE_DIR, SUBMISSIONS_DIR, LAST_RUN_CONFIG, DEFAULT_OUT_PARQUET, DEFAULT_OUT_CSV
from .utils import ensure_dir, save_json, load_json
from .sec_api import build_ticker_cik_sic
from .ff_mapping import load_ff_mapping, build_sic_lookup, attach_ff_groups
def load_universe(source:str)->pd.DataFrame:
    if source.endswith('.csv'):
        df=pd.read_csv(source); assert 'ticker' in df.columns
        return df[['ticker']].dropna().drop_duplicates()
    if source.startswith('http://') or source.startswith('https://'):
        r=requests.get(source, timeout=30); r.raise_for_status(); text=r.text
    else:
        with open(source,'r',encoding='utf-8') as f: text=f.read()
    tickers=[t.strip().upper() for t in text.splitlines() if t.strip()]
    return pd.DataFrame({'ticker':tickers})
def build(symbols_source:str, ff_map_source:str, out_parquet:str=DEFAULT_OUT_PARQUET, out_csv:str=DEFAULT_OUT_CSV)->pd.DataFrame:
    ensure_dir(CACHE_DIR); ensure_dir(SUBMISSIONS_DIR)
    uni=load_universe(symbols_source)
    tcs=build_ticker_cik_sic(uni)
    mapping=load_ff_mapping(ff_map_source)
    lookup=build_sic_lookup(mapping)
    classified=attach_ff_groups(tcs, lookup)
    classified['sic_desc']=''
    classified.to_parquet(out_parquet, index=False)
    if out_csv: classified.to_csv(out_csv, index=False)
    save_json({'symbols_source':symbols_source,'ff_map_source':ff_map_source,'out':out_parquet}, LAST_RUN_CONFIG)
    return classified
def update()->pd.DataFrame:
    cfg=load_json(LAST_RUN_CONFIG)
    return build(cfg['symbols_source'], cfg['ff_map_source'], cfg.get('out', DEFAULT_OUT_PARQUET))
def coverage_report(df:pd.DataFrame)->str:
    total=len(df); with_cik=df['cik'].notna().sum(); with_sic=df['sic'].notna().sum(); with_ff=(df['ff48']!=-1).sum()
    return f'Total: {total}\nCIK: {with_cik} ({with_cik/total:.1%})\nSIC: {with_sic} ({with_sic/total:.1%})\nFF tag: {with_ff} ({with_ff/total:.1%})'
