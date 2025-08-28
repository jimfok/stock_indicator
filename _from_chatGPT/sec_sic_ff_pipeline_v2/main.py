import argparse, sys, pandas as pd
from src.sec_pipeline.pipeline import build, update, coverage_report
from src.sec_pipeline.config import DEFAULT_OUT_PARQUET
def main(argv=None):
    p=argparse.ArgumentParser(description='SEC→SIC→FF pipeline')
    sp=p.add_subparsers(dest='cmd', required=True)
    pb=sp.add_parser('build'); pb.add_argument('--symbols-url', required=True); pb.add_argument('--ff-map-url', required=True); pb.add_argument('--out', default=DEFAULT_OUT_PARQUET)
    sp.add_parser('update'); sp.add_parser('verify')
    a=p.parse_args(argv)
    if a.cmd=='build':
        df=build(a.symbols_url, a.ff_map_url, a.out); print(coverage_report(df))
    elif a.cmd=='update':
        df=update(); print(coverage_report(df))
    elif a.cmd=='verify':
        try:
            df=pd.read_parquet(DEFAULT_OUT_PARQUET); print(coverage_report(df))
        except Exception as e:
            print(f'Failed to load {DEFAULT_OUT_PARQUET}: {e}', file=sys.stderr); sys.exit(1)
if __name__=='__main__': main()
