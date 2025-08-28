import io, requests, pandas as pd
def load_ff_mapping(source:str)->pd.DataFrame:
    if source.startswith('http://') or source.startswith('https://'):
        r=requests.get(source, timeout=30); r.raise_for_status(); df=pd.read_csv(io.BytesIO(r.content))
    else:
        df=pd.read_csv(source)
    df.columns=[c.strip().lower() for c in df.columns]
    if 'sic' in df.columns and 'sic_start' not in df.columns:
        df['sic_start']=df['sic'].astype(int); df['sic_end']=df['sic'].astype(int)
    for c in ('ff12','ff48','ff49'):
        if c not in df.columns: df[c]=-1
    if 'label' not in df.columns: df['label']=''
    df['sic_start']=df['sic_start'].astype(int); df['sic_end']=df['sic_end'].astype(int)
    return df[['sic_start','sic_end','ff12','ff48','ff49','label']]
def build_sic_lookup(mapping_df:pd.DataFrame)->pd.DataFrame:
    rows=[]
    for _,r in mapping_df.iterrows():
        s,e=int(r.sic_start),int(r.sic_end)
        for sic in range(s,e+1):
            rows.append({'sic':sic,'ff12':int(r.ff12),'ff48':int(r.ff48),'ff49':int(r.ff49),'ff_label':r.label})
    return pd.DataFrame(rows)
def attach_ff_groups(df:pd.DataFrame, sic_lookup:pd.DataFrame)->pd.DataFrame:
    out=df.merge(sic_lookup, on='sic', how='left')
    out['ff12']=out['ff12'].fillna(-1).astype(int)
    out['ff48']=out['ff48'].fillna(-1).astype(int)
    out['ff49']=out['ff49'].fillna(-1).astype(int)
    out['ff_label']=out['ff_label'].fillna('UNKNOWN')
    return out
