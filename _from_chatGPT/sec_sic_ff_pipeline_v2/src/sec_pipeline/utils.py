import os, time, re, json, pathlib
def ensure_dir(path): os.makedirs(path, exist_ok=True)
def save_json(obj, path):
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path,'w',encoding='utf-8') as f: json.dump(obj,f,ensure_ascii=False,indent=2)
def load_json(path):
    with open(path,'r',encoding='utf-8') as f: return json.load(f)
def sleep_polite(seconds=0.12): time.sleep(seconds)
def normalize_ticker(t, dot_dash=True):
    t=(t or '').strip().upper()
    return re.sub(r'(?<=\w)-(?!$)', '.', t) if dot_dash else t
