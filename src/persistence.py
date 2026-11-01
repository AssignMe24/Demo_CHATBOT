from __future__ import annotations
import json,os,pickle,tempfile
from pathlib import Path
from typing import Any
import numpy as np

def save_chunks(chunks,path): _atomic_write(Path(path),pickle.dumps(chunks,protocol=pickle.HIGHEST_PROTOCOL))
def load_chunks(path):
    with open(path,"rb") as f:return pickle.load(f)
def save_embeddings(embeddings,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+".tmp.npy");np.save(tmp,embeddings);tmp.replace(path)
def load_embeddings(path): return np.load(path)
def save_manifest(manifest,path): _atomic_write(Path(path),json.dumps(manifest,indent=2,sort_keys=True).encode())
def load_manifest(path):
    try:return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:return None
def manifest_matches(path,expected):
    actual=load_manifest(path)
    return bool(actual and all(actual.get(k)==v for k,v in expected.items()))
def _atomic_write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=".tmp_")
    try:
        with os.fdopen(fd,"wb") as f:f.write(data)
        os.replace(tmp,path)
    except Exception:
        try:os.unlink(tmp)
        except OSError:pass
        raise
