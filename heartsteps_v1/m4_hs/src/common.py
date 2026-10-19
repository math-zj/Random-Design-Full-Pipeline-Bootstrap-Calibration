"""Isolated M4-HS storage and exact parent adapter reuse."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
from types import FunctionType
import numpy as np
import pandas as pd
from heartsteps_v1.scripts import adapter

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'heartsteps_v1/m4_hs'
PARENT = ROOT / 'heartsteps_v1'
CFG = json.loads((BASE/'specification/config.json').read_text())
CANDIDATES = (12096,18144,24192,30240,36288,42336,48384)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()

def clean(x):
    if isinstance(x, dict): return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(tmp,path)

def protected_check():
    old=json.loads((PARENT/'hash_registry_final.json').read_text())['files']
    core=json.loads((PARENT/'diagnostics/protected_source_hashes_before.json').read_text())
    changed=[str(PARENT/p) for p,h in old.items() if sha(PARENT/p)!=h]
    changed += [str(ROOT/p) for p,h in core.items() if sha(ROOT/p)!=h]
    assert not changed, changed
    return {'phase1_files_checked':len(old),'core_files_checked':len(core),'changed':changed}

class Row(dict):
    def __getattr__(self, name): return self[name]

class ILoc:
    def __init__(self, frame): self.frame=frame
    def __getitem__(self, key):
        if isinstance(key,slice): return Frame(self.frame.rows[key])
        return self.frame.rows[int(key)]

class Frame:
    """Read-only pandas-access facade. No transition logic lives here."""
    def __init__(self, rows): self.rows=rows; self.iloc=ILoc(self)
    @classmethod
    def from_df(cls,d): return cls([Row(x) for x in d.to_dict('records')])
    def __len__(self): return len(self.rows)
    def __getitem__(self,key):
        vals=[x[key] for x in self.rows]
        if key.endswith('.utime'): return pd.Series(vals,dtype='datetime64[ns, UTC]')
        return pd.Series(vals)
    def __getattr__(self,key): return self[key]
    def groupby(self,key):
        groups={}
        for r in self.rows: groups.setdefault(r[key],[]).append(r)
        return [(k,Frame(v)) for k,v in sorted(groups.items())]
    def reset_index(self,drop=True): return self
    def iterrows(self): return enumerate(self.rows)

def rebuild(d, fast=True):
    data=Frame.from_df(d) if fast else d
    bindings=dict(adapter.build.__globals__)
    bindings['load_suggestions']=lambda:data
    build=FunctionType(adapter.build.__code__,bindings,adapter.build.__name__,adapter.build.__defaults__)
    assert build.__code__ is adapter.build.__code__
    return build(save=False)

def screen(panel):
    from src.rs_cusum.support import load_support_rule,screen_candidates
    c=adapter.config()['analysis']
    rule=load_support_rule(ROOT/c['support_config'],c['support_rule'])
    sc=screen_candidates(panel.support_view(),CANDIDATES,0,60480,rule)
    rows=[]
    for v in sc.candidates:
        row={'candidate':v.candidate,'admissible':v.admissible,'reasons':'|'.join(v.failure_reasons)}
        for side in [v.left,v.right]:
            row.update({side.side+'_n':side.transition_count,side.side+'_A0':side.action_counts[0],side.side+'_A1':side.action_counts[1],side.side+'_patients':len(side.active_patients),side.side+'_dominance_A1':side.maximum_action_patient_share[1]})
        rows.append(row)
    return sc,rows
