"""Build the embedded data pack for the Tape Room trainer."""
import json, os

SP = os.environ.get('SP', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FX = f"{SP}/fx"

eur_raw = json.load(open(f"{FX}/eurusd.json"))['rates']
dates = sorted(eur_raw)
eur = [round(eur_raw[d]['USD'], 5) for d in dates]

def sa_map(path):
    rows = json.load(open(path))['data']
    return {r['t']: float(r['a']) for r in rows}

uup_m, gld_m = sa_map(f"{FX}/uup.json"), sa_map(f"{FX}/gld.json")
y_raw = json.load(open(f"{FX}/us10y.json"))['data']
y_m = {r['date']: float(r['value']) for r in y_raw if r['value'] not in ('.', '', None)}

def align(m, nd):
    out, last = [], None
    for d in dates:
        if d in m: last = m[d]
        out.append(round(last, nd) if last is not None else None)
    return out

pack = dict(
    d=[int(d.replace('-', '')) for d in dates],
    e=eur,
    u=align(uup_m, 3),
    g=align(gld_m, 2),
    y=align(y_m, 2),
)
js = json.dumps(pack, separators=(',', ':'))
open(f"{FX}/pack.json", 'w').write(js)
first_u = next(i for i, v in enumerate(pack['u']) if v is not None)
print(f"days {len(dates)}  {dates[0]}..{dates[-1]}  pack {len(js)//1024}KB  "
      f"uup starts idx {first_u} ({dates[first_u]})")
