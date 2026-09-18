import json,sys,numpy as np
for tag in sys.argv[1:]:
    ev=[json.loads(l) for l in open(f"out/events_{tag}.jsonl")][1:]
    meta=json.loads(open(f"out/events_{tag}.jsonl").readline())["meta"]
    res={}
    for panel in "LR":
        blk=[e for e in ev if e.get("would_swipe") and ((e["t"]<15) if panel=="L" else (e["t"]>=15))]
        n=len(blk); ok=sum(1 for e in blk if e["would_swipe"]==panel)
        res[panel]=(ok,n)
    tot_ok=sum(v[0] for v in res.values()); tot=sum(v[1] for v in res.values())
    print(f"{tag}: decoder side correct  L-block {res['L'][0]}/{res['L'][1]}  R-block {res['R'][0]}/{res['R'][1]}  overall {tot_ok}/{tot} = {tot_ok/max(1,tot):.2f}")
