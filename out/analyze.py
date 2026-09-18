import json,sys,numpy as np
tag=sys.argv[1]
ev=[json.loads(l) for l in open(f"out/events_{tag}.jsonl")][1:]
sw=[(round(e["t"],2),e["swipe"]) for e in ev if e["swipe"]]
t=np.array([e["t"] for e in ev]); dn=np.array([e["pops"]["dn_L"]+e["pops"]["dn_R"] for e in ev])
eyeL=np.array([sum(e["eye"]["L"]) for e in ev]); eyeR=np.array([sum(e["eye"]["R"]) for e in ev])
print(tag, "swipes:", len(sw), "L", sum(1 for _,s in sw if s=="L"), "R", sum(1 for _,s in sw if s=="R"), sw)
T=int(t[-1])+1
for i in range(0,T,5):
    m=(t>=i)&(t<i+5); print(f"{i:2d}s eye L {eyeL[m].mean():5.1f} R {eyeR[m].mean():5.1f} | dn max {dn[m].max():4d} mean {dn[m].mean():5.1f}")
sm=json.load(open(f"out/summary_{tag}.json")); print("posts", sm["posts"], "pop Hz %.2f"%sm["mean_pop_rate_hz"], "ms/step %.0f"%sm["ms_per_control_step"])
