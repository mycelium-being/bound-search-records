import sys
tag=sys.argv[1]; out='out-k4' if tag=='k4' else 'out-k4rand'
L=["@echo off","cd /d D:\\mycelium\\vta\\nanoGPT_k4"]
P=r"D:\mycelium\vta\.venv\Scripts\python.exe eval_ckpt.py"
for s in (1337,42,7):
    L.append(rf"{P} {out}-s{s}\ckpt.pt data\shakespeare_char info >> D:\mycelium\vta\eval_{tag}.txt 2>&1")
    L.append(rf"{P} {out}-s{s}\ckpt.pt data\shakespeare_char >> D:\mycelium\vta\eval_{tag}.txt 2>&1")
    L.append(rf"{P} out-ret-{tag}-s{s}\ckpt.pt data\shakespeare_char info >> D:\mycelium\vta\eval_{tag}.txt 2>&1")
    L.append(rf"{P} out-ret-{tag}-s{s}\ckpt.pt data\shakespeare_char >> D:\mycelium\vta\eval_{tag}.txt 2>&1")
    L.append(rf"{P} out-ret-{tag}-s{s}\ckpt.pt data\pg_char >> D:\mycelium\vta\eval_{tag}.txt 2>&1")
L.append(rf"echo done > D:\mycelium\vta\eval_{tag}.done")
open(rf'run_eval_{tag}.bat','w',newline='\r\n').write("\n".join(L)+"\n")
