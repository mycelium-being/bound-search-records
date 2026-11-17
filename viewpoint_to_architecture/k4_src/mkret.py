import re,sys
ONLY=sys.argv[1]
SEEDS=[int(x) for x in sys.argv[2].split(',')] if len(sys.argv)>2 else [1337,42,7]
SUF=sys.argv[3] if len(sys.argv)>3 else ''
def best(log):
    b=(9e9,0)
    for m in re.finditer(r"step (\d+): train loss [\d.]+, val loss ([\d.]+)",open(log).read()):
        it,v=int(m.group(1)),float(m.group(2))
        if v<b[0]: b=(v,it)
    return b[1]
L=["@echo off"]
for tag,out,form in (("k4","out-k4",True),("k4rand","out-k4rand",False)):
    for s in (1337,42,7):
        if tag!=ONLY or s not in SEEDS: continue
        it=best(rf"D:\mycelium\vta\{tag}_s{s}.log"); mx=it+2000
        L+= [r"cd /d D:\mycelium\vta\nanoGPT_k4", f"mkdir out-ret-{tag}-s{s}", f"copy /Y {out}-s{s}\\ckpt.pt out-ret-{tag}-s{s}\\ckpt.pt",
          rf"..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed={s} --init_from=resume --out_dir=out-ret-{tag}-s{s} --dataset=pg_char --max_iters={mx} --lr_decay_iters={mx} --decay_lr=False --learning_rate=1e-4 --warmup_iters=0 --eval_interval=250 --log_interval=250 --eval_iters=200 --always_save_checkpoint=True --form_heads=all"+(rf" --form_ckpt=D:\mycelium\vta\nanoGPT\out-form2-s{s}\ckpt.pt" if form else "")+rf" > D:\mycelium\vta\{tag}_ret_s{s}.log 2>&1"]
        print(tag,s,it)
L.append(rf"echo done > D:\mycelium\vta\{ONLY}{SUF}_ret_batch.done")
open(rf'D:\mycelium\vta\run_{ONLY}{SUF}_ret.bat','w',newline='\r\n').write("\n".join(L)+"\n")
