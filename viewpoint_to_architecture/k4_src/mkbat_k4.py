L=["@echo off"]
for s in (1337,42,7):
    L.append(r"cd /d D:\mycelium\vta\nanoGPT_k4")
    L.append(rf"..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed={s} --form_heads=all --form_ckpt=D:\mycelium\vta\nanoGPT\out-form2-s{s}\ckpt.pt --out_dir=out-k4-s{s} --eval_interval=250 --log_interval=250 --always_save_checkpoint=False > D:\mycelium\vta\k4_s{s}.log 2>&1")
for s in (1337,42,7):
    L.append(r"cd /d D:\mycelium\vta\nanoGPT_k4")
    L.append(rf"..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed={s} --form_heads=all --out_dir=out-k4rand-s{s} --eval_interval=250 --log_interval=250 --always_save_checkpoint=False > D:\mycelium\vta\k4rand_s{s}.log 2>&1")
L.append(r"echo done > D:\mycelium\vta\k4_batch.done")
open(r'D:\mycelium\vta\run_k4.bat','w',newline='\r\n').write("\n".join(L)+"\n")
