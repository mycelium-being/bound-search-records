L=["@echo off"]
for s in (1337,42,7):
    L.append(r"cd /d D:\mycelium\vta\nanoGPT")
    L.append(rf"..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed={s} --dataset=form_char --max_iters=1000 --lr_decay_iters=1000 --eval_interval=250 --eval_iters=50 --log_interval=250 --out_dir=out-form-s{s} --always_save_checkpoint=True > D:\mycelium\vta\form_s{s}.log 2>&1")
L.append(r"echo done > D:\mycelium\vta\form_batch.done")
open(r'D:\mycelium\vta\run_form.bat','w',newline='\r\n').write("\n".join(L)+"\n")
