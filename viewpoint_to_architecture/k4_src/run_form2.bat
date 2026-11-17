@echo off
cd /d D:\mycelium\vta\nanoGPT
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=1337 --dataset=form2_char --max_iters=1500 --lr_decay_iters=1500 --eval_interval=250 --eval_iters=50 --log_interval=250 --out_dir=out-form2-s1337 --always_save_checkpoint=True > D:\mycelium\vta\form2_s1337.log 2>&1
cd /d D:\mycelium\vta\nanoGPT
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=42 --dataset=form2_char --max_iters=1500 --lr_decay_iters=1500 --eval_interval=250 --eval_iters=50 --log_interval=250 --out_dir=out-form2-s42 --always_save_checkpoint=True > D:\mycelium\vta\form2_s42.log 2>&1
cd /d D:\mycelium\vta\nanoGPT
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=7 --dataset=form2_char --max_iters=1500 --lr_decay_iters=1500 --eval_interval=250 --eval_iters=50 --log_interval=250 --out_dir=out-form2-s7 --always_save_checkpoint=True > D:\mycelium\vta\form2_s7.log 2>&1
echo done > D:\mycelium\vta\form2_batch.done
