@echo off
cd /d D:\mycelium\vta\nanoGPT_k4
mkdir out-ret-k4-s1337
copy /Y out-k4-s1337\ckpt.pt out-ret-k4-s1337\ckpt.pt
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=1337 --init_from=resume --out_dir=out-ret-k4-s1337 --dataset=pg_char --max_iters=5500 --lr_decay_iters=5500 --decay_lr=False --learning_rate=1e-4 --warmup_iters=0 --eval_interval=250 --log_interval=250 --eval_iters=200 --always_save_checkpoint=True --form_heads=all --form_ckpt=D:\mycelium\vta\nanoGPT\out-form2-s1337\ckpt.pt > D:\mycelium\vta\k4_ret_s1337.log 2>&1
cd /d D:\mycelium\vta\nanoGPT_k4
mkdir out-ret-k4-s42
copy /Y out-k4-s42\ckpt.pt out-ret-k4-s42\ckpt.pt
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=42 --init_from=resume --out_dir=out-ret-k4-s42 --dataset=pg_char --max_iters=4250 --lr_decay_iters=4250 --decay_lr=False --learning_rate=1e-4 --warmup_iters=0 --eval_interval=250 --log_interval=250 --eval_iters=200 --always_save_checkpoint=True --form_heads=all --form_ckpt=D:\mycelium\vta\nanoGPT\out-form2-s42\ckpt.pt > D:\mycelium\vta\k4_ret_s42.log 2>&1
cd /d D:\mycelium\vta\nanoGPT_k4
mkdir out-ret-k4-s7
copy /Y out-k4-s7\ckpt.pt out-ret-k4-s7\ckpt.pt
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=7 --init_from=resume --out_dir=out-ret-k4-s7 --dataset=pg_char --max_iters=5250 --lr_decay_iters=5250 --decay_lr=False --learning_rate=1e-4 --warmup_iters=0 --eval_interval=250 --log_interval=250 --eval_iters=200 --always_save_checkpoint=True --form_heads=all --form_ckpt=D:\mycelium\vta\nanoGPT\out-form2-s7\ckpt.pt > D:\mycelium\vta\k4_ret_s7.log 2>&1
echo done > D:\mycelium\vta\k4_ret_batch.done
