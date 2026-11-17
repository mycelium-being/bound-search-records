@echo off
cd /d D:\mycelium\vta\nanoGPT_k4
mkdir out-ret-k4rand-s7
copy /Y out-k4rand-s7\ckpt.pt out-ret-k4rand-s7\ckpt.pt
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=7 --init_from=resume --out_dir=out-ret-k4rand-s7 --dataset=pg_char --max_iters=6000 --lr_decay_iters=6000 --decay_lr=False --learning_rate=1e-4 --warmup_iters=0 --eval_interval=250 --log_interval=250 --eval_iters=200 --always_save_checkpoint=True --form_heads=all > D:\mycelium\vta\k4rand_ret_s7.log 2>&1
echo done > D:\mycelium\vta\k4randb_ret_batch.done
