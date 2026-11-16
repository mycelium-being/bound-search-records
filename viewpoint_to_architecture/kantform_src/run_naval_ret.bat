@echo off
cd /d D:\mycelium\vta\nanoGPT_naval
mkdir out-ret-naval-s1337
copy /Y out-naval-s1337\ckpt.pt out-ret-naval-s1337\ckpt.pt
echo start %time% > D:\mycelium\vta\naval_ret_s1337.start
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=1337 --init_from=resume --out_dir=out-ret-naval-s1337 --dataset=pg_char --max_iters=3750 --lr_decay_iters=3750 --decay_lr=False --learning_rate=1e-4 --warmup_iters=0 --eval_interval=250 --log_interval=250 --eval_iters=200 --always_save_checkpoint=True > D:\mycelium\vta\naval_ret_s1337.log 2>&1
echo end %time% > D:\mycelium\vta\naval_ret_s1337.end
cd /d D:\mycelium\vta\nanoGPT_naval
mkdir out-ret-naval-s42
copy /Y out-naval-s42\ckpt.pt out-ret-naval-s42\ckpt.pt
echo start %time% > D:\mycelium\vta\naval_ret_s42.start
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=42 --init_from=resume --out_dir=out-ret-naval-s42 --dataset=pg_char --max_iters=3750 --lr_decay_iters=3750 --decay_lr=False --learning_rate=1e-4 --warmup_iters=0 --eval_interval=250 --log_interval=250 --eval_iters=200 --always_save_checkpoint=True > D:\mycelium\vta\naval_ret_s42.log 2>&1
echo end %time% > D:\mycelium\vta\naval_ret_s42.end
cd /d D:\mycelium\vta\nanoGPT_naval
mkdir out-ret-naval-s7
copy /Y out-naval-s7\ckpt.pt out-ret-naval-s7\ckpt.pt
echo start %time% > D:\mycelium\vta\naval_ret_s7.start
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=7 --init_from=resume --out_dir=out-ret-naval-s7 --dataset=pg_char --max_iters=3750 --lr_decay_iters=3750 --decay_lr=False --learning_rate=1e-4 --warmup_iters=0 --eval_interval=250 --log_interval=250 --eval_iters=200 --always_save_checkpoint=True > D:\mycelium\vta\naval_ret_s7.log 2>&1
echo end %time% > D:\mycelium\vta\naval_ret_s7.end
echo done > D:\mycelium\vta\naval_ret_batch.done
