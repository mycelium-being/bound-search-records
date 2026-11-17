@echo off
cd /d D:\mycelium\vta\nanoGPT_k4
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=1337 --form_heads=all --form_ckpt=D:\mycelium\vta\nanoGPT\out-form2-s1337\ckpt.pt --out_dir=out-k4-s1337 --eval_interval=250 --log_interval=250 --always_save_checkpoint=False > D:\mycelium\vta\k4_s1337.log 2>&1
cd /d D:\mycelium\vta\nanoGPT_k4
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=42 --form_heads=all --form_ckpt=D:\mycelium\vta\nanoGPT\out-form2-s42\ckpt.pt --out_dir=out-k4-s42 --eval_interval=250 --log_interval=250 --always_save_checkpoint=False > D:\mycelium\vta\k4_s42.log 2>&1
cd /d D:\mycelium\vta\nanoGPT_k4
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=7 --form_heads=all --form_ckpt=D:\mycelium\vta\nanoGPT\out-form2-s7\ckpt.pt --out_dir=out-k4-s7 --eval_interval=250 --log_interval=250 --always_save_checkpoint=False > D:\mycelium\vta\k4_s7.log 2>&1
cd /d D:\mycelium\vta\nanoGPT_k4
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=1337 --form_heads=all --out_dir=out-k4rand-s1337 --eval_interval=250 --log_interval=250 --always_save_checkpoint=False > D:\mycelium\vta\k4rand_s1337.log 2>&1
cd /d D:\mycelium\vta\nanoGPT_k4
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=42 --form_heads=all --out_dir=out-k4rand-s42 --eval_interval=250 --log_interval=250 --always_save_checkpoint=False > D:\mycelium\vta\k4rand_s42.log 2>&1
cd /d D:\mycelium\vta\nanoGPT_k4
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=7 --form_heads=all --out_dir=out-k4rand-s7 --eval_interval=250 --log_interval=250 --always_save_checkpoint=False > D:\mycelium\vta\k4rand_s7.log 2>&1
echo done > D:\mycelium\vta\k4_batch.done
