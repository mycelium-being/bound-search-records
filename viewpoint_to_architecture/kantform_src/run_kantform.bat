@echo off
cd /d D:\mycelium\vta\nanoGPT_kantform
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=1337 --form_ckpt=D:\mycelium\vta\nanoGPT\out-form-s1337\ckpt.pt --out_dir=out-kantform-s1337 --eval_interval=250 --log_interval=250 --always_save_checkpoint=False > D:\mycelium\vta\kantform_s1337.log 2>&1
cd /d D:\mycelium\vta\nanoGPT_kantform
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=42 --form_ckpt=D:\mycelium\vta\nanoGPT\out-form-s42\ckpt.pt --out_dir=out-kantform-s42 --eval_interval=250 --log_interval=250 --always_save_checkpoint=False > D:\mycelium\vta\kantform_s42.log 2>&1
cd /d D:\mycelium\vta\nanoGPT_kantform
..\.venv\Scripts\python.exe train.py config\train_shakespeare_char.py --compile=False --seed=7 --form_ckpt=D:\mycelium\vta\nanoGPT\out-form-s7\ckpt.pt --out_dir=out-kantform-s7 --eval_interval=250 --log_interval=250 --always_save_checkpoint=False > D:\mycelium\vta\kantform_s7.log 2>&1
echo done > D:\mycelium\vta\kantform_batch.done
