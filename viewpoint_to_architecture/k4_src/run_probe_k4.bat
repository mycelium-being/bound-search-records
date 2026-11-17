@echo off
cd /d D:\mycelium\vta\nanoGPT_k4
D:\mycelium\vta\.venv\Scripts\python.exe probe_head.py --ckpt D:\mycelium\vta\nanoGPT_k4\out-k4-s1337\ckpt.pt --data D:\mycelium\vta\nanoGPT\data\shakespeare_char --out D:\mycelium\vta\probe_k4_s1337.json --batches 10 > D:\mycelium\vta\probe_k4_s1337.log 2>&1
cd /d D:\mycelium\vta\nanoGPT_k4
D:\mycelium\vta\.venv\Scripts\python.exe probe_head.py --ckpt D:\mycelium\vta\nanoGPT_k4\out-k4-s42\ckpt.pt --data D:\mycelium\vta\nanoGPT\data\shakespeare_char --out D:\mycelium\vta\probe_k4_s42.json --batches 10 > D:\mycelium\vta\probe_k4_s42.log 2>&1
cd /d D:\mycelium\vta\nanoGPT_k4
D:\mycelium\vta\.venv\Scripts\python.exe probe_head.py --ckpt D:\mycelium\vta\nanoGPT_k4\out-k4-s7\ckpt.pt --data D:\mycelium\vta\nanoGPT\data\shakespeare_char --out D:\mycelium\vta\probe_k4_s7.json --batches 10 > D:\mycelium\vta\probe_k4_s7.log 2>&1
echo done > D:\mycelium\vta\probe_k4.done
