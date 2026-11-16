@echo off
cd /d D:\mycelium\vta\nanoGPT_naval
D:\mycelium\vta\.venv\Scripts\python.exe probe_head.py --ckpt D:\mycelium\vta\nanoGPT_naval\out-naval-s7\ckpt.pt --data D:\mycelium\vta\nanoGPT\data\shakespeare_char --out D:\mycelium\vta\probe_naval_s7.json --batches 10 > D:\mycelium\vta\probe_naval_s7.log 2>&1
echo done > D:\mycelium\vta\probe_n7_batch.done
