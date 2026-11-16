import sys, os, numpy as np, torch
from contextlib import nullcontext
sys.path.insert(0, r'D:\mycelium\vta\nanoGPT_retention')
from model import GPTConfig, GPT
ckpt_path, data_dir = sys.argv[1], sys.argv[2]
BLOCK, BATCH, NB = 256, 64, 200
dev = 'cuda'
ck = torch.load(ckpt_path, map_location=dev)
if len(sys.argv) > 3 and sys.argv[3] == 'info':
    print('info', ckpt_path, 'iter_num', ck['iter_num'], 'best_val_loss', float(ck['best_val_loss']), 'model_args', ck['model_args'])
m = GPT(GPTConfig(**ck['model_args']))
sd = ck['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
m.load_state_dict(sd); m.to(dev); m.eval()
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)
data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
torch.manual_seed(1337); np.random.seed(1337)
tot = 0.0
with torch.no_grad():
    for _ in range(NB):
        ix = torch.randint(len(data) - BLOCK, (BATCH,))
        x = torch.stack([torch.from_numpy(data[i:i+BLOCK].astype(np.int64)) for i in ix]).to(dev)
        y = torch.stack([torch.from_numpy(data[i+1:i+1+BLOCK].astype(np.int64)) for i in ix]).to(dev)
        with ctx:
            _, loss = m(x, y)
        tot += loss.item()
print(f'eval {ckpt_path} {data_dir} {tot/NB:.4f}')
