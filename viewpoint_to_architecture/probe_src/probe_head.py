import sys, os, json, argparse, time
import numpy as np, torch
from contextlib import nullcontext
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import GPTConfig, GPT

ap = argparse.ArgumentParser()
ap.add_argument('--ckpt', required=True); ap.add_argument('--data', required=True)
ap.add_argument('--out', required=True); ap.add_argument('--batches', type=int, default=10)
a = ap.parse_args()
BLOCK, BATCH, SEED = 256, 64, 1337
RIDGE_PER_ROW, LR, STEPS = 1e-3, 1e-2, 500
dev = 'cuda'
t0 = time.time()
ck = torch.load(a.ckpt, map_location=dev)
m = GPT(GPTConfig(**ck['model_args'])); sd = ck['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
m.load_state_dict(sd); m.to(dev); m.eval()
NL = ck['model_args']['n_layer']; NH = ck['model_args']['n_head']; HS = ck['model_args']['n_embd'] // NH
dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
ctx = torch.amp.autocast(device_type='cuda', dtype=dtype)
store = [[] for _ in range(NL)]; calls = [0] * NL
def mk(L):
    def hook(mod, inp):
        x = inp[0]; assert x.shape[-1] == NH * HS and x.dim() == 3
        calls[L] += 1
        store[L].append(x.detach().float().half().cpu())
    return hook
for L, blk in enumerate(m.transformer.h):
    blk.attn.c_proj.register_forward_pre_hook(mk(L))
data = np.memmap(os.path.join(a.data, 'val.bin'), dtype=np.uint16, mode='r')
torch.manual_seed(SEED); np.random.seed(SEED)
idxs = []
with torch.no_grad():
    for _ in range(a.batches):
        ix = torch.randint(len(data) - BLOCK, (BATCH,))
        x = torch.stack([torch.from_numpy(data[i:i+BLOCK].astype(np.int64)) for i in ix]).to(dev)
        with ctx:
            m(x, x)
        idxs.append(x.cpu())
assert all(c == a.batches for c in calls), calls
nfit = max(1, a.batches - 2)
idx = torch.stack(idxs)  # (nb, B, T)
tpos = torch.arange(BLOCK).expand(BATCH, BLOCK)
def split(t):  # t: (nb,B,T,...) -> fit, held
    return t[:nfit], t[nfit:]
def flat_mask(mask_t0):
    return None
prev = torch.zeros_like(idx); prev[:, :, 1:] = idx[:, :, :-1]
valid = torch.ones_like(idx, dtype=torch.bool)
pos = tpos.expand_as(idx).float()
vprev = valid.clone(); vprev[:, :, 0] = False
tg = {'position': (pos, valid), 'current': (idx, valid), 'previous': (prev, vprev)}
rng = np.random.default_rng(SEED)
fitm = {}; heldm = {}; Y = {}; perm = {}
for name, (y, v) in tg.items():
    yf, yh = split(y); vf, vh = split(v)
    Y[name] = (yf[vf].to(dev), yh[vh].to(dev)); fitm[name] = vf; heldm[name] = vh
    perm[name] = torch.from_numpy(rng.permutation(int(vf.sum()))).to(dev)

def ridge(Xf, yf, Xh, yh):
    n = Xf.shape[0]; Xd = Xf.double(); ym = yf.mean()
    A = Xd.T @ Xd + RIDGE_PER_ROW * n * torch.eye(Xd.shape[1], device=dev, dtype=torch.float64)
    w = torch.linalg.solve(A, Xd.T @ (yf.double() - ym))
    p = Xh.double() @ w + ym
    return float(1 - ((yh.double() - p) ** 2).sum() / ((yh.double() - yh.double().mean()) ** 2).sum())
def linfit(Xf, yf, Xh, yh):
    torch.manual_seed(0)
    lin = torch.nn.Linear(Xf.shape[1], 65).to(dev)
    opt = torch.optim.Adam(lin.parameters(), lr=LR)
    for _ in range(STEPS):
        opt.zero_grad(); torch.nn.functional.cross_entropy(lin(Xf), yf).backward(); opt.step()
    with torch.no_grad():
        return float((lin(Xh).argmax(-1) == yh).float().mean())
res = {}
for L in range(NL):
    S = torch.stack(store[L])  # (nb,B,T,384) half
    for H in range(NH):
        sl = S[..., H*HS:(H+1)*HS]
        fl, hl = split(sl)
        mean = fl.float().reshape(-1, HS).mean(0); std = fl.float().reshape(-1, HS).std(0) + 1e-6
        r = {}
        for name in tg:
            Xf = ((fl[fitm[name]].float() - mean) / std).to(dev)
            Xh = ((hl[heldm[name]].float() - mean) / std).to(dev)
            yf, yh = Y[name]
            if name == 'position':
                sc = ridge(Xf, yf, Xh, yh); sh = ridge(Xf, yf[perm[name]], Xh, yh)
                r[name] = {'score': sc, 'shuffled': sh, 'above_chance': bool(sc > sh + 0.05)}
            else:
                sc = linfit(Xf, yf, Xh, yh); sh = linfit(Xf, yf[perm[name]], Xh, yh)
                maj = float((yh == torch.bincount(yh).argmax()).float().mean())
                r[name] = {'score': sc, 'shuffled': sh, 'majority': maj, 'above_chance': bool(sc > sh + 0.05)}
        res[f'L{L}H{H}'] = r
    print('layer', L, 'done', round(time.time() - t0, 1), flush=True)
out = {'ckpt': a.ckpt, 'batches': a.batches, 'gpu': torch.cuda.get_device_name(0), 'torch_version': torch.__version__,
       'iter_num': ck['iter_num'], 'wall_seconds': time.time() - t0, 'heads': res}
json.dump(out, open(a.out, 'w'))
print('wrote', a.out, len(res), 'heads', round(time.time() - t0, 1), 's')
