import numpy as np, os, shutil
d=r'D:\mycelium\vta\nanoGPT\data\form_char'
os.makedirs(d,exist_ok=True)
def mk(seed,n):
    r=np.random.RandomState(seed) if False else np.random.default_rng(seed)
    out=[]
    for _ in range(n):
        b=r.integers(0,65,size=128)
        out.append(np.concatenate([b,b]))
    return np.concatenate(out).astype(np.uint16)
tr=mk(0,8000); va=mk(1,400)
tr.tofile(os.path.join(d,'train.bin')); va.tofile(os.path.join(d,'val.bin'))
shutil.copy(r'D:\mycelium\vta\nanoGPT\data\shakespeare_char\meta.pkl',os.path.join(d,'meta.pkl'))
print(len(tr),len(va))
print('block0',tr[:10].tolist()); print('repeat',tr[128:138].tolist())
