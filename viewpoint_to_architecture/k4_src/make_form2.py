import numpy as np, os, shutil
d=r'D:\mycelium\vta\nanoGPT\data\form2_char'
os.makedirs(d,exist_ok=True)
def mk(seed,nb):
    r=np.random.default_rng(seed)
    X=[];M=[]
    for _ in range(nb):
        x=[];m=[]
        while len(x)<256:
            n=int(r.integers(2,65))
            s=r.integers(0,65,size=n).tolist()
            x+=s+s; m+=[0]*n+[1]*n
        X+=x[:256]; M+=m[:256]
    return np.array(X,dtype=np.uint16),np.array(M,dtype=np.uint8)
for name,seed,nb in (('train',0,8000),('val',1,400)):
    x,m=mk(seed,nb)
    x.tofile(os.path.join(d,name+'.bin')); m.tofile(os.path.join(d,name+'_mask.bin'))
    if name=='train':
        print('block0 ids',x[:40].tolist()); print('block0 mask',m[:40].tolist())
    print(name,len(x),m.mean())
shutil.copy(r'D:\mycelium\vta\nanoGPT\data\shakespeare_char\meta.pkl',os.path.join(d,'meta.pkl'))
