import sys, json, numpy as np, torch
sys.path.insert(0, r'D:\mycelium\vta\nanoGPT')
from model import GPTConfig, GPT
res={}
d=np.fromfile(r'D:\mycelium\vta\nanoGPT\data\form2_char\val.bin',dtype=np.uint16).astype(np.int64)
mk=np.fromfile(r'D:\mycelium\vta\nanoGPT\data\form2_char\val_mask.bin',dtype=np.uint8)
nb=len(d)//256
D=torch.from_numpy(d[:nb*256]).view(nb,256).cuda(); M=torch.from_numpy(mk[:nb*256]).view(nb,256).cuda()
for s in (1337,42,7):
    ck=torch.load(rf'D:\mycelium\vta\nanoGPT\out-form2-s{s}\ckpt.pt',map_location='cuda')
    m=GPT(GPTConfig(**ck['model_args'])); sd={k.replace('_orig_mod.',''):v for k,v in ck['model'].items()}
    m.load_state_dict(sd); m.cuda().eval()
    tot=[0.0,0.0]; cnt=[0,0]
    with torch.no_grad():
        for i in range(0,nb,50):
            x=D[i:i+50]; mm=M[i:i+50,1:]   # mask of the target position
            b,t=x[:,:-1].shape
            pos=torch.arange(t,device='cuda')
            h=m.transformer.drop(m.transformer.wte(x[:,:-1])+m.transformer.wpe(pos))
            for blk in m.transformer.h: h=blk(h)
            lg=m.lm_head(m.transformer.ln_f(h))
            l=torch.nn.functional.cross_entropy(lg.reshape(-1,lg.size(-1)),x[:,1:].reshape(-1),reduction='none').view(b,t)
            tot[1]+=l[mm==1].sum().item(); cnt[1]+=int((mm==1).sum()); tot[0]+=l[mm==0].sum().item(); cnt[0]+=int((mm==0).sum())
    res[s]={'iter':int(ck['iter_num']),'unmasked_loss':tot[0]/cnt[0],'masked_loss':tot[1]/cnt[1],'n_unmasked':cnt[0],'n_masked':cnt[1]}
    print(s,res[s])
json.dump(res,open(r'D:\mycelium\vta\form2_loss.json','w'))
