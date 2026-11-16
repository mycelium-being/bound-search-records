import sys, json, numpy as np, torch
sys.path.insert(0, r'D:\mycelium\vta\nanoGPT')
from model import GPTConfig, GPT
res={}
for s in (1337,42,7):
    ck=torch.load(rf'D:\mycelium\vta\nanoGPT\out-form-s{s}\ckpt.pt',map_location='cuda')
    m=GPT(GPTConfig(**ck['model_args'])); sd=ck['model']
    m.load_state_dict(sd); m.cuda().eval()
    d=np.fromfile(r'D:\mycelium\vta\nanoGPT\data\form_char\val.bin',dtype=np.uint16).astype(np.int64)
    nb=len(d)//256; d=torch.from_numpy(d[:nb*256]).view(nb,256).cuda()
    tot=[0.0,0.0]
    with torch.no_grad():
        for i in range(0,nb,50):
            x=d[i:i+50]
            lg,_=m(x[:,:-1],x[:,1:]) if False else (None,None)
            # model returns last-position logits only when targets None; use targets to get full logits via manual forward
            b,t=x[:,:-1].shape
            pos=torch.arange(t,device='cuda')
            h=m.transformer.drop(m.transformer.wte(x[:,:-1])+m.transformer.wpe(pos))
            for blk in m.transformer.h: h=blk(h)
            lg=m.lm_head(m.transformer.ln_f(h))
            l=torch.nn.functional.cross_entropy(lg.reshape(-1,lg.size(-1)),x[:,1:].reshape(-1),reduction='none').view(b,t)
            tot[0]+=l[:,:127].sum().item(); tot[1]+=l[:,128:].sum().item()
    n=nb
    res[s]={'iter':int(ck['iter_num']),'first_half_loss':tot[0]/(n*127),'second_half_loss':tot[1]/(n*127)}
    print(s,res[s])
json.dump(res,open(r'D:\mycelium\vta\form_halfloss.json','w'))
