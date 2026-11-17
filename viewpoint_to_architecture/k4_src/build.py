import json, re, os, sys, subprocess
V='/Users/erenkahveci/mycelium/.claude/mycelium/being/bound_search/vta/'
F='/tmp/vta_k4/fetch/'
os.makedirs(F,exist_ok=True)
SEEDS=(1337,42,7)
def curve(log):
    return [[int(a),float(b),float(c)] for a,b,c in re.findall(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)",open(log).read())]
ref=json.load(open(V+'kantform_10.json')); refr=json.load(open(V+'retention_kantform_10.json'))
f2=json.load(open(F+'form2_loss.json'))
def evalp(tag):
    t=open(F+f'eval_{tag}.txt').read()
    infos=re.findall(r"info \S+ iter_num (\d+) best_val_loss ([\d.]+)",t)
    ev=re.findall(r"eval (\S+) (\S+) ([\d.]+)",t)
    return infos,ev
def train_json(tag,rowname,form):
    out={'config':ref['config'],'gpu':ref['gpu'],'torch_version':ref['torch_version'],'params':{'trainable':10745088,'frozen':0},'seeds':list(SEEDS)}
    if form:
        out['config_note']=("tree nanoGPT_k4 (copy of nanoGPT_kantform; model.py unchanged, train.py gains --form_heads=all, default 0): the whole query third (rows 0:384) and whole key third (rows 384:768) of c_attn in every layer, all six heads, are copied from the phase-F2 checkpoint out-form2-s<seed> (stock tree, 1500 iters on data/form2_char: 8000 blocks of 256 symbols, each block filled with random strings of length 2..64 uniform each followed by its own repeat, numpy default_rng seed 0; val 400 blocks seed 1; masks in train_mask.bin/val_mask.bin) into a fresh seed-init model via --form_ckpt, then frozen by the row-8 gradient hook (frozen_idx widened to arange(0,768)) plus restore after every optimizer step. Load print 'form_rows loaded layers=6 heads=6 maxabs_diff_vs_ckpt 0.0'; end print 'frozen_rows equal True maxabs 0.0'; trainable parameter count 10745088 (stock; requires_grad count, frozen rows are held by hook and restore as in rows 8 and 10). Value third, c_proj, MLP, embeddings, LN trainable.")
    else:
        out['config_note']=("tree nanoGPT_k4, same as row 11 with no --form_ckpt: the same query and key thirds of every layer, all six heads, are left at the fresh seed-init random values and frozen by the same hook and restore. Print 'form_rows frozen_at_random_init layers=6 heads=6 (no form_ckpt)'; 'frozen_rows equal True maxabs 0.0' at the end. Trainable count 10745088.")
    cur={};fv={};bv={}
    pre='k4' if form else 'k4rand'
    for s in SEEDS:
        c=curve(f'{F}{pre}_s{s}.log'); cur[str(s)]=c
        fv[str(s)]=c[-1][2]; b=min(c,key=lambda r:r[2]); bv[str(s)]=[b[0],b[2]]
    out['curves']=cur; out['final_val']=fv; out['best_val_iter_and_loss']=bv
    out['mean_best_val']=sum(v[1] for v in bv.values())/3
    out['wall_seconds']={str(s):None for s in SEEDS}; out['wall_note']='per-run wall not recorded; runs shared the card with probes and phase-B jobs'
    if form: out['phase_F2']=f2
    return out
def ret_json(tag,form):
    pre='k4' if form else 'k4rand'
    infos,ev=evalp(pre)
    out={'gpu':refr['gpu'],'torch_version':refr['torch_version'],'eval_script':refr['eval_script'],'second_text':refr['second_text'],'config':refr['config'],'params':{'trainable':10745088,'frozen':0},'seeds':list(SEEDS)}
    pa={};pb={};rise={}
    for i,s in enumerate(SEEDS):
        it,bv=infos[2*i]; before=float(ev[5*i+1][2]); after=float(ev[5*i+3][2]); ess=float(ev[5*i+4][2])
        if (not form) and s==7:
            rt=open(F+'eval_k4rand_s7_redo.txt').read(); rv=re.findall(r"eval (\S+) (\S+) ([\d.]+)",rt)
            after=float(rv[0][2]); ess=float(rv[1][2])
        lg=curve(f'{F}{pre}_ret_s{s}.log')
        txt=open(f'{F}{pre}_ret_s{s}.log').read()
        mi=int(re.search(r"max_iters = (\d+)",txt.split('Overriding: max_iters = ')[0]+'max_iters = 0') .group(1)) if False else int(re.search(r"Overriding: max_iters = (\d+)",txt).group(1))
        pa[str(s)]={'iter_num':int(it),'best_val_loss':float(bv),'shakespeare_val_before':before}
        pb[str(s)]={'max_iters':mi,'learning_rate':1e-4,'curves':lg,'pg_val_after':ess,'shakespeare_val_after':after,'wall_seconds':None,'frozen_rows_after':re.search(r"frozen_rows equal (\w+) maxabs ([\d.e+-]+)",txt).group(0)}
        rise[str(s)]=round(after-before,4)
    out['config_note']=(f"row {'11' if form else '11c'} checkpoints (out-{pre.replace('k4','k4')}-s<seed>, best-val iters {[pa[str(s)]['iter_num'] for s in SEEDS]}) resumed on pg_char for 2000+ iterations (max_iters=iter_num+2000) at constant 1e-4; optimizer state restored; query and key thirds of every layer held frozen by --form_heads=all"+(" (--form_ckpt also passed)" if form else "")+"; eval_ckpt.py path-patched to nanoGPT_k4; wall_seconds not recorded (jobs shared the card).")
    out['phase_a']=pa;out['phase_b']=pb;out['rise']=rise
    out['mean_rise']=sum(rise.values())/3
    out['mean_pg_after']=sum(pb[str(s)]['pg_val_after'] for s in SEEDS)/3
    return out
def probe_merge(name_pat,outname):
    ps={s:json.load(open(F+name_pat.format(s=s))) for s in SEEDS}
    o=json.load(open(V+'probe_kantform_10.json')); o={k:v for k,v in o.items() if k in('config','gpu','torch_version','seeds')}
    o['seeds']=list(SEEDS); o['heads']={}
    for s in SEEDS:
        p=ps[s]; o['heads'][str(s)]=p.get('heads',p)
    keys=list(o['heads'][str(SEEDS[0])].keys())
    o['mean_over_seeds']={}
    for k in keys:
        o['mean_over_seeds'][k]={m:sum(o['heads'][str(s)][k][m]['score'] for s in SEEDS)/3 for m in ('position','current','previous')}
    json.dump(o,open(V+outname,'w'))
if __name__=='__main__':
    what=sys.argv[1]
    if what=='train':
        json.dump(train_json('k4',11,True),open(V+'k4_11.json','w')); json.dump(train_json('k4rand',11,False),open(V+'k4rand_11c.json','w'))
    if what=='ret_k4': json.dump(ret_json('k4',True),open(V+'retention_k4_11.json','w'))
    if what=='ret_k4rand': json.dump(ret_json('k4rand',False),open(V+'retention_k4rand_11c.json','w'))
    if what=='probe':
        probe_merge('probe_form2_s{s}.json','probe_form2_F.json'); probe_merge('probe_k4_s{s}.json','probe_k4_11.json'); probe_merge('probe_k4rand_s{s}.json','probe_k4rand_11c.json')
