import sys,os,json,time,datetime,numpy as np
from scipy.optimize import linprog
D=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,D)
from verifier import evaluate
EPS=1e-7
def viol(keys,f,X,chunk=2000):
    out=np.empty(len(X))
    for i in range(0,len(X),chunk):
        x=X[i:i+chunk]
        out[i:i+chunk]=np.floor(x[:,None]/keys[None,:])@f
    return out
def solve(name,keys,ub=1-EPS,maxr=40,tlimit=1e9):
    keys=np.array(sorted(keys),dtype=np.float64); K=int(keys.max()); N=10*K
    c=np.log(keys)/keys; Aeq=(1/keys)[None,:]
    allx=np.arange(1,N+1)
    rows=np.unique(np.concatenate([np.arange(1,K+1),np.arange(K,N+1,10)]))
    t0=time.time(); f=None; r0=1; prior=0.0
    ck=os.path.join(D,f'ckpt_{name}.json')
    print('started',name,datetime.datetime.now().isoformat(),flush=True)
    if os.path.exists(ck):
        c0=json.load(open(ck)); rows=np.array(c0['rows'],dtype=np.int64); r0=c0['round']+1; prior=c0['seconds']; f=np.array(c0['f'])
        print('resuming',name,'from round',r0,'with',len(rows),'rows (checkpoint of round',c0['round'],')',flush=True)
    t0-=prior
    for r in range(r0,maxr+1):
        A=np.floor(rows[:,None]/keys[None,:])
        res=linprog(c,A_ub=A,b_ub=np.full(len(rows),ub),A_eq=Aeq,b_eq=[0],bounds=(-10,10),method='highs')
        if res.status!=0: print(name,'status',res.status,res.message,flush=True); return None
        f=res.x; v=viol(keys,f,allx.astype(float))
        mx=float(v.max()-ub); new=allx[v>ub+1e-9]
        new=np.setdiff1d(new,rows)
        rec=dict(name=name,round=r,rows=len(rows),score=float(-res.fun),maxviol=mx,seconds=round(time.time()-t0,1))
        open(os.path.join(D,'rounds.jsonl'),'a').write(json.dumps(rec)+'\n'); print(rec,flush=True)
        if len(new)==0: break
        # add the worst few thousand to keep size sane
        if len(new)>3000: new=new[np.argsort(-v[new-1])[:3000]]
        rows=np.union1d(rows,new)
        tmp=ck+'.tmp'
        json.dump(dict(name=name,round=r,rows=[int(x) for x in rows],f=[float(x) for x in f],seconds=round(time.time()-t0,1),time=datetime.datetime.now().isoformat()),open(tmp,'w')); os.replace(tmp,ck)
        if time.time()-t0>tlimit: break
    sec=time.time()-t0
    cand={"partial_function":{str(int(k)):float(x) for k,x in zip(keys,f)}}
    json.dump(cand,open(os.path.join(D,f'cand_{name}.json'),'w'))
    t1=time.time(); vs=evaluate(cand)
    out=dict(name=name,keys=len(keys),largest=K,program=float(-res.fun),verifier=vs,rounds=r,rows=len(rows),seconds=round(sec,1),verifier_seconds=round(time.time()-t1,1))
    open(os.path.join(D,'results.jsonl'),'a').write(json.dumps(out)+'\n'); print(out,flush=True)
    return out
if __name__=='__main__':
    which=sys.argv[1]
    rec=[]
    if which=='h2':
        rec=[int(k) for k in json.load(open(os.path.join(D,'togetherai_2026.json')))['partial_function']]
    def sqf(n):
        return [k for k in range(2,n+1) if all(k%(p*p) for p in range(2,int(k**.5)+1))]
    ks={'h2':sorted(set(rec)|{1}),'sqf2999':sqf(2999)+[1],'all2999':list(range(1,3000)),'all1499':list(range(1,1500)),'all4999':list(range(1,5000))}[which]
    ub=1.0 if len(sys.argv)>2 and sys.argv[2]=='exact' else 1-EPS
    solve(which+('_exact' if ub==1.0 else ''),ks,ub)
