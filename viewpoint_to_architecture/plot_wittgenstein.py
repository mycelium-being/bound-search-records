import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
V="./"
base=json.load(open(V+"baseline.json")); s7=json.load(open(V+"baseline_s7.json"))
use=json.load(open(V+"wittgenstein_use.json")); fro=json.load(open(V+"wittgenstein_frozen.json"))
def curves(d):
    c=d["curves"]
    if isinstance(c,dict): return {k:[(r[0],r[2]) for r in v] for k,v in c.items()}
    return {"0":[(r[0],r[2]) for r in c]}
B={**curves(base),**{"7":curves(s7)[list(curves(s7))[0]]}}; U=curves(use); F=curves(fro)
bm,um,fm=1.4622,3.3448,1.4703; band=0.0137
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(16,11),dpi=110,sharex=True,gridspec_kw=dict(height_ratios=[1,2.2],hspace=0.06))
cols={"base":"#1f77b4","use":"#d62728","fro":"#2ca02c"}
for ax in (ax1,ax2):
    for k,v in B.items(): ax.plot(*zip(*v),color=cols["base"],lw=1.6,alpha=.85)
    for k,v in F.items(): ax.plot(*zip(*v),color=cols["fro"],lw=1.6,alpha=.85)
    for k,v in U.items(): ax.plot(*zip(*v),color=cols["use"],lw=1.6,alpha=.85)
ax1.set_ylim(3.25,3.60); ax2.set_ylim(1.40,1.80)
ax2.axhspan(bm-band,bm+band,color=cols["base"],alpha=.15,lw=0)
ax1.spines["bottom"].set_visible(False); ax2.spines["top"].set_visible(False); ax1.tick_params(bottom=False)
d=.008
for ax,y in ((ax1,0),(ax2,1)):
    ax.plot((-d,+d),(y-d*2,y+d*2),transform=ax.transAxes,color="k",clip_on=False,lw=1)
ax1.text(0.99,0.9,"the use arm never leaves the unigram plateau",transform=ax1.transAxes,ha="right",va="top",fontsize=17,color=cols["use"])
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
h=[Line2D([],[],color=cols["base"],lw=3,label=f"baseline, stock nanoGPT: mean best val {bm:.4f} (3 seeds)"),
   Line2D([],[],color=cols["use"],lw=3,label=f"use: no input table, running record of use: {um:.4f} (3 seeds)"),
   Line2D([],[],color=cols["fro"],lw=3,label=f"control: input table frozen at random: {fm:.4f} (3 seeds)"),
   Patch(color=cols["base"],alpha=.15,label=f"tie band ±{band} around the baseline mean, set before the arms ran")]
ax2.legend(handles=h,loc="upper center",fontsize=16,frameon=False)
ax2.set_xlabel("iteration",fontsize=18); fig.text(0.04,0.5,"val loss",va="center",rotation="vertical",fontsize=18)
for ax in (ax1,ax2): ax.tick_params(labelsize=15)
fig.suptitle("Wittgenstein, PI §43: meaning as use  —  row 4, 2026-11-15, 3 seeds per arm",fontsize=19,y=0.95)
out=V+"wittgenstein_val_loss.png"; fig.savefig(out,bbox_inches="tight"); print(out)
