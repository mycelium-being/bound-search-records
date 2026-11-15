import json, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
V="./"
base=json.load(open(V+"baseline.json")); s7=json.load(open(V+"baseline_s7.json"))
use=json.load(open(V+"wittgenstein_use.json")); fro=json.load(open(V+"wittgenstein_frozen.json")); ui=json.load(open(V+"wittgenstein_use_init.json")); us=json.load(open(V+"wittgenstein_use_slow.json"))
def curves(d):
    c=d["curves"]
    if isinstance(c,dict): return {k:[(r[0],r[2]) for r in v] for k,v in c.items()}
    return {"0":[(r[0],r[2]) for r in c]}
B={**curves(base),**{"7":curves(s7)[list(curves(s7))[0]]}}; U=curves(use); F=curves(fro); UI=curves(ui); US=curves(us)
bm,um,fm,uim=1.4622,3.3448,1.4703,3.3443; usm=1.4688; band=0.0137
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(16,11),dpi=110,sharex=True,gridspec_kw=dict(height_ratios=[1,2.2],hspace=0.06))
cols={"base":"#1f77b4","use":"#d62728","fro":"#2ca02c","ui":"#9467bd","us":"#ff7f0e"}
for ax in (ax1,ax2):
    for k,v in B.items(): ax.plot(*zip(*v),color=cols["base"],lw=1.6,alpha=.85)
    for k,v in F.items(): ax.plot(*zip(*v),color=cols["fro"],lw=1.6,alpha=.85)
    for k,v in U.items(): ax.plot(*zip(*v),color=cols["use"],lw=1.6,alpha=.85)
    for k,v in UI.items(): ax.plot(*zip(*v),color=cols["ui"],lw=1.6,alpha=.85,ls="--")
    for k,v in US.items(): ax.plot(*zip(*v),color=cols["us"],lw=1.6,alpha=.85)
ax1.set_ylim(3.25,3.60); ax2.set_ylim(1.40,1.80)
ax2.axhspan(bm-band,bm+band,color=cols["base"],alpha=.15,lw=0)
ax1.spines["bottom"].set_visible(False); ax2.spines["top"].set_visible(False); ax1.tick_params(bottom=False)
d=.008
for ax,y in ((ax1,0),(ax2,1)):
    ax.plot((-d,+d),(y-d*2,y+d*2),transform=ax.transAxes,color="k",clip_on=False,lw=1)
ax1.text(0.99,0.9,"fast-forgetting use arms sit on the unigram plateau: zero start (row 4) and random start (row 4b) agree to 0.001;\nslowed to 0.999 (row 4c) the same record drops into the band below",transform=ax1.transAxes,ha="right",va="top",fontsize=15,color=cols["use"])
ax2.text(0.99,0.03,"row 4c, use-slow: 1.4688, inside the band, 0.0015 from the frozen control",transform=ax2.transAxes,ha="right",va="bottom",fontsize=15,color=cols["us"])
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
h=[Line2D([],[],color=cols["base"],lw=3,label=f"baseline, stock nanoGPT: mean best val {bm:.4f} (3 seeds)"),
   Line2D([],[],color=cols["use"],lw=3,label=f"use: no input table, running record of use: {um:.4f} (3 seeds)"),
   Line2D([],[],color=cols["ui"],lw=3,ls="--",label=f"use-init, row 4b: record of use from the control's random table: {uim:.4f} (3 seeds)"),
   Line2D([],[],color=cols["us"],lw=3,label=f"use-slow, row 4c: the same record, forgetting 0.999 per step: {usm:.4f} (3 seeds)"),
   Line2D([],[],color=cols["fro"],lw=3,label=f"control: input table frozen at random: {fm:.4f} (3 seeds)"),
   Patch(color=cols["base"],alpha=.15,label=f"tie band ±{band} around the baseline mean, set before the arms ran")]
ax2.legend(handles=h,loc="upper center",fontsize=16,frameon=False)
ax2.set_xlabel("iteration",fontsize=18); fig.text(0.04,0.5,"val loss",va="center",rotation="vertical",fontsize=18)
for ax in (ax1,ax2): ax.tick_params(labelsize=15)
fig.suptitle("Wittgenstein, PI §43: meaning as use  —  rows 4, 4b and 4c, 2026-11-15, 3 seeds per arm",fontsize=19,y=0.95)
out=V+"wittgenstein_val_loss.png"; fig.savefig(out,bbox_inches="tight"); print(out)
