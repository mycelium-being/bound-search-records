import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
D = "/Users/erenkahveci/mycelium/.claude/mycelium/being/bound_search/vta/"
O = D + "page/"
V = [("baseline", "baseline (stock nanoGPT)", "tab:blue"),
     ("kant_prevtoken", "Kant: fixed previous-token head, MLP widened to match", "tab:red"),
     ("kant_control", "Kant control: fixed head, stock MLP", "tab:orange"),
     ("hume_nopos", "Hume: no position embedding", "tab:green")]
data = {k: json.load(open(D + k + ".json")) for k, _, _ in V}
stock = data["kant_prevtoken"]["params"]["stock"]
fig, ax = plt.subplots(figsize=(9, 5), dpi=160)
rows = {}
for k, lab, c in V:
    cur = data[k]["curves"]
    seeds = ["1337", "42"]
    its = [r[0] for r in cur["1337"]]
    vals = {s: [r[2] for r in cur[s]] for s in seeds}
    for s in seeds:
        ax.plot(its, vals[s], color=c, alpha=0.25, lw=0.8)
    mean = [(a + b) / 2 for a, b in zip(vals["1337"], vals["42"])]
    ax.plot(its, mean, color=c, lw=2, label=lab)
    mi = min(range(len(mean)), key=lambda i: mean[i])
    ax.plot([its[mi]], [mean[mi]], "o", color=c, ms=5)
    best = {s: min(zip(vals[s], its)) for s in seeds}
    mb = sum(best[s][0] for s in seeds) / 2
    ax.annotate(f"{mb:.4f}", (its[mi], mean[mi]), textcoords="offset points", xytext=((-34, -26) if k=="kant_prevtoken" else (4, -13)), fontsize=8, color=c)
    p = stock if k == "baseline" else data[k]["params"]["variant"]
    rows[k] = dict(lab=lab, p=p, best=best, mb=mb, fin=(vals["1337"][-1] + vals["42"][-1]) / 2, curvemin=mean[mi])
ax.set_ylim(1.40, 2.00); ax.set_xlim(0, 5000)
ax.set_xlabel("iteration"); ax.set_ylabel("val loss (shakespeare_char, 6 layers, 384 wide, 5000 iters)")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(O + "val_loss.png"); fig.savefig(O + "val_loss.svg")
b = rows["baseline"]["mb"]
L = ["| variant | parameters | best val seed 1337 (iter) | best val seed 42 (iter) | mean best val | delta from baseline mean | final val mean |", "|---|---|---|---|---|---|---|"]
for k, r in rows.items():
    L.append(f"| {r['lab']} | {r['p']:,} | {r['best']['1337'][0]:.4f} ({r['best']['1337'][1]}) | {r['best']['42'][0]:.4f} ({r['best']['42'][1]}) | {r['mb']:.4f} | {r['mb']-b:+.4f} | {r['fin']:.4f} |")
L += ["", "Validation loss on shakespeare_char over 5000 iterations; the y-axis is clipped to 1.40-2.00 so the iteration-0 value (about 4.2) is off the chart; faint lines are the two seeds (1337, 42), solid lines their mean, dots mark each mean curve's minimum; the number beside a dot is the mean of the two seeds' own best values, which sits up to 0.005 below the mean curve's minimum where the seeds peak at different iterations (kant control, hume).",
      "", "Parameter count of the baseline is taken from kant_prevtoken.json's matched stock count (baseline.json has none)."]
open(O + "numbers.md", "w").write("\n".join(L) + "\n")
for k, r in rows.items(): print(k, round(r["mb"], 4), round(r["curvemin"], 4), round(r["fin"], 4))
for k in rows:
    c = data[k]["curves"]; m = lambda i: (c["1337"][i][2] + c["42"][i][2]) / 2
    print(k, "it2000", round(m(8), 4), "it5000", round(m(20), 4))
