# Exact check, mpmath 50 digits, margin 0. usage: python3 check_exact.py best_wN.json [out.json]
import sys, json, itertools
from mpmath import mp, mpf, cos, sin, pi, sqrt
mp.dps = 50
d = json.load(open(sys.argv[1]))
z = [mpf(s) for s in d['z']]
N = 12
H = [(z[3*i], z[3*i+1], z[3*i+2]) for i in range(N)]
S3 = sqrt(3) / 2

def verts(c, t):
    return [(c[0] + cos(t + pi*k/3), c[1] + sin(t + pi*k/3)) for k in range(6)]

def worst_h(Hs, L):
    global H
    old = H; H = Hs
    try: return worst(L)
    finally: H = old

def worst(L):
    m = mpf(10)
    # containment: outer vertices at angle pi*k/3, side L; edge normals at pi/6+k*pi/3, apothem S3*L
    for (x, y, t) in H:
        for (vx, vy) in verts((x, y), t):
            for k in range(6):
                a = pi/6 + k*pi/3
                m = min(m, S3*L - (cos(a)*vx + sin(a)*vy))
    # separation: best SAT axis over edge normals of both
    for i, j in itertools.combinations(range(N), 2):
        vi, vj = verts(H[i][:2], H[i][2]), verts(H[j][:2], H[j][2])
        best = mpf(-10)
        for owner in (i, j):
            for k in range(6):
                a = H[owner][2] + pi/6 + k*pi/3
                n = (cos(a), sin(a))
                pi_ = [n[0]*v[0] + n[1]*v[1] for v in vi]
                pj_ = [n[0]*v[0] + n[1]*v[1] for v in vj]
                best = max(best, max(min(pj_) - max(pi_), min(pi_) - max(pj_)))
        m = min(m, best)
    return m

L = z[3*N]
w = worst(L)
infl = mpf(0); H0 = list(H); L0 = L
TARGET = mpf('2e-9')  # their verifier calls a pair overlapping unless separated by >1e-9; margin 0 is our own bar
if w < TARGET:
    # scale centres and L together by (L0+infl)/L0 (only grows gaps); smallest 1e-10 multiple that passes, by bisection
    def at(k):
        infl = k * mpf('1e-10'); s = (L0 + infl) / L0
        return [(h[0]*s, h[1]*s, h[2]) for h in H0], L0 + infl
    lo, hi = 0, 1
    while True:
        Hh, Lh = at(hi)
        if worst_h(Hh, Lh) >= TARGET: break
        lo, hi = hi, hi * 2
        if hi > 10**7: break
    while hi - lo > 1:
        mid = (lo + hi) // 2
        Hm, Lm = at(mid)
        if worst_h(Hm, Lm) >= TARGET: hi = mid
        else: lo = mid
    H, L = at(hi); H = list(H); infl = hi * mpf('1e-10'); w = worst_h(H, L)
print('L =', mp.nstr(L, 12), 'worst margin =', mp.nstr(w, 5), 'inflation =', mp.nstr(infl, 3), 'PASS' if w >= 0 else 'FAIL')
if len(sys.argv) > 2:
    json.dump({'hexagons': [[mp.nstr(h[0], 25), mp.nstr(h[1], 25), mp.nstr(h[2]*180/pi, 25)] for h in H],
               'outer_side_length': mp.nstr(L, 25), 'outer_center': [0, 0], 'outer_angle_deg': 0}, open(sys.argv[2], 'w'))
