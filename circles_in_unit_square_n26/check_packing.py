"""Exact checker: circles in the unit square, maximise sum of radii.
Usage: python3 check_packing.py file.json  (JSON {"n","x","y","r"} as decimal strings)
Zero tolerance, fractions.Fraction arithmetic. Exit 1 on FAIL."""
import sys, json
from fractions import Fraction as F


def main():
    d = json.load(open(sys.argv[1]))
    n = int(d["n"])
    x = [F(s) for s in d["x"]]; y = [F(s) for s in d["y"]]; r = [F(s) for s in d["r"]]
    assert len(x) == len(y) == len(r) == n, "length mismatch"
    worst = None
    ok = True

    def note(m):
        nonlocal worst
        worst = m if worst is None or m < worst else worst

    for i in range(n):
        if r[i] <= 0:
            ok = False
        for m in (x[i] - r[i], 1 - x[i] - r[i], y[i] - r[i], 1 - y[i] - r[i]):
            note(m)
            if m < 0:
                ok = False
    for i in range(n):
        for j in range(i + 1, n):
            m = (x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2 - (r[i] + r[j]) ** 2
            note(m)
            if m < 0:
                ok = False
    s = sum(r)
    print(f"n={n} sum={float(s):.12f} exact_sum_num_den={s.numerator}/{s.denominator} worst_margin={float(worst):.3e}")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


main()
