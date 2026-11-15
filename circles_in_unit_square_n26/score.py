import json, os
from fractions import Fraction as F
here = os.path.dirname(os.path.abspath(__file__))
d = json.load(open(os.path.join(here, "n26_best.json")))
s = sum(F(r) for r in d["r"])
print("sum of radii:", float(s), "exact:", "%d/%d" % (s.numerator, s.denominator))
