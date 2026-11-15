import sys, os
import numpy as np
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from verifier import verify_and_compute_c3
for name in ("best.npy", "best_n400.npy"):
    f = np.load(os.path.join(here, name))
    print(name, "N =", len(f), "C3 =", "%.16f" % verify_and_compute_c3(f))
