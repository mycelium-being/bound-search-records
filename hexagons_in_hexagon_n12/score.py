import sys, json, os
here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, here)
from verifier import evaluate
print("outer side (verifier score, penalty 0 means feasible):", repr(evaluate(json.load(open(os.path.join(here, "hex12_best.json"))))))
