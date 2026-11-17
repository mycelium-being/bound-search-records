"""Runs the unmodified arena verifier on best_verified.json and prints the score to 16 digits.
Exits nonzero if the score differs from the recorded one by more than 1e-12."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verifier import evaluate
d = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'best_verified.json')))
s = evaluate(d)
print('flat polynomial degree 69, verifier score', repr(s), 'recorded', repr(d['verifier_score']))
assert abs(s - d['verifier_score']) < 1e-12, 'score drifted from the recorded value'
