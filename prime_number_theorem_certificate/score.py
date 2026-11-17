import json,os,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from verifier import evaluate
d=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"best_verified.json")))
s=evaluate({"partial_function":d["partial_function"]}); print(s); assert abs(s-d["verifier_score"])<1e-9
