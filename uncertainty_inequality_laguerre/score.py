import sys,json,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import verifier as v
b=json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'best_verified.json')))
e=v.get_upper_bound_laguerre(b['roots'])
print(e); assert abs(e-b['exact_bound'])<1e-9
