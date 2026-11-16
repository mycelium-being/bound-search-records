import os, pickle, numpy as np
from collections import Counter
base = r'D:\mycelium\vta\nanoGPT_retention\data'
meta = pickle.load(open(os.path.join(base, 'shakespeare_char', 'meta.pkl'), 'rb'))
stoi = meta['stoi']
text = open(r'D:\mycelium\vta\pg_all.txt', 'r', encoding='utf-8', newline='').read()
dropped = Counter(c for c in text if c not in stoi)
kept = [c for c in text if c in stoi]
print('total chars', len(text), 'kept', len(kept), 'dropped', sum(dropped.values()), 'distinct dropped', len(dropped))
print('distinct dropped chars', ''.join(sorted(dropped)).encode('unicode_escape').decode())
ids = np.array([stoi[c] for c in kept], dtype=np.uint16)
n = int(len(ids) * 0.9)
out = os.path.join(base, 'pg_char'); os.makedirs(out, exist_ok=True)
ids[:n].tofile(os.path.join(out, 'train.bin')); ids[n:].tofile(os.path.join(out, 'val.bin'))
pickle.dump(meta, open(os.path.join(out, 'meta.pkl'), 'wb'))
print('train', n, 'val', len(ids) - n)
