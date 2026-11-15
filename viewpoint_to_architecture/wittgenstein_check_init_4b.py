import sys, torch
sys.path.insert(0, r"D:\mycelium\vta\nanoGPT_wittgenstein")
sys.argv = ['x']
from model import GPT, GPTConfig
# train.py: torch.manual_seed(seed + seed_offset), seed_offset = 0 (single process); model built from scratch afterwards
for s in (1337, 42, 7):
    torch.manual_seed(s)
    m = GPT(GPTConfig(block_size=256, vocab_size=65, n_layer=6, n_head=6, n_embd=384, dropout=0.2, bias=False, embed_mode='use_init'))
    ck = torch.load(rf"D:\mycelium\vta\nanoGPT_wittgenstein\out-witt-frozen-s{s}\ckpt.pt", map_location="cpu", weights_only=False)
    sd = ck["model"]; k = [x for x in sd if x.endswith("transformer.wte.weight")][0]
    d = (0.02 * m.U - sd[k]).abs().max().item()
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"seed {s} max_abs_diff {d:.3e} trainable {tr}")
