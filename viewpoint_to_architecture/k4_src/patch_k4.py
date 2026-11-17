s=open('kf_train.py').read()
def rep(a,b):
    global s
    assert s.count(a)==1,a
    s=s.replace(a,b)
rep("form_ckpt = '' #","form_heads = '0' # K4: 'all' = query and key thirds of every layer's c_attn are copied from form_ckpt (or kept at random init if none) and frozen; '0' = row-10 behaviour\nform_ckpt = '' #")
rep("""    print(f"form_rows loaded layers={len(model.transformer.h)} maxabs_diff_vs_ckpt {_mx}")
""","""    if form_heads != 'all':
        print(f"form_rows loaded layers={len(model.transformer.h)} maxabs_diff_vs_ckpt {_mx}")

# K4: all-heads freeze. Full query third and key third of every layer's c_attn weight (no bias in this config) are
# snapshotted (from form_ckpt if given, else the model's current rows) and held by the hook (frozen_idx) and by restore after each step.
if form_heads == 'all':
    assert not bias
    _mx = 0.0
    if form_ckpt:
        _fsd = torch.load(form_ckpt, map_location='cpu')['model']
        _fsd = {(k[len('_orig_mod.'):] if k.startswith('_orig_mod.') else k): v for k, v in _fsd.items()}
    with torch.no_grad():
        for _li, _blk in enumerate(model.transformer.h):
            _at = _blk.attn
            if form_ckpt:
                _src = _fsd[f'transformer.h.{_li}.attn.c_attn.weight'].to(_at.c_attn.weight.device)
                _at.c_attn.weight[0:n_embd].copy_(_src[0:n_embd])
                _at.c_attn.weight[n_embd:2*n_embd].copy_(_src[n_embd:2*n_embd])
                _mx = max(_mx, (_at.c_attn.weight[0:n_embd] - _src[0:n_embd]).abs().max().item(), (_at.c_attn.weight[n_embd:2*n_embd] - _src[n_embd:2*n_embd]).abs().max().item())
            _at.qf = _at.c_attn.weight[0:n_embd].detach().clone()
            _at.kf = _at.c_attn.weight[n_embd:2*n_embd].detach().clone()
            _at.q0_init.copy_(_at.qf[0:n_embd // n_head]); _at.k0_init.copy_(_at.kf[0:n_embd // n_head])
            _at.frozen_idx = torch.arange(0, 2 * n_embd, device=_at.c_attn.weight.device)
    if form_ckpt:
        print(f"form_rows loaded layers={len(model.transformer.h)} heads={n_head} maxabs_diff_vs_ckpt {_mx}")
    else:
        print(f"form_rows frozen_at_random_init layers={len(model.transformer.h)} heads={n_head} (no form_ckpt)")
""")
rep("""            _at.c_attn.weight[0:_at.q0_init.shape[0]].copy_(_at.q0_init)
            _at.c_attn.weight[_at.n_embd:_at.n_embd + _at.k0_init.shape[0]].copy_(_at.k0_init)
""","""            if form_heads == 'all':
                _at.c_attn.weight[0:_at.n_embd].copy_(_at.qf)
                _at.c_attn.weight[_at.n_embd:2*_at.n_embd].copy_(_at.kf)
            else:
                _at.c_attn.weight[0:_at.q0_init.shape[0]].copy_(_at.q0_init)
                _at.c_attn.weight[_at.n_embd:_at.n_embd + _at.k0_init.shape[0]].copy_(_at.k0_init)
""")
rep("""        _at=_blk.attn; hs=_at.q0_init.shape[0]
        for _w,_i in ((_at.c_attn.weight[0:hs],_at.q0_init),(_at.c_attn.weight[_at.n_embd:_at.n_embd+hs],_at.k0_init)):""","""        _at=_blk.attn; hs=_at.q0_init.shape[0]
        _pairs = ((_at.c_attn.weight[0:hs],_at.q0_init),(_at.c_attn.weight[_at.n_embd:_at.n_embd+hs],_at.k0_init))
        if form_heads == 'all':
            _pairs = ((_at.c_attn.weight[0:_at.n_embd],_at.qf),(_at.c_attn.weight[_at.n_embd:2*_at.n_embd],_at.kf))
        for _w,_i in _pairs:""")
open('k4_train.py','w').write(s)
