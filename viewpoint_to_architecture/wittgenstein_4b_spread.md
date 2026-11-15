Equality lines (check_init_4b.py, before training):
seed 1337 max_abs_diff 0.000e+00 trainable 10745088
seed 42 max_abs_diff 0.000e+00 trainable 10745088
seed 7 max_abs_diff 0.000e+00 trainable 10745088

U spread at the checkpoint (u_spread.py, path pointed at out-witt-useinit-s<seed>; checkpoint is the best-val one):
seed 1337 iter 4000 best_val 3.3443: rows (65, 384), mean row norm 17.597, deviation-from-mean-row norm (mean over rows) 0.1094, ratio dev/norm 0.0062, pairwise cosine min 0.9989 mean 0.9999, zero rows 0
seed 42 iter 1000 best_val 3.3452: rows (65, 384), mean row norm 18.316, deviation-from-mean-row norm (mean over rows) 0.2690, ratio dev/norm 0.0147, pairwise cosine min 0.9630 mean 0.9986, zero rows 0
seed 7 iter 4250 best_val 3.3433: rows (65, 384), mean row norm 17.630, deviation-from-mean-row norm (mean over rows) 0.1249, ratio dev/norm 0.0071, pairwise cosine min 0.9971 mean 0.9999, zero rows 0
