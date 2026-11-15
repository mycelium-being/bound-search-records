# Row 4c spread (use_slow, decay 0.999), best-val checkpoints

## Equality lines (before training)
seed 1337 max_abs_diff 0.000e+00 trainable 10745088
seed 42 max_abs_diff 0.000e+00 trainable 10745088
seed 7 max_abs_diff 0.000e+00 trainable 10745088

## u_decay lines from the logs
seed 1337: u_decay 0.999
seed 42: u_decay 0.999
seed 7: u_decay 0.999

## u_spread_4c.py output
seed 1337 iter 2000 best_val 1.4666: rows (65, 384), mean row norm 11.196, deviation-from-mean-row norm (mean over rows) 9.5004, ratio dev/norm 0.8485, pairwise cosine min -0.3706 mean 0.2583, zero rows 0
seed 42 iter 1750 best_val 1.4726: rows (65, 384), mean row norm 10.986, deviation-from-mean-row norm (mean over rows) 9.4543, ratio dev/norm 0.8605, pairwise cosine min -0.3041 mean 0.2405, zero rows 0
seed 7 iter 1750 best_val 1.4672: rows (65, 384), mean row norm 10.958, deviation-from-mean-row norm (mean over rows) 9.4718, ratio dev/norm 0.8644, pairwise cosine min -0.3044 mean 0.2323, zero rows 0
