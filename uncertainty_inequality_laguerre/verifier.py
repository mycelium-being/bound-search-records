import numpy as np
import sympy
import sys
sys.set_int_max_str_digits(0)

def verify_hermite_combination(inputs: sympy.Expr):
    """Check that the Hermite combination vanishes at 0 and is positive at infinity."""
    x = sympy.symbols('x')
    value_at_0 = inputs.subs(x, 0)
    assert value_at_0 == 0, f'The value at 0 is {value_at_0} != 0.'
    assert sympy.limit(inputs, x, sympy.oo) > 0, 'Limit at infty is not positive.'


def find_hermite_combination(coeffs: np.ndarray) -> sympy.Expr:
    """Computes the Hermite combination for given coefficients."""
    m = len(coeffs)
    rational_coeffs = [sympy.Rational(c) for c in coeffs]
    degrees = np.arange(0, 4 * m + 4, 4)
    x = sympy.symbols('x')
    hps = [
        sympy.polys.orthopolys.hermite_poly(n=i, x=x, polys=False)
        for i in degrees
    ]

    partial_fn: sympy.Expr = sum(
        rational_coeffs[i] * hps[i] for i in range(len(rational_coeffs))
    )

    a = hps[-1].subs(x, 0)
    b = -partial_fn.subs(x, 0)
    last_coeff = b / a
    rational_coeffs.append(sympy.Rational(last_coeff))

    res_fn = sum(rational_coeffs[i] * hps[i] for i in range(len(rational_coeffs)))

    if sympy.limit(res_fn, x, sympy.oo) < 0:
        res_fn = -res_fn

    return res_fn


def get_upper_bound(coeffs: np.ndarray) -> float:
    """Computes the upper bound for the given Hermite combination."""
    g_exp = find_hermite_combination(coeffs)
    x = sympy.symbols('x')
    gq_fn = sympy.exquo(g_exp, x**2)
    rroots = sympy.real_roots(gq_fn, x)
    approx_roots = list()
    largest_sign_change = 0

    for root in rroots:
        approx_root = root.eval_rational(n=200)
        approx_root_p = approx_root + sympy.Rational(1e-198)
        approx_root_m = approx_root - sympy.Rational(1e-198)
        approx_roots.append(approx_root)
        is_sign_change = (
            (gq_fn.subs(x, approx_root_p) > 0 and gq_fn.subs(x, approx_root_m) < 0)
            or (
                gq_fn.subs(x, approx_root_p) < 0
                and gq_fn.subs(x, approx_root_m) > 0
            )
        )
        if is_sign_change:
            largest_sign_change = max(largest_sign_change, approx_root)

    return float(largest_sign_change**2) / (2 * np.pi)

def _find_laguerre_combination(zs):
    """Construct Laguerre test function with prescribed double roots."""
    m = len(zs)
    alpha = sympy.Rational(1, 2) - 1
    x = sympy.symbols('x')
    degrees = np.arange(0, 4 * m + 4, 2)
    lps = [
        sympy.polys.orthopolys.laguerre_poly(n=int(i), x=x, alpha=alpha, polys=False)
        for i in degrees
    ]
    num_lps = len(lps)
    num_conditions = 2 * m + 2

    mat = sympy.Matrix(num_conditions, num_lps, lambda i, j: 0)
    b = sympy.Matrix(num_conditions, 1, lambda i, j: 0)
    b[1] = 1

    for j in range(num_lps):
        mat[0, j] = lps[j].subs(x, 0)
        mat[1, j] = lps[j].diff(x).subs(x, 0)

    for i in range(m):
        zi = sympy.Rational(zs[i])
        for j in range(num_lps):
            mat[2 * i + 2, j] = lps[j].subs(x, zi)
            mat[2 * i + 3, j] = lps[j].diff(x).subs(x, zi)

    coeffs = mat.LUsolve(b)
    return sum(coeffs[i] * lps[i] for i in range(num_lps))


def get_upper_bound_laguerre(zs) -> float:
    """Compute upper bound using the Laguerre LP framework.

    Matches the Einstein Arena verifier (uncertainty-principle.ts).
    """
    g_fn = _find_laguerre_combination(zs)
    x = sympy.symbols('x')

    div = sympy.prod([(x - sympy.Rational(z)) ** 2 for z in zs]) * x
    gq_fn = sympy.exquo(g_fn, div)

    real_roots = sympy.real_roots(gq_fn, x)
    gq_np = sympy.lambdify(x, gq_fn, modules="numpy")
    largest_sign_change = 0.0
    for root in real_roots:
        r_val = float(root.evalf(30))
        eps = 1e-6
        if np.sign(gq_np(r_val - eps)) != np.sign(gq_np(r_val + eps)):
            largest_sign_change = max(largest_sign_change, r_val)

    assert largest_sign_change > 0, "No sign-changing roots found."
    return float(largest_sign_change) / (2 * np.pi)
