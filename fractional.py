import sympy as sp
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.transforms import mellin_transform


def rl_frac_deriv_poly(expr, var, order):
    """Calculates the Riemann-Liouville fractional derivative for a polynomial."""
    poly = expr.as_poly(var)
    result = sp.Integer(0)
    for monom, coeff in poly.terms():
        k = monom[0]
        term_deriv = coeff * gamma(k + 1) / gamma(k + 1 - order) * var ** (k - order)
        result += term_deriv
    return result


def calculate_fractional_derivative(expr, var, order):
    """Calculate fractional derivative of a function."""
    try:
        # If polynomial, use Riemann

        if expr.is_polynomial(var):
            result = rl_frac_deriv_poly(expr, var, order)
            result = sp.simplify(result)
            return result, None

        # Else, fallback to Mellin transform

        s = sp.Symbol('s', real=True)
        t = sp.Symbol('t', positive=True)
        # Replace var with t in the expression
        expr_t = expr.subs(var, t)

        # Calculate Mellin transform
        mellin_expr, (a, b), _ = mellin_transform(expr_t, t, s)

        # Apply fractional derivative formula using Mellin transform
        frac_deriv_mellin = mellin_expr * sp.gamma(s) / sp.gamma(s - order)

        # Inverse Mellin transform
        x = sp.Symbol('x', positive=True)

        try:
            result = sp.inverse_mellin_transform(frac_deriv_mellin, s, x, (None, None))
        except:
            result = sp.inverse_mellin_transform(frac_deriv_mellin, s, x, (a, b))

        # Substitute back the original variable
        result = result.subs(x, var)

        return result, None
    except:
        return None, f"Votre fonction n'est pas supportée par l'application pour l'instant"
