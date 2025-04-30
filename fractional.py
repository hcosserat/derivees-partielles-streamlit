import sympy as sp


def calculate_fractional_derivative(expr, var, order):
    """Calculate fractional derivative of a function."""
    try:
        from sympy.integrals.transforms import mellin_transform
        from sympy.functions.special.gamma_functions import gamma

        s = sp.Symbol('s', real=True)
        # Use Mellin transform approach for fractional derivative
        t = sp.Symbol('t', positive=True)
        # Replace var with t in the expression
        expr_t = expr.subs(var, t)

        # Calculate Mellin transform
        mellin_expr, a, b = mellin_transform(expr_t, t, s)

        # Apply fractional derivative formula using Mellin transform
        frac_deriv_mellin = mellin_expr * sp.gamma(s) / sp.gamma(s - order)

        # Inverse Mellin transform
        x = sp.Symbol('x', positive=True)
        result = sp.inverse_mellin_transform(frac_deriv_mellin, s, x, (a, b))

        # Substitute back the original variable
        result = result.subs(x, var)

        return result, None
    except Exception as e:
        return None, f"Erreur dans le calcul de la dérivée fractionnaire : {str(e)}"
