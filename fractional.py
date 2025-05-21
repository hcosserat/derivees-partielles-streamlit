import sympy as sp
from sympy import mellin_transform, inverse_mellin_transform, InverseMellinTransform
from sympy.functions.special.gamma_functions import gamma


def rl_frac_deriv_power(expr, var, order):
    """Calculates the Riemann-Liouville fractional derivative for power functions (x^a for any real a > -1)."""
    if expr.is_polynomial(var):
        # Handle polynomial case as before
        poly = expr.as_poly(var)
        result = sp.Integer(0)
        for monom, coeff in poly.terms():
            k = monom[0]
            term_deriv = coeff * gamma(k + 1) / gamma(k + 1 - order) * var ** (k - order)
            result += term_deriv
        return result

    # Try to handle non-polynomial power functions like sqrt(x), x^(1/3), etc.
    try:
        # Check if expression is of form c*x^a
        if expr.is_Mul:
            coeff, power_term = None, None
            for arg in expr.args:
                if var in arg.free_symbols:
                    power_term = arg
                else:
                    coeff = arg if coeff is None else coeff * arg

            # Handle x^a form
            if power_term and power_term.is_Pow and power_term.base == var:
                a = power_term.exp
                coeff = sp.Integer(1) if coeff is None else coeff
                return coeff * gamma(a + 1) / gamma(a + 1 - order) * var ** (a - order)

        # Handle direct x^a form
        elif expr.is_Pow and expr.base == var:
            a = expr.exp
            return gamma(a + 1) / gamma(a + 1 - order) * var ** (a - order)

        # Direct handling for sqrt(x) case
        elif expr.has(sp.sqrt(var)):
            # sqrt(x) = x^(1/2)
            a = sp.Rational(1, 2)
            # Extract coefficient if any
            coeff = expr / sp.sqrt(var) if expr != sp.sqrt(var) else 1
            return coeff * gamma(a + 1) / gamma(a + 1 - order) * var ** (a - order)

    except Exception:
        pass

    return None


def calculate_fractional_derivative(expr, var, order):
    """
    Calculate fractional derivative of a function.
    Uses Riemann-Liouville for powers and polynomials, then falls back to Mellin transform with lookup table.
    """
    # Mellin transform lookup table
    # Format: (original_expr_pattern, fractional_derivative)
    mellin_lookup = [
        # Functions in form of x^a
        (lambda expr: expr.is_Pow and expr.base == var,
         lambda expr, order: gamma(expr.exp + 1) / gamma(expr.exp + 1 - order) * var ** (expr.exp - order)),

        # sin(x)
        (lambda expr: expr == sp.sin(var),
         lambda expr, order: {
             0.5: sp.sqrt(2 * sp.pi / var) * sp.sin(var + sp.pi / 4),
             1.5: -sp.sqrt(2 * sp.pi / var) * sp.cos(var + sp.pi / 4),
             # Add more specific orders as needed
         }.get(order, None)),

        # cos(x)
        (lambda expr: expr == sp.cos(var),
         lambda expr, order: {
             0.5: sp.sqrt(2 * sp.pi / var) * sp.cos(var + sp.pi / 4),
             1.5: sp.sqrt(2 * sp.pi / var) * sp.sin(var + sp.pi / 4),
             # Add more specific orders as needed
         }.get(order, None)),

        # exp(x)
        (lambda expr: expr == sp.exp(var),
         lambda expr, order: var ** (-order) * sp.exp(var)),

        # ln(x)
        (lambda expr: expr == sp.log(var),
         lambda expr, order: {
             0.5: sp.sqrt(sp.pi / (2 * var)),
             1: 1 / var,
             # Add more specific orders as needed
         }.get(order, None)),
    ]

    try:
        # First try with extended Riemann-Liouville for powers
        power_result = rl_frac_deriv_power(expr, var, order)
        if power_result is not None:
            return sp.simplify(power_result), "Riemann-Liouville (puissance)"

        # Check lookup table
        for pattern_check, derivative_func in mellin_lookup:
            if pattern_check(expr):
                lookup_result = derivative_func(expr, order)
                if lookup_result is not None:
                    return sp.simplify(lookup_result), "table de transformées de Mellin"

        # If lookup table doesn't have a match, fall back to the original Mellin method
        s = sp.Symbol('s', real=True)
        t = sp.Symbol('t', positive=True)
        expr_t = expr.subs(var, t)

        # Mellin transform code (same as before)
        mellin_expr_tuple = mellin_transform(expr_t, t, s, noconds=False)

        if isinstance(mellin_expr_tuple, sp.integrals.transforms.MellinTransform):
            return None, f"La transformée de Mellin de {expr} n'a pas pu être calculée symboliquement."

        mellin_expr, (a_strip, b_strip), conditions = mellin_expr_tuple
        frac_deriv_mellin = mellin_expr * gamma(s) / gamma(s - order)
        frac_deriv_mellin = sp.simplify(frac_deriv_mellin)
        x_out = var

        try:
            result_inv = inverse_mellin_transform(frac_deriv_mellin, s, x_out, (a_strip, b_strip), noconds=True)
            assert not isinstance(result_inv, InverseMellinTransform)
        except (ValueError, TypeError, AssertionError):
            result_inv = inverse_mellin_transform(frac_deriv_mellin, s, x_out, (None, None), noconds=True)

        if isinstance(result_inv, InverseMellinTransform):
            return None, f"L'inversion de la transformée de Mellin pour {frac_deriv_mellin} n'a pas pu être calculée symboliquement."

        result_inv = sp.simplify(result_inv)
        return result_inv, "transformée de Mellin"

    except Exception as e:
        return None, f"Cette fonction n'est pas supportée par l'application pour le moment: {str(e)}"

