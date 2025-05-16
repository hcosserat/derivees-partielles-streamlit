import sympy as sp
from sympy.functions.special.gamma_functions import gamma
from sympy.integrals.transforms import mellin_transform, inverse_mellin_transform, InverseMellinTransform


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
    """
    Calculate fractional derivative of a function.
    For polynomials, it uses the Riemann-Liouville formula term by term.
    For other functions, it attempts to use a Mellin transform method.
    The Mellin transform method implemented here corresponds to a specific definition:
    M{D_M^alpha f(x); s} = [Gamma(s) / Gamma(s - alpha)] * M{f(x); s}
    """
    try:
        # Essayez d'abord de traiter comme un polynôme
        if expr.is_polynomial(var):
            poly_result = rl_frac_deriv_poly(expr, var, order)
            if poly_result is not None:
                poly_result = sp.simplify(poly_result)
                return poly_result, "Riemann-Liouville (polynôme)"

        # Si ce n'est pas un polynôme ou si rl_frac_deriv_poly a échoué pour une raison,
        # fallback vers la transformée de Mellin.
        s = sp.Symbol('s', real=True)  # Variable pour la transformée de Mellin
        t = sp.Symbol('t', positive=True)  # Variable d'intégration muette

        expr_t = expr.subs(var, t)

        # Calculer la transformée de Mellin
        # mellin_transform retourne (F, (a,b), cond)
        # F = transformée, (a,b) = bande de convergence, cond = conditions auxiliaires
        mellin_expr_tuple = mellin_transform(expr_t, t, s, noconds=False)  # noconds=False est le défaut

        if isinstance(mellin_expr_tuple, sp.integrals.transforms.MellinTransform):
            # La transformée n'a pas pu être évaluée
            return None, f"La transformée de Mellin de {expr} n'a pas pu être calculée symboliquement."

        mellin_expr, (a_strip, b_strip), conditions = mellin_expr_tuple

        # Appliquer la formule de dérivation fractionnaire via la transformée de Mellin
        # D_M^alpha[f](s) = Gamma(s)/Gamma(s-alpha) * F(s)
        # Note: Gamma(s-order) peut introduire de nouveaux pôles ou modifier la bande de convergence.
        # La bande pour l'inversion devra idéalement être choisie dans l'intersection
        # de la bande de F(s) et la région où Gamma(s)/Gamma(s-order) est analytique.
        frac_deriv_mellin = mellin_expr * gamma(s) / gamma(s - order)

        # Simplification avant inversion peut aider
        frac_deriv_mellin = sp.simplify(frac_deriv_mellin)

        # Inversion de la transformée de Mellin
        x_out = var  # Utiliser la variable originale pour la sortie

        # Essayer avec la bande de convergence originale, puis sans spécification
        # L'utilisateur de inverse_mellin_transform doit choisir c tel que a_strip < c < b_strip
        # et que tous les pôles de frac_deriv_mellin à gauche de la ligne d'intégration Re(s)=c soient pris en compte.
        # SymPy tente d'inférer cela si strip est (None, None).

        # On pourrait essayer d'ajuster la bande ici, mais c'est complexe.
        # Par exemple, les pôles de Gamma(s) sont à s = 0, -1, -2, ...
        # Les pôles de 1/Gamma(s-order) sont où s-order = 0, -1, -2,... (donc s = order, order-1, ...)
        # La bande (a_strip, b_strip) doit être compatible avec ces pôles.

        result_inv = inverse_mellin_transform(frac_deriv_mellin, s, x_out, (a_strip, b_strip), noconds=True)

        if isinstance(result_inv, InverseMellinTransform):  # Si l'inversion a retourné un objet non évalué
            # Tenter sans spécifier la bande explicitement, laissant SymPy décider
            result_inv = inverse_mellin_transform(frac_deriv_mellin, s, x_out, (None, None), noconds=True)

        if isinstance(result_inv, InverseMellinTransform):
            return None, f"L'inversion de la transformée de Mellin pour {frac_deriv_mellin} n'a pas pu être calculée symboliquement."

        result_inv = sp.simplify(result_inv)
        return result_inv, "transformée de Mellin"

    except Exception as e:
        return None, f"Une erreur est survenue : {e}"
