import re

from sympy.parsing.sympy_parser import parse_expr


def parse_function(function_str, all_vars):
    """Parse a mathematical function string into a SymPy expression."""
    try:
        # Replace classical notations with Python syntax
        function_str = function_str.replace("^", "**")

        # Handle implicit multiplication (2x -> 2*x)
        for var in all_vars:
            var_name = var.name
            # Replace instances like "2x" with "2*x" but avoid replacing "x2" or "ax"
            function_str = re.sub(r'(\d)(' + var_name + r')', r'\1*\2', function_str)

        expr = parse_expr(function_str, local_dict={var.name: var for var in all_vars},
                          transformations='all')
        return expr, None
    except Exception as e:
        error_msg = f"Erreur dans la syntaxe de la fonction: {str(e)}"
        return None, error_msg
