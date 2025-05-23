import streamlit as st
import sympy as sp
from sympy import Equality

from derivatives import (calculate_derivative, calculate_gradient, calculate_jacobian,
                         calculate_hessian, plot_single_variable_function, plot_two_variable_function)
from fractional import calculate_fractional_derivative
from utils import parse_function


def display_tips():
    """Display usage tips for the application."""
    st.markdown("""
    ### Conseils :
    - Vous pouvez utiliser la syntaxe mathématique classique: `2x`, `x^2`, `sin(x)`
    - La syntaxe Python est également acceptée: `2*x`, `x**2`
    - Toutes les fonctions ordinaires sont disponibles : sin, cos, tan, exp, log, sqrt, etc...
    - Les fonctions moins usuelles comme `gamma`, `beta`, `erf`, `erfc`, `Ei`, etc., sont également supportées.
    """)
    st.markdown("""---""")
    st.markdown("""Fait avec :streamlit: Streamlit, SymPy et Matplotlib""")


def display_derivative_results(expr, all_vars, diff_vars):
    """Calculate and display all derivative results."""
    deriv = calculate_derivative(expr, diff_vars)

    # Improved derivative notation
    if len(diff_vars) == 1:
        derivative_notation = f"\\frac{{\\partial f}}{{\\partial {diff_vars[0]}}} = "
    else:
        var_notation = "".join([f"\\partial {var.name}" for var in diff_vars])
        derivative_notation = f"\\frac{{\\partial^{len(diff_vars)}f}}{{{var_notation}}} = "

    st.write(f"### La dérivée de la fonction par rapport à {', '.join([var.name for var in diff_vars])} est :")
    st.latex(derivative_notation + sp.latex(deriv))

    if len(all_vars) > 1:
        gradient = calculate_gradient(expr, all_vars)
        st.write("### Gradient :")
        st.latex(r"\nabla f = " + sp.latex(gradient))

        jacobian = calculate_jacobian(gradient)
        st.write("### Matrice Jacobienne :")
        st.latex("J = " + sp.latex(jacobian))

        hessian = calculate_hessian(expr, all_vars)
        st.write("### Matrice Hessienne :")
        st.latex("H = " + sp.latex(hessian))

    if len(all_vars) == 1:
        st.write("### Graphe de la fonction sur [-5, 5] :")
        fig, error = plot_single_variable_function(expr, all_vars[0])
        if fig:
            st.pyplot(fig)
        if error:
            st.write(error)
    elif len(all_vars) == 2:
        st.write("### Graphe de la fonction sur [-5, 5]² :")
        fig, error = plot_two_variable_function(expr, all_vars, sp.latex(expr))
        if fig:
            st.pyplot(fig)
        if error:
            st.write(error)

    # Calcul des zéros
    st.write("### Zéros de la fonction :")

    try:
        func_roots = sp.solve(expr, all_vars)

        if not func_roots:
            st.info("Aucun zéro n'a été trouvé.")
        else:
            st.write("Ces valeurs annulent la fonction :")
            for i, root in enumerate(func_roots):
                st.write(f"$x_{i} = {sp.latex(root)}$")
    except NotImplementedError:
        st.info("Aucun zéro n'a été trouvé.")

    st.write("### Zéros de la dérivée :")

    try:
        func_roots = sp.solve(deriv, all_vars)

        if not func_roots:
            st.info("Aucun zéro n'a été trouvé.")
        else:
            st.write("Ces valeurs annulent la fonction dérivée :")
            for i, root in enumerate(func_roots):
                x_tilde = r"\tilde{x_" + str(i) + "}"
                st.write(f"${x_tilde} = {sp.latex(root)}$")
    except NotImplementedError:
        st.info("Aucun zéro n'a été trouvé.")


def standard_derivatives_tab():
    st.header("Dérivées standards")

    function_str = st.text_input("Fonction à dériver", "x^2 + y*sin(z)", key="std_function")
    all_vars_str = st.text_input("Variables de la fonction (séparées par des virgules \",\")", "x,y,z", key="std_vars")
    all_vars = [sp.Symbol(var.strip()) for var in all_vars_str.split(',') if var.strip()]

    expr, error = parse_function(function_str, all_vars)

    if isinstance(expr, Equality):
        expr = expr.rhs
    if expr:
        st.latex(sp.latex(expr))
    elif error:
        st.error(error)

    diff_vars_str = st.text_input("Variables par rapport auxquelles la fonction sera dérivée (dans l'ordre)", "y,z")
    diff_vars = [sp.Symbol(var.strip()) for var in diff_vars_str.split(',') if var.strip()]

    st.write("### Dérivée :")

    try:
        if st.button("Calculer", key="std_calc") and expr:
            display_derivative_results(expr, all_vars, diff_vars)
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du calcul : {str(e)}")


def fractional_derivatives_tab():
    st.header("Dérivées fractionnaires")

    function_str = st.text_input("Fonction en $x$ à dériver", "x^3", key="frac_function")
    all_vars = [sp.Symbol("x")]

    expr, error = parse_function(function_str, all_vars)

    if expr:
        st.latex(sp.latex(expr))
    elif error:
        st.error(error)

    order_str = st.text_input("Ordre de la dérivée (par exemple., 1/2, 2, 1+i)", "1/5", key="frac_order")

    try:
        order = sp.sympify(order_str)
        valid_order = True
    except Exception:
        valid_order = False
        st.error("L'ordre entré n'est pas valide. Utilisez des expressions comme 1/2, 2, 1+i")

    st.write("### Dérivée fractionnaire :")

    try:
        if st.button("Calculer", key="frac_calc") and expr and valid_order:
            frac_result, frac_msg = calculate_fractional_derivative(expr, all_vars[0], order)
            if frac_result:
                st.write(f"### La dérivée fractionnaire d'ordre ${sp.latex(order)}$ par rapport à ${all_vars[0]}$ est :")
                st.info(f"Méthode : {frac_msg}")
                st.latex(f"D^{{{sp.latex(order)}}}_{{{all_vars[0]}}} f = " + sp.latex(frac_result))
            elif frac_msg:
                st.error(frac_msg)
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du calcul : {str(e)}")
