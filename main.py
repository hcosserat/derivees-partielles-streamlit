import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


def parse_function(function_str, all_vars):
    """Parse a mathematical function string into a SymPy expression."""
    try:
        expr = parse_expr(function_str, local_dict={var.name: var for var in all_vars})
        return expr, None
    except Exception as e:
        error_msg = f"Attention à la syntaxe de la fonction. Merci de bien utiliser la syntaxe Python (2*x à la place de 2x, x**2 à la place de x^2, sin(x) à la place de sin x, ...). ({str(e)})"
        return None, error_msg


def calculate_derivative(expr, diff_vars):
    """Calculate the derivative with respect to specified variables in order."""
    deriv = expr
    for var in diff_vars:
        deriv = sp.diff(deriv, var)
    return deriv


def calculate_gradient(expr, all_vars):
    """Calculate the gradient vector of the function."""
    gradient = [sp.diff(expr, var) for var in all_vars]
    return sp.Matrix(gradient)


def calculate_jacobian(gradient):
    """Calculate the Jacobian matrix from the gradient."""
    return sp.Matrix([gradient])


def calculate_hessian(expr, all_vars):
    """Calculate the Hessian matrix (second-order partial derivatives)."""
    return sp.Matrix([[sp.diff(sp.diff(expr, var1), var2) for var2 in all_vars] for var1 in all_vars])


def plot_single_variable_function(expr, var):
    """Create plots for a single-variable function and its derivative."""
    try:
        f_lambda = sp.lambdify(var, expr, "numpy")
        df_lambda = sp.lambdify(var, sp.diff(expr, var), "numpy")

        x_range = np.linspace(-5, 5, 100)

        y_vals = np.array([f_lambda(xi) for xi in x_range])
        dy_vals = np.array([df_lambda(xi) for xi in x_range])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

        ax1.plot(x_range, y_vals, 'b-', label=f'f({var})')
        ax1.set_title(f'Fonction f({var})')
        ax1.set_xlabel(f'{var}')
        ax1.set_ylabel(f'f({var})')
        ax1.grid(True)
        ax1.legend()

        ax2.plot(x_range, dy_vals, 'r-', label=f"f'({var})")
        ax2.set_title(f"Dérivée f'({var})")
        ax2.set_xlabel(f'{var}')
        ax2.set_ylabel(f"f'({var})")
        ax2.grid(True)
        ax2.legend()

        return fig, None
    except Exception as e:
        return None, f"La création du graphe a échoué : {str(e)}"


def plot_two_variable_function(expr, vars, function_str):
    """Create a 3D plot for a two-variable function."""
    try:
        x, y = vars
        f_lambda = sp.lambdify((x, y), expr, "numpy")

        x_range = np.linspace(-5, 5, 100)
        y_range = np.linspace(-5, 5, 100)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    Z[i, j] = f_lambda(X[i, j], Y[i, j])
                except:
                    Z[i, j] = np.nan

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax.set_xlabel(x.name)
        ax.set_ylabel(y.name)
        ax.set_zlabel('f(' + x.name + ',' + y.name + ')')
        ax.set_title(f'Graphe 3D de {function_str}')
        fig.colorbar(surf)

        return fig, None
    except Exception as e:
        return None, f"La création du graphe a échoué : {str(e)}"


def display_derivative_results(expr, all_vars, diff_vars):
    """Calculate and display all derivative results."""
    deriv = calculate_derivative(expr, diff_vars)
    st.write(f"### La dérivée de la fonction par rapport à {', '.join([var.name for var in diff_vars])} est :")
    st.latex(sp.latex(deriv))

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
        st.write("### Graphe de la fonction sur [-5, 5]² :")
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


def display_tips():
    """Display usage tips for the application."""
    st.markdown("""
    ### Conseils :
    - Utilisez la syntaxe Python: `2*x` (pas 2x), `x**2` (pas x^2 ou x²), `sin(x)` (pas sin x)
    - Toutes les fonctions ordinaires sont disponibles : sin, cos, tan, exp, log, sqrt, etc.
    """)


def main():
    st.title("Dérivées partielles")
    st.write("Entrez une fonction pour calculer ses dérivées partielles.")

    st.write("### Fonction :")

    function_str = st.text_input("Fonction à dériver", "x**2 + y*sin(z)")

    all_vars_str = st.text_input("Variables de la fonction (séparées par des virgules \",\")", "x,y,z")
    all_vars = [sp.Symbol(var.strip()) for var in all_vars_str.split(',')]

    expr, error = parse_function(function_str, all_vars)

    if expr:
        st.latex(sp.latex(expr))
    elif error:
        st.error(error)

    diff_vars_str = st.text_input("Variables par rapport auxquelles la fonction sera dérivée (dans l'ordre, séparées par des virgules)", "y,z")
    diff_vars = [sp.Symbol(var.strip()) for var in diff_vars_str.split(',')]

    st.write("### Dérivée :")

    try:
        if st.button("Calculer la dérivée") and expr:
            display_derivative_results(expr, all_vars, diff_vars)
    except Exception as e:
        st.error(f"Une erreur s'est produite lors du calcul : {str(e)}")

    display_tips()


if __name__ == "__main__":
    main()
