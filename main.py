import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import parse_expr


def main():
    st.title("Dérivées partielles")
    st.write("Entrez une fonction pour calculer ses dérivées partielles.")

    st.write("### Fonction :")

    function_str = st.text_input("Fonction à dériver", "x**2 + y*sin(z)")

    all_vars_str = st.text_input("Variables de la fonction (séparées par des virgules \",\")", "x,y,z")
    all_vars = [sp.Symbol(var.strip()) for var in all_vars_str.split(',')]

    try:
        expr = parse_expr(function_str, local_dict={var.name: var for var in all_vars})

        st.latex(sp.latex(expr))
    except Exception as e:
        st.error(f"Attention à la syntaxe de la fonction. Merci de bien utiliser la syntaxe Python (2*x à la place de 2x, x**2 à la place de x^2, sin(x) à la place de sin x, ...). ({str(e)})")
        expr = None

    diff_vars_str = st.text_input("Variables par rapport auxquelles la fonction sera dérivée (dans l'ordre, séparées par des virgules)", "y,z")
    diff_vars = [sp.Symbol(var.strip()) for var in diff_vars_str.split(',')]

    st.write("### Dérivée :")

    try:
        if st.button("Calculer la dérivée"):
            deriv = expr

            for var in diff_vars:
                deriv = sp.diff(deriv, var)

            st.write(f"### La dérivée de la fonction par rapport à {', '.join([var.name for var in diff_vars])} est :")
            st.latex(sp.latex(deriv))

            st.write("### Toutes les derivées premières :")
            for var in all_vars:
                partial = sp.diff(expr, var)
                st.write(f"∂f/∂{var.name} = ")
                st.latex(sp.latex(partial))

            # Plot si 2 variables
            if len(all_vars) == 2:
                st.write("### Graphe de la fonction sur [-5, 5]² :")
                try:
                    x, y = all_vars
                    # Convert to numerical function for plotting
                    f_lambda = sp.lambdify((x, y), expr, "numpy")

                    # Create mesh grid
                    x_range = np.linspace(-5, 5, 100)
                    y_range = np.linspace(-5, 5, 100)
                    X, Y = np.meshgrid(x_range, y_range)

                    # Compute Z values (handling potential numerical issues)
                    Z = np.zeros_like(X)
                    for i in range(X.shape[0]):
                        for j in range(X.shape[1]):
                            try:
                                Z[i, j] = f_lambda(X[i, j], Y[i, j])
                            except:
                                Z[i, j] = np.nan

                    # Create plot
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
                    ax.set_xlabel(x.name)
                    ax.set_ylabel(y.name)
                    ax.set_zlabel('f(' + x.name + ',' + y.name + ')')
                    ax.set_title('3D Plot of ' + function_str)
                    fig.colorbar(surf)

                    st.pyplot(fig)
                except Exception as e:
                    st.write("La création du graphe a échoué : ", str(e))

    except Exception as e:
        st.error(f"Attention à la syntaxe de la fonction. Merci de bien utiliser la syntaxe Python (2*x à la place de 2x, x**2 à la place de x^2, sin(x) à la place de sin x, ...). ({str(e)})")

    st.markdown("""
    ### Conseils :
    - Utilisez la syntaxe Python: `2*x` (pas 2x), `x**2` (pas x^2 ou x²), `sin(x)` (pas sin x)
    - Toutes les fonctions ordinaires sont disponibles : sin, cos, tan, exp, log, sqrt, etc.
    """)


if __name__ == "__main__":
    main()
