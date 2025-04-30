import matplotlib.pyplot as plt
import numpy as np
import sympy as sp


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

        x_range = np.linspace(-5, 5, 1000)

        y_vals = np.zeros(len(x_range))
        dy_vals = np.zeros(len(x_range))

        for i, xi in enumerate(x_range):
            try:
                y_vals[i] = f_lambda(xi)
            except:
                y_vals[i] = np.nan
            try:
                dy_vals[i] = df_lambda(xi)
            except:
                dy_vals[i] = np.nan

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
