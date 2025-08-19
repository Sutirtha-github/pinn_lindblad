import torch
from data.data_utils import phonon_abs
from figures.visualizations import plot_derivatives

# Grid definitions
omega_vals = torch.linspace(-10.0, 10.0, 100)
alpha_vals = torch.linspace(0.01, 0.3, 100)
omega_c_vals = torch.linspace(1.0, 5.0, 100)

T = 10.0

# Constants
const_params = {
    "alpha": 0.126,
    "omega_c": 3.04
}

# Numerical derivatives
def num_derivative_gamma(param_vals, param_name, omega_vals, const_params):
    """
    Compute partial derivatives of phonon absorption rate w.r.t. alpha and omega_c separately at T = 10K

    Args:
    param_vals : range of alpha/omega_c values
    param_name : "alpha" or "omega_c"
    omega_vals : range of omega values
    const_params : dictionary of true parameter values

    Returns : partial derivative of gamma wrt parameter as a function of omega_vals and param_vals

    """
    Omega, Param = torch.meshgrid(omega_vals, param_vals, indexing='ij')
    h = 1e-3
    if param_name == "alpha":
        gamma_plus = phonon_abs(Omega, const_params["omega_c"], Param + h/2, T)
        gamma_minus = phonon_abs(Omega, const_params["omega_c"], Param - h/2, T)
    elif param_name == "omega_c":
        gamma_plus = phonon_abs(Omega, Param + h/2, const_params["alpha"], T)
        gamma_minus = phonon_abs(Omega, Param - h/2, const_params["alpha"], T)
    else:
        raise ValueError("Invalid parameter name.")
    return (gamma_plus - gamma_minus) / h


# Compute derivatives of phonon absorption rate
dgamma_dalpha = num_derivative_gamma(alpha_vals, "alpha", omega_vals, const_params)
dgamma_domega_c = num_derivative_gamma(omega_c_vals, "omega_c", omega_vals, const_params)


plot_derivatives(omega_vals, alpha_vals, dgamma_dalpha, omega_c_vals, dgamma_domega_c)