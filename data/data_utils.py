import torch
from torch import pi, exp

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Physical constants (in SI/ps units)
hbar = torch.tensor(1.0546e-22, device=device)      # J*ps
kb = torch.tensor(1.3806e-23, device=device)        # J/K


# Define required functions
def rmse(y1, y2):
    return torch.sqrt(torch.mean((y1 - y2)**2))

# Spectral density J
def spec_dens(v, A, v_c):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    v_c: cutoff frequency (1/ps)
    
    '''
    return hbar * A / (pi * kb) * v ** 3 * exp(-(v / v_c) ** 2)


# Phonon occupation number n
def phonon_occ(v, T):
    '''
    v: frequency (1/ps)
    T: bath temperature (K)
    
    '''
    eps = 1e-9
    return 1.0 / (exp(hbar * v / (kb * T)) - 1 + eps)


# define spectral density 
def spec_dens(v, alpha, v_c):
    '''
    v: frequency (1/ps)
    alpha: system-bath coupling strength (dimensionless)
    v_c: cutoff frequency (1/ps)
    '''
    return 2 * alpha * (v**3 / v_c**2) * torch.exp(-(v/v_c)**2)


# define phonon absorption rate 
def phonon_abs(v, v_c, alpha, T):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    T: bath temperature (K)
    v_c: cutoff frequency (1/ps)
    theta: mixing angle (rad)
    
    '''
    return 0.5 * pi * spec_dens(v=v, alpha=alpha, v_c=v_c) * phonon_occ(v=v, T=T)


# define phonon emission rate 
def phonon_emiss(v, v_c, alpha, T):
    '''
    v: frequency (1/ps)
    A: system-bath coupling strength (ps/K)
    T: bath temperature (K)
    v_c: cutoff frequency (1/ps)
    theta: mixing angle (rad)
    
    '''
    return 0.5 * pi * spec_dens(v=v, alpha=alpha, v_c=v_c) * (1 + phonon_occ(v=v, T=T))
