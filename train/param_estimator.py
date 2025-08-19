import torch
from figures.visualizations import plot_midtraining_inv1, plot_midtraining_inv2
from train.pinn_simulation import BlochNN, compute_physics_loss
from data.data_utils import phonon_abs, phonon_emiss




def train_inv_pinn_1(alpha0, v_c, T, D, t_obs, sx_obs, sy_obs, sz_obs, t_test, t_intervals=150, 
                     hidden_dim=32, n_layers=4, epochs=301, lr=5e-3, lam1=1, lam2=1e2, plot_interval=50):

    """
    Estimate the value of system-bath coupling strength 'A' (in ps/K units) from the given 
    noisy dataset, differential equations and initial boundary conditions.

    Args:
      A0: actual known value of coupling strength (ps/K)
      v_c: cutoff frequency (1/ps)
      T: bath temperature (K)
      D: pulse duration (ps)
      t_obs: time instances of the observed noisy dataset (ps)
      sx_obs, sy_obs, sz_obs: noisy dataset
      t_test: time points over the entire domain [0,D] for inferencing
      t_intervals: # time points between [0,D]
      hidden_dim: # units per hidden layer
      n_layers: # hidden layers
      epochs: # training iterations
      lr: learning rate
      plot_interval: frequency of displaying learning plots

    Returns:
      alpha, pinn: predicted value of coupling strength, trained pinn model
    """
    
    torch.manual_seed(2025)

    eps = 1e-10

    # define PINN and additional learnable parameter 'alpha' to learn the system-bath coupling strength (in ps/K units)
    pinn = BlochNN(1, 3, hidden_dim, n_layers)
    alpha = torch.nn.Parameter(1 * torch.ones(1, requires_grad=True))    
                       
    optimiser = torch.optim.Adam(list(pinn.parameters()) + [alpha], lr=lr)

    t_physics = torch.linspace(0, D, t_intervals).view(-1, 1).requires_grad_(True)

    alpha_list = []

    for i in range(epochs):
        optimiser.zero_grad()

        # Compute the time dependent parameters over the entire time domain
        om = torch.pi/D
        ga = phonon_abs(v=om, v_c=v_c, alpha=alpha, T=T)
        ge = phonon_emiss(v=om, v_c=v_c, alpha=alpha, T=T)

        # Inference the NN for predictions over the entire time domain
        s_pred = pinn(t_physics)

        # Calculate physics loss over the entire time domain
        loss_p = compute_physics_loss(s_pred, t_physics, om, ga, ge)

        # Inference the NN for predictions over the observation time instances only
        s_obs_pred = pinn(t_obs)
        sx_obs_pred, sy_obs_pred, sz_obs_pred = s_obs_pred[:, 0:1], s_obs_pred[:, 1:2], s_obs_pred[:, 2:3]

        # Calculate data loss as the mean loss between observed data and NN predictions squared
        loss_d = torch.mean((sx_obs - sx_obs_pred)**2 + (sy_obs - sy_obs_pred)**2 + (sz_obs - sz_obs_pred)**2)

        # Total loss
        loss = lam1 * loss_p + lam2 * loss_d
        loss.backward()
        optimiser.step()

        alpha_list.append(alpha.item())

        if (i+1) % plot_interval == 0:
            s = pinn(t_test).detach()
            plot_midtraining_inv1(i, alpha0, alpha_list, t_test, s, t_obs, sx_obs, sy_obs, sz_obs)

    return alpha.item(), pinn





def train_inv_pinn_2(v_c0, alpha, T, D, t_obs, sx_obs, sy_obs, sz_obs, t_test, t_intervals=150, 
                     hidden_dim=32, n_layers=4, lambda1=1e2, lr=1e-2, seed=2021, epochs=1001, plot_intervals=100):

    """
    Estimate the value of cutoff frequency 'v_c_pred' (in 1/ps units) from the given 
    noisy dataset, differential equations and initial boundary conditions.

    Args:
      v_c0: actual known value of cutoff frequency (1/ps)
      alpha: coupling strength (dimensionless)
      T: bath temperature (K)
      D: pulse duration (ps)
      t_obs: time instances of the observed noisy dataset (ps)
      sx_obs, sy_obs, sz_obs: noisy dataset
      t_test: time points over the entire domain [0,D] for inferencing
      t_intervals: # time points between [0,D]
      hidden_dim: # units per hidden layer
      n_layers: # hidden layers
      lambda1: regularization hyperparamter
      lr: learning rate
      seed: for reproducibility of results
      epochs: # training iterations
      plot_intervals: frequency of displaying learning plots

    Returns:
      v_c_pred, pinn: predicted value of cutoff frequency, trained pinn model
    """
    
    
    torch.manual_seed(seed)
    
                      
    # define PINN and additional learnable parameter 'v_c_pred' to learn the bath cutoff frequency (in 1/ps units)
    pinn = BlochNN(1, 3, hidden_dim, n_layers)
    v_c_pred = torch.nn.Parameter(5 * torch.ones(1, requires_grad=True))      
    t_physics = torch.linspace(0, D, t_intervals).view(-1, 1).requires_grad_(True)                  
    optimiser = torch.optim.Adam(list(pinn.parameters()) + [v_c_pred], lr=lr)
    v_c_list = []

    for i in range(epochs):
        optimiser.zero_grad()

        # Compute the time dependent parameters over the entire time domain
        om = torch.pi/D
        ga = phonon_abs(v=om, v_c=v_c_pred, alpha=alpha, T=T)
        ge = phonon_emiss(v=om, v_c=v_c_pred, alpha=alpha, T=T)
        
        # Inference the NN for predictions over the entire time domain
        s_pred = pinn(t_physics)

        loss_p = compute_physics_loss(s_pred, t_physics, om, ga, ge)

        # Inference the NN for predictions over the observation time instances only
        s_obs_pred = pinn(t_obs)
        sx_obs_pred, sy_obs_pred, sz_obs_pred = s_obs_pred[:, 0:1], s_obs_pred[:, 1:2], s_obs_pred[:, 2:3]

        # Calculate data loss as the mean loss between observed data and NN predictions squared
        loss_d = torch.mean((sx_obs - sx_obs_pred) ** 2 + (sy_obs - sy_obs_pred) ** 2 + (sz_obs - sz_obs_pred) ** 2)

        # Total loss
        loss = loss_p + lambda1 * loss_d
        loss.backward()
        optimiser.step()

        v_c_list.append(v_c_pred.item())

        if (i+1) % plot_intervals == 0:
            s = pinn(t_test).detach()
            plot_midtraining_inv2(i, v_c0, v_c_list, t_test, s, t_obs, sx_obs, sy_obs, sz_obs)

    return v_c_pred.item(), pinn

