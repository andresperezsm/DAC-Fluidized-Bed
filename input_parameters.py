import numpy as np
from scipy.constants import gas_constant as R_gas

def init_params_PB(model):
    """
    Initialize the parameters for the packed bed model.

    Parameters:
    - model: An instance of the packed bed model to be initialized.
    """
    # Temperatures
    model.T_feed = 298            # Suggested temperature range (0 - 120 C) Optmized(32 (both), 38-41 (0.8-0.2), 20-24 (0.8, 2))
    model.T_initial = 298
    model.T_wall = 298
    model.desorption = False
    model.adiabatic = False

    # Pressures and humidity
    model.p_CO2_initial =  0
    model.p_CO2_feed = 400e-6 * 101325

    model.p_tot_initial = 101325
    model.p_tot_feed = 101325

    # Bed Column parameters
    model.L_D_ratio = 3             # Length to Radius ratio
    model.R_b = 0.25                # column radius [m]
    model.L_b = 0.5                 # column lenght [m] before fluidization
    model.eps_b = 0.4438            # packed-bed porosity
    model.v_inlet = 0.09             # axial velocity [m/s]

    # Particle parameters
    model.R_p = 2e-4                 # particle radius [m]
    model.tau_p = 2.3                # particle tortuosity
    model.rho_p = 744                # bed density [kg/m3] 
    model.eps_p = 0.338              # particle porosity (Ashlyn Low 2023)
    model.r_pore = 28.5e-9           # pore radius [m]                            

    # Solver options
    model.dt = 1000             # Set time-step to infinity for steady-state solution
    model.t_end = 900000 # 500 #5000                   # Set total simulated time
    model.Nr = 1                                     # Set number of gridpoints
    model.Nz = 100                                     # Set number of gridpoints
    model.maxfev = 100                                 # Maximum function evaluations in the solver
    model.tol = 1e-6
    model.verbose = False
    model.outer_message = True





