import numpy as np
from scipy.sparse import csc_array
import scipy.sparse.linalg as sla
import math
from pymrm import (newton, construct_div, non_uniform_grid, construct_coefficient_matrix, 
                   construct_grad, numjac_local, interp_cntr_to_stagg, clip_approach)
from scipy.constants import gas_constant as R_gas
from matplotlib import pyplot as plt
import scipy.integrate as integrate
import pandas as pd
from input_parameters import init_params_PB

from config import PackedBedFactory

class ParticleModelFick:
    """
    Class representing a 1D spherical symmetric dry adsorption-diffusion model.
    Governing equation: eps_p(1/RT dp/dt) + rho_p(dq/dt) = div(N_diff)
    Diluted-isobaric conditions: solving for n-1 components (CO2, H2O).
    Fick diffusion model: driving force = - 1/RT grad(p).
    Components structure: [p_CO2 p_H2O, q_CO2, q_H2O].
    
    Attributes: 
    shape_p (tuple): Dimensions of the partial pressures grid (r, species).
    dr_large (float): Initial large grid spacing.
    refinement_factor (float): Factor for refining the grid.
    kinetics (callable): Kinetics model function.
    r_f (numpy.ndarray): Radial face grid points.
    r_c (numpy.ndarray): Radial center grid points.
    bc (dict): Boundary conditions.
    Flux (scipy.sparse.csc_matrix): Flux matrix.
    flux_bc (numpy.ndarray): Flux boundary conditions.
    Jac_sf (scipy.sparse.csc_matrix): Jacobian matrix for the species flux.
    Jac_rate_app_s (scipy.sparse.csc_matrix): Jacobian matrix for the apparent reaction rate concerning species concentration.
    Jac_rate_app_f (scipy.sparse.csc_matrix): Jacobian matrix for the apparent reaction rate concerning fluid concentration.
    Jac_const (scipy.sparse.csc_matrix): Constant part of the Jacobian matrix.
    Jac_ss (scipy.sparse.csc_matrix): Steady-state Jacobian matrix.
    """

    def __init__(self, init_params=None, kinetics=None, thermo=None, co_adsorption=False):
        """
        Initialize the ParticleModel class.

        Parameters:
        - init_params (function): Optional function to initialize model parameters.
        - kinetics (list of functions): Optional functions to set the adsorption kinetics.
        - thermo (list of functions): Optional functions to set the thermodynamic models.
        - co_adsorption (bool): Flag to indicate if co-adsorption is considered.
        """
        self.co_adsorption = co_adsorption

        # Ensure thermo and kinetics are provided
        if thermo is None or len(thermo) < 2:
            raise ValueError("Thermodynamic models for both CO2 and H2O must be provided.")
        if kinetics is None or len(kinetics) < 2:
            raise ValueError("Kinetic models for both CO2 and H2O must be provided.")

        # Set the thermodynamic and kinetic models
        if thermo is not None:
            self.thermo_CO2 = thermo[0]
            self.thermo_H2O = thermo[1]

        if kinetics is not None:
            self.kin_CO2 = kinetics[0]
            self.kin_H2O = kinetics[1]

        self.non_uniform_grid = True
        self.geometry = 2
        self.tau_model = 'Wakao-Smith'
        self.external_MTL_toggle = False

        # Default parameters (constants)
        self.p_tot = 101325                 # [Pa]

        self.MW_CO2 = 44.01/1000            # [kg/mol]
        self.MW_H2O = 18.01528/1000         # [kg/mol]
        self.MW_N2 = 28.0134/1000           # [kg/mol]   

        self.sigma_CO2= 3.941             # [Å]
        self.sigma_H2O= 2.641             # [Å]
        self.sigma_N2=  3.798             # [Å]

        self.eps_k_CO2= 195.2             # [K] 
        self.eps_k_H2O= 809.1             # [K] 
        self.eps_k_N2= 71.4               # [K] 

        self.lambda_g_CO2= 0.0166         # [W/m/K] (298K, 1atm) The Engineering Toolbox
        self.lambda_g_H2O= 0.02457        # [W/m/K] (398K, 1atm) The Engineering Toolbox
        self.lambda_g_N2= 0.02583         # [W/m/K] (298K, 1atm) The Engineering Toolbox

        self.cp_g_CO2= 0.84*1000          # [J/kg/K] (298K, 1atm) The Engineering Toolbox
        self.cp_g_H2O= 1.864*1000         # [J/kg/K] (298K, 1atm) The Engineering Toolbox
        self.cp_g_N2= 1.04*1000           # [J/kg/K] (298K, 1atm) The Engineering Toolbox                        
        
        self.Nc_diff = 2
        self.Nc_nodiff = 2

        # User-specified parameters 
        if init_params is not None:
            init_params(self)

        # Check for input data errors 
        self.exceptions()

        # Total number of species and shape
        self.Nc = self.Nc_diff + self.Nc_nodiff
        self.shape_p = (self.Nr, self.Nc)

        self.size_p_f = math.prod(self.shape_p[1:])
        print(f'THE SIZE IS {self.size_p_f}')

        self.N_faces = self.Nr + 1

         # Default radial grid setup
        self.dr_large = 0.1 * self.R_p    # Initial large grid spacing
        self.refinement_factor = 0.75     # Factor for refining the grid
        if self.non_uniform_grid:
            self.r_f = non_uniform_grid(0, self.R_p, self.shape_p[0] + 1, self.dr_large, self.refinement_factor)
        else:
            self.r_f = np.linspace(0, self.R_p, self.N_faces, dtype=np.float64)
        self.r_c = 0.5 * (self.r_f[:-1] + self.r_f[1:])
        self.r_c2 = self.r_c**self.geometry
        self.r_f2 = self.r_f**self.geometry

        # Geometric parameters
        self.a_v_star = ((self.geometry+1)/self.R_p) * (self.eps_p/self.rho_p) 

        # Initialize diffusion matrix
        self.init_diff(tau_model=self.tau_model)

        # Initialize external mass transfer
        self.init_EMT()

        # Initialize Jacobian matrices for the system
        self.init_Jac()

    def exceptions(self):
        '''
        Check if values exceed the recommended interval.
        '''
        if self.rh_feed > 0.9 or self.rh_initial > 0.9:
            raise Exception('Please enter humidity value from the valid range (0 - 0.9)')
        if self.T_feed > 120+273.15 or self.T_initial > 120+273.15:
            print('Temperature exceedes the suggested operating window, accuracy may decrease')

    def set_ic_bc(self, p, p_f):
        """
        Set the initial conditions and the boundary conditions.

        Parameters:
        - p (numpy.ndarray): Species partial pressure.
        - p_f (numpy.ndarray): Fluid partial pressure at the boundary.

        """
        # Set initial CO2 and H2O partial pressures
        p[:, 0] = self.p_CO2_initial + 1e-32
        p[:, 1] = self.rh_initial * self.thermo_H2O.antoine_pressure(self.T_initial) + 1e-32

        # Calculate initial H2O adsorption
        q_h2o = self.thermo_H2O.isotherm(self.rh_initial * self.thermo_H2O.antoine_pressure(self.T_initial), self.T_initial)[0] + 1e-32

        # Calculate initial CO2 adsorption based on co-adsorption flag
        if self.co_adsorption:
            p[:, 2] = self.thermo_CO2.isotherm(self.p_CO2_initial, q_h2o, self.T_initial)[0] + 1e-32
        else:
            p[:, 2] = self.thermo_CO2.isotherm(self.p_CO2_initial, self.T_initial)[0] + 1e-32

        p[:, 3] = q_h2o

        # Set boundary conditions
        p_f[:, 0] = self.p_CO2_feed + 1e-32
        p_f[:, 1] = self.rh_feed * self.thermo_H2O.antoine_pressure(self.T_feed) + 1e-32

        # Calculate boundary H2O adsorption
        q_f_h2o = self.thermo_H2O.isotherm(self.rh_feed * self.thermo_H2O.antoine_pressure(self.T_feed), self.T_feed)[0] + 1e-32

        # Calculate boundary CO2 adsorption based on co-adsorption flag
        if self.co_adsorption:
            p_f[:, 2] = self.thermo_CO2.isotherm(self.p_CO2_feed, q_f_h2o, self.T_feed)[0] + 1e-32
        else:
            p_f[:, 2] = self.thermo_CO2.isotherm(self.p_CO2_feed, self.T_feed)[0] + 1e-32

        p_f[:, 3] = q_f_h2o

        # Set bulk and adsorbed phase concentrations at the reactor boundaries
        self.c_bulk_R = p_f[..., :self.Nc_diff] / R_gas / self.T_feed
        self.q_ads_R = p_f[..., self.Nc_diff:]

        # Set initial bulk and adsorbed phase concentrations
        self.c_bulk_initial = p[0, :self.Nc_diff] / R_gas / self.T_feed
        self.q_ads_initial = p[0, self.Nc_diff:]

        self.p = p
        self.p_f = p_f

    def init_diff(self, tau_model='Wakao-Smith'):
        """
        Initializes the diffusion properties including tortuosity factors, Knudsen diffusion coefficients,
        molecular diffusion coefficients, and the viscosity matrix.
        It allows the selection of different tortuosity models which affect the calculation of diffusion coefficients.
        
        Parameters:
        - tau_model (str): Specifies the tortuosity model to use, defaults to 'Wakao-Smith'. Supported models include
        'Weissberg', 'Bruggeman', 'Wakao-Smith', 'Beeckman', and 'Maxwell'.
        
        Raises:
        - Exception: If an invalid tortuosity model is selected.
        
        Outputs:
        - Prints the selected tortuosity model and its factor, Knudsen diffusion coefficients for CO2, H2O, N2,
        and molecular diffusion coefficients for pair interactions between these molecules. It also initializes
        the molecular diffusion matrix and viscosity matrix based on Chapman-Enskog theory for the mixture viscosity.
        """
        # Tortuosity factor
        match tau_model:
            case 'Weissberg':
                tau2 = (1 - 0.5*np.log(self.eps_p))
                print(f'Selected tortuosity model: {tau_model}, tau factor = {tau2:0.2f}')
            case 'Bruggeman':
                tau2 = (self.eps_p**(-0.5))
                print(f'Selected tortuosity model: {tau_model}, tau factor = {tau2:0.2f}')
            case 'Wakao-Smith':
                tau2 = 1/self.eps_p
                print(f'Selected tortuosity model: {tau_model}, tau factor = {tau2:0.2f}')
            case 'Beeckman':
                tau2 = (self.eps_p/(1-(1-self.eps_p)**(1/3)))
                print(f'Selected tortuosity model: {tau_model}, tau factor = {tau2:0.2f}')
            case 'Maxwell':
                tau2 = (1 + 0.5*(1 - self.eps_p))**2
                print(f'Selected tortuosity model: {tau_model}, tau factor = {tau2:0.2f}')
            case 'Abbasi': 
                tau2 = self.eps_p/(0.4*self.eps_p - 0.0328)
                print(f'Selected tortuosity model: {tau_model}, tau factor = {tau2:0.2f}')
            case 'User': 
                tau2 = self.tau2
                print(f'Selected tortuosity model: {tau_model}, tau factor = {tau2:0.2f}')
            case _:
                raise Exception("Please select a valid tortuosity model for molecular diffusion")

        # Knudsen diffusion
        tau_Kn = tau2
        self.D_kn = 1/tau_Kn  * 2/3 * self.r_pore * np.sqrt(8 * R_gas * self.T_feed / np.array([self.MW_CO2, self.MW_H2O]) / np.pi)
        print(f'Knudsen diffusion coefficients m2/s: CO2 {self.D_kn[0]*tau_Kn:0.2E}, H2O {self.D_kn[1]*tau_Kn:0.2E}')   

        # Molecular diffusion in nitrogen (excess)
        binary_par_1n = [[self.MW_CO2, self.MW_N2], [self.sigma_CO2, self.sigma_N2], [self.eps_k_CO2, self.eps_k_N2],]
        binary_par_2n = [[self.MW_H2O, self.MW_N2], [self.sigma_H2O, self.sigma_N2], [self.eps_k_H2O, self.eps_k_N2],]
        self.D_1n0 = 1/tau2 * self.binary_diff_coeff(*binary_par_1n, self.T_feed)/1e4  
        self.D_2n0 = 1/tau2 * self.binary_diff_coeff(*binary_par_2n, self.T_feed)/1e4 
        print(f'Molecular diffusion coefficients m2/s: CO2-N2 {self.D_1n0*tau2:0.2E}, H2O-N2 {self.D_2n0*tau2:0.2E}')   

        # Effective diffusion coefficients in Nitrogen excess (Bosanquet)
        D_eff = np.append((1/self.D_kn + [1/self.D_1n0, 1/self.D_2n0])**(-1), [0, 0])
        self.tau_D = self.R_p**2 / D_eff[:2]
        self.D_matrix = construct_coefficient_matrix([D_eff], self.shape_p, axis=0)
        print(f'Effective diluted diffusion coefficients m2/s: CO2 {D_eff[0]:0.2E}, H2O {D_eff[1]:0.2E}\n')

    def init_EMT(self):
        """
        Initialize the external mass transfer model at the particle boundary.
        """
        # Default boundary conditions
        self.bc = {'a': [1, 0], 'b': [0, 1], 'd': [0, 1]}
        print(f'Selected external mass transfer model: no resistances\n')

        # External mass transfer boundary conditions
        if self.external_MTL_toggle:
            Sh = 10
            k_m = Sh*self.D_eff/(self.R_p*2)
            self.bc = {'a': [[[1]], [self.D_eff]], 'b': [[[0]], [k_m]], 'd': [[[0]], [k_m]]}
            print(f'Selected external mass transfer model: film resistances')
            print(f'Mass transfer coefficient m/s: CO2 {k_m[0]:0.4f}, H2O {k_m[1]:0.4f} at Sh = {Sh}\n')

    def init_Jac(self):
        """
        Initialize the Jacobian matrices for the system.
        This includes setting up the accumulation, gradient, and divergence terms.
        """
        Jac_accum = construct_coefficient_matrix(1/self.dt, self.shape_p)
        
        # Gradient and flux matrices
        self.Grad, self.grad_bc = construct_grad(self.shape_p, self.r_f, self.r_c, self.bc, axis=0)
        self.Div = construct_div(self.shape_p, self.r_f, nu=self.geometry)

        # Constant part of the Jacobian matrix
        self.Jac_const = Jac_accum
        
        # Construct the diffusion SF matrix
        self.size_p_f = math.prod(self.shape_p[1:])
        self.indx_j = np.arange(self.size_p_f)
        self.indx_i = self.size_p_f * self.shape_p[0] + self.indx_j
        self.grad_bc_sf = csc_array((np.asarray(self.grad_bc[self.indx_i, [0]]), (self.indx_i, self.indx_j)), 
                            shape=(self.size_p_f * (self.shape_p[0] + 1), self.size_p_f))    
    
    def compute_Diff_flux(self, p, p_f):
        """
        Calculates both molecular and Knudsen and diffusion fluxes using updated pressure profiles. 
        Parameters:
        - p (numpy.ndarray): Current partial pressures of the species in the computational domain.
        - p_f (numpy.ndarray): Partial pressures at tKhe fluid boundary.

        Returns:
        - Tuple: A tuple containing the diffusion Jacobian matrixes
        """
        # Diffusion flux
        self.diff_Flux = - self.D_matrix @ self.Grad
        self.diff_Flux_sf = - self.D_matrix @ self.grad_bc_sf
        
        # Diffusion Jacobian
        J_diff = - self.Div @ self.diff_Flux
        J_sf = - self.Div @ self.diff_Flux_sf

        return J_diff, J_sf

    def construct_g(self, p, p_old, p_f):
        """
        Construct the residual vector g and the Jacobian matrix for the system.

        Parameters:
        - p (numpy.ndarray): Current partial pressure field.
        - p_old (numpy.ndarray): Previous partial pressure field.
        - p_f (numpy.ndarray): Fluid partial pressure at the boundary.

        Returns:
        - g (numpy.ndarray): Residual vector.
        - Jac (scipy.sparse.csc_matrix): Jacobian matrix.
        """
        # Reaction term and its Jacobian
        g_react, Jac_react = numjac_local(lambda p: self.kinetics(p, CO2=self.kin_CO2, H2O=self.kin_H2O), p)

        # Diffusion term and its Jacobian
        Jac_diff, Jac_sf = self.compute_Diff_flux(p, p_f)
        
        # Residual vector
        g = ((self.Jac_const - Jac_diff) @ p.reshape((-1, 1)) - Jac_sf @ p_f.reshape((-1, 1)) + 
             - p_old.reshape((-1, 1))/self.dt - g_react.reshape((-1, 1)))
        
        # Jacobian matrix
        Jac = self.Jac_const - Jac_react - Jac_diff
        
        return g, Jac

    def rate_app(self, p, p_f):
        """
        Compute the apparent reaction rate based on species and fluid partial pressures.

        Parameters:
        - p (numpy.ndarray): Species partial pressure.
        - p_f (numpy.ndarray): Fluid partial pressure at the boundary.

        Returns:
        - rate (numpy.ndarray): Apparent reaction rate mol/kg/s.
        """
        last_face = range(-self.shape_p[-1], 0)

        r_app = self.a_v_star/R_gas/self.T_feed*(self.diff_Flux[last_face, :] @ p.reshape((-1, 1)) + self.diff_Flux_sf[last_face, :] @ p_f.reshape((-1, 1))).reshape(-1) 
        
        return r_app
    
    def solve_transient(self):
        """
        Solve the system for a specified number of time steps.

        Parameters:
        - p (numpy.ndarray): Initial species partial pressure.
        - p_f (numpy.ndarray): Fluid partial pressure at the boundary.

        Returns:
        - p (numpy.ndarray): Updated species partial pressure after solving.
        - p_mean_r (numpy.ndarray): Volume average partial pressure field (*(Rp^3/3)).
        """
        print(f"PDEs system: start solving")

        p = self.p.copy()
        p_f = self.p_f.copy()

        step = 0
        self.max_steps = int(self.t_end / self.dt) + 1

        p_mean_r = np.zeros((self.Nc, self.max_steps))

        p_stagg = interp_cntr_to_stagg(p, self.r_f, self.r_c, axis=0)
        p_stagg[0] = p_stagg[1]
        p_stagg[-1] = p_f

        r_app = np.zeros((self.Nc, self.max_steps))
        t_SS = np.full(self.Nc, np.inf)  # Initialize with inf to indicate unrecorded times
        tol = np.array([1e-3, 1e-3, 1e-4, 1e-4])  # Tolerance below which steady state is considered reached

        # Newton-Raphson iterations
        for t in np.linspace(0, self.t_end, self.max_steps):
            #print(f"\ntime: {t:.5f}")
            p_mean_r[:, step] = [(self.geometry+1)/self.R_p**(self.geometry+1) * integrate.simpson(p_stagg[:, i] * self.r_f2, self.r_f) for i in range(self.Nc)]
            p_old = p.copy()  
            result = newton(lambda p: self.construct_g(p, p_old, p_f), p_old, maxfev=self.maxfev, callback=lambda x,f:clip_approach(x, f))
            p = result.x  
            #print(result.message)

            # Store data for post-processing and update staggered pressures
            r_app[:, step] = self.rate_app(p, p_f)
            p_stagg = interp_cntr_to_stagg(p, self.r_f, self.r_c, axis=0)
            p_stagg[0] = p_stagg[1]
            p_stagg[-1] = p_f

            if self.update_plot_toggle:
                self.update_plot(p, t)

            # Efficiently check and update times using numpy
            mask = (np.abs(p_mean_r[:, step] - p_f[0, :]) < tol) & (t_SS == np.inf)
            t_SS[mask] = t/3600
            step += 1

        print(f"PDEs system: end solving")   

        # Convert infinite values back to None or a suitable placeholder if needed
        self.t_SS = np.where(t_SS == np.inf, None, t_SS)

        return p, p_mean_r, r_app
    
    def kinetics(self, p, CO2=None, H2O=None):
        """
        Calculate the reaction rates for CO2 and H2O.
        
        Parameters:
        - p (numpy.ndarray): Partial pressures of the species.
        - CO2: Kinetic model for CO2.
        - H2O: Kinetic model for H2O.
        
        Returns:
        - r (numpy.ndarray): Reaction rates for the species.
        """
        r = np.zeros_like(p)
        
        if self.co_adsorption:
            r[..., 0], r[..., 2] = CO2.reaction_rate(p[..., 0], p[..., 2], p[..., 3], self.T_feed)
        else:
            r[..., 0], r[..., 2] = CO2.reaction_rate(p[..., 0], p[..., 2], self.T_feed)
        
        r[..., 1], r[..., 3] = H2O.reaction_rate(p[..., 1], p[..., 3], self.T_feed)
        
        return r

    def print_par(self):
        '''
        Compute and print the non-dimensional adsorption-diffusion equation parameters for CO2.
        '''

        print(f'Characteristic diffusion time CO2 [s]: {self.tau_D[0]:0.5f}')
        print(f'Characteristic gas adsorption time CO2 [s]: {self.kin_CO2.tau_ads_g:0.5f}')
        print(f'Characteristic solid adsorption time CO2 [s]: {self.kin_CO2.tau_ads_s:0.5f}')

    def check_steady_state(self, p_f):            
        contains_none = np.any([item is None for item in self.t_SS])
        if contains_none:
            print("Steady state not reached for every species")
        else:
            print(f'All species reached steady state, with 1e-3 [Pa] CO2 tolerance')
            print(f'Time required to reach CO2 steady state: \nGas phase = {self.t_SS[0]:0.5f} [h] \nSolid phase = {self.t_SS[2]:0.5f} [h]')
            print(f'Referred to a CO2 initial pressure = {self.p_CO2_initial}')

    def binary_diff_coeff(self, MW, sigma, eps_k, T):
        '''
        Compute the binary molecular diffusion coeffients using the Chapman-Enskog model.

        Parameters:
        - MW (list): Molar masses of A and B species
        - sigma (list): Lennard-jones potential parameters for A and B species.
        - eps_k (list): Lennard-jones potential parameters for A and B species.
        - T (numpy.ndarray): Temperature field.

        Return:
        - D_AB: Binary molecular diffusion coefficient.
        '''
        
        # Molar masses
        MW_A = MW[0]*1000           # [g/mol]
        MW_B = MW[1]*1000           # [g/mol]
        MW = (1/MW_A + 1/MW_B)

        # Lennard-Jones Parameters 
        sigma = np.mean(sigma)                        # [A]

        eps_kb = np.sqrt(np.prod(eps_k))              # [K] 

        # Neufield collision integral correlation parameters (1972) [A, B, C, D, E, F, G, H]
        o ={"A": 1.06036,
            "B": 0.15610,
            "C": 0.19300,
            "D": 0.47635,
            "E": 1.03587,
            "F": 1.52996,
            "G": 1.76474,
            "H": 3.89411}

        # Calculate Tstar
        Tstar = T/eps_kb

        # Calculate the collision integral
        omega_AB =  o["A"]/(Tstar**o["B"]) + o["C"]/np.exp(o["D"]*Tstar) + o["E"]/np.exp(o["F"]*Tstar) + o["G"]/np.exp(o["H"]*Tstar)

        # Calculate diffusion coefficient
        D_AB =  0.0018583*np.sqrt(T**3 * MW)/(self.p_tot*1e-5 * sigma**2 * omega_AB)

        return D_AB

    def init_plot(self, p, p_f):
        # Initialize plotting: set consistent styling and thicker lines
        plt.ion()
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 14,
            'font.serif': ['Times New Roman'],
            'axes.linewidth': 1.5,
            'lines.linewidth': 3.0  # Thicker lines
        })

        figure, ax = plt.subplots(1, 3, figsize=(14, 6))
        self.indxs_plot = [0, 2, 1, 3] 

        figure.suptitle("Radial profiles (t = 0 s)", fontsize=18, y=1.05)
        axs = [ax[0], ax[0].twinx(), ax[1], ax[1].twinx(), ax[-1]]

        colors = ['tab:blue', 'tab:orange', 'tab:blue', 'tab:orange', 'tab:green']
        labels = ['p$_{CO2}$', 'q$_{CO2}$', 'p$_{H2O}$', 'q$_{H2O}$', 'p$_{N2}$']
        species = ['CO$_2$', 'CO$_2$', 'H$_2$O', 'H$_2$O', 'N$_2$']
        phases = ['Gas phase [$Pa$]', 'Adsorbed phase [$mol/kg$]', 'Gas phase [$Pa$]', 'Adsorbed phase [$mol/kg$]', 'Gas phase [$Pa$]']
        scales_max = [1.3, 1.5, 1.3, 1.5, 1.2]
        scales_min = [0.7, 0.5, 0.7, 0.5, 0.8]

        line = []
        for i, indx in enumerate(self.indxs_plot):
            line += axs[i].plot(self.r_c * 1000, p[..., indx], label=labels[i], color=colors[i])
            axs[i].set_ylim(scales_min[i] * min([p[0, indx], p_f[..., indx]]), scales_max[i] * max([p[0, indx], p_f[..., indx]]))
            axs[i].set_xlabel("r [$mm$]", fontsize=14)
            axs[i].set_ylabel(f'{species[i]} {phases[i]}', fontsize=14, color=colors[i])
            axs[i].tick_params(axis='y', labelcolor=colors[i])

        if np.all(p[..., 1] < 1e-30) and np.all(p_f[..., 1] < 1e-30):
            axs[2].set_ylim(0, 0)
            axs[3].set_ylim(0, 0)

        N2_end = self.p_tot - np.sum(p_f[..., :self.Nc_diff])
        N2_init = self.p_tot - np.sum(p[0, :self.Nc_diff], axis=0)

        line += axs[-1].plot(1000 * self.r_c, self.p_tot * np.ones(self.Nr) - np.sum(p[:, :self.Nc_diff], axis=1), label=labels[-1], color=colors[-1])
        axs[-1].set_xlabel("r [$mm$]", fontsize=14)
        axs[-1].set_ylabel(f'{species[-1]} Gas phase [$Pa$]', fontsize=14, color=colors[-1])
        axs[-1].set_ylim(scales_min[-1] * min([N2_init, N2_end]), scales_max[-1] * max([N2_init, N2_end]))
        axs[-1].tick_params(axis='y', labelcolor=colors[-1])
        axs[-1].axhline(y=N2_end, color='green', linestyle='--')
        axs[-1].axhline(y=N2_init, color='green', linestyle='--')

        self.line = line
        self.axs = axs
        self.figure = figure

        plt.tight_layout()

    def update_plot(self, p, t):
        self.figure.suptitle(f"Radial profiles (t = {t:.3f}s, T = {self.T_feed - 273.15:.2f}°C)", fontsize=18, y=0.98)

        for i, indx in enumerate(self.indxs_plot):
            self.line[i].set_ydata(p[:, indx])

        self.line[-1].set_ydata(self.p_tot * np.ones(self.Nr) - np.sum(p[:, :self.Nc_diff], axis=1))
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
    
    def plot_uptake(self, p_mean_r, plotCO2ExpData=None, plotH2OExpData=None, plot=True):
        # Averaged Quantities
        q_mean_ads = p_mean_r[self.Nc_diff:]
        q_mean_bulk = (self.eps_p / self.rho_p) * p_mean_r[:self.Nc_diff] / R_gas / self.T_feed
        q_mean_total = q_mean_ads + q_mean_bulk

        if plot:
            plt.rcParams.update({
                'font.family': 'serif',
                'font.size': 14,
                'font.serif': ['Times New Roman'],
                'axes.linewidth': 1.5,
                'lines.linewidth': 3.0  # Thicker lines
            })

            figure, axs = plt.subplots(1, 2, figsize=(14, 6))
            figure.suptitle("Average uptake inside the particle", fontsize=20)
            colors = ['tab:red', 'tab:orange', 'tab:blue']
            labels = ['$CO_2$', '$H_2O$']

            time_hours = np.linspace(0, self.t_end, self.max_steps) / 3600

            axs[0].plot(time_hours, np.sum(q_mean_total, axis=0), label='Total Uptake', color=colors[0])
            axs[0].set_ylabel("Total uptake [$mol/kg$]", fontsize=16)
            axs[0].set_xlabel("Time [h]", fontsize=16)
            axs[0].grid(True)

            axs[1].plot(time_hours, q_mean_total[0], label=labels[0], color=colors[1])
            axs[1].plot(time_hours, q_mean_total[1], label=labels[1], color=colors[2])
            axs[1].set_ylabel("Species uptake [$mol/kg$]", fontsize=16)
            axs[1].set_xlabel("Time [h]", fontsize=16)
            axs[1].legend()
            axs[1].grid(True)

            if plotCO2ExpData:
                data = pd.read_csv(plotCO2ExpData, sep=';', decimal='.')
                axs[1].plot(data['time'].values / 3600, data['q'].values, marker='o', linestyle='none',
                            markerfacecolor='none', markeredgecolor='red', label='CO2 Experimental data')
                axs[1].legend()

            if plotH2OExpData:
                data = pd.read_csv(plotH2OExpData, sep=';', decimal='.')
                axs[1].plot(data['time'].values / 3600, data['q'].values, marker='o', linestyle='none',
                            markerfacecolor='none', markeredgecolor='red', label='H2O Experimental data')
                axs[1].legend()

            plt.tight_layout()
            plt.show(block=True)

        return q_mean_total
        
    def plot_fractional_uptake(self, p_mean_r, plot=True):
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 14,
            'font.serif': ['Times New Roman'],
            'axes.linewidth': 1.5,
            'lines.linewidth': 3.0  # Thicker lines
        })

        q_mean_ads = p_mean_r[self.Nc_diff:]
        c_mean_bulk = p_mean_r[:self.Nc_diff] / self.T_feed / R_gas

        F_initial = self.eps_p * self.c_bulk_initial + self.rho_p * self.q_ads_initial
        F_boundary = self.eps_p * self.c_bulk_R + self.rho_p * self.q_ads_R
        F_t = self.eps_p * c_mean_bulk + self.rho_p * q_mean_ads
        F = (F_t - F_initial.reshape(-1, 1)) / (F_boundary - F_initial).reshape(-1, 1)


        if plot:
            figure, axs = plt.subplots(figsize=(14, 6))
            figure.suptitle("Fractional uptake inside the particle", fontsize=20)
            colors = ['tab:blue', 'tab:orange']
            labels = ['$CO_2$', '$H_2O$']
            time_steps = np.linspace(0, self.t_end, self.max_steps)

            axs.plot(time_steps, F[0], label=labels[0], color=colors[0])
            axs.plot(time_steps, F[1], label=labels[1], color=colors[1])
            axs.set_ylabel("Species Fractional Uptake [-]", fontsize=16)
            axs.set_xlabel("Time [s]", fontsize=16)
            axs.set_ylim((0, 1.2))
            axs.axhline(y=1, linestyle='--')
            axs.legend()
            axs.grid(True)

            plt.tight_layout()
            plt.show(block=True)

        return F
    
    def plot_r_app(self, r_app):
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 14,
            'font.serif': ['Times New Roman'],
            'axes.linewidth': 1.5,
            'lines.linewidth': 3.0  # Thicker lines
        })

        figure, axs = plt.subplots(1, self.Nc_diff, figsize=(14, 6))
        figure.suptitle("Apparent adsorption rate", fontsize=20)
        colors = ['tab:red', 'tab:orange', 'tab:blue']
        labels = ['$CO_2$', '$H_2O$', '$N_2$']

        time_hours = np.linspace(0, self.t_end, self.max_steps) / 3600

        for i in range(self.Nc_diff):
            axs[i].plot(time_hours, r_app[i], label=labels[i], color=colors[i])
            axs[i].set_ylabel(f"{labels[i]} apparent rate of adsorption [mol/kg$_p$/s]", fontsize=16)
            axs[i].set_xlabel("Time [h]", fontsize=16)
            axs[i].legend()
            axs[i].grid(True)

        plt.tight_layout()
        plt.show(block=True)


data_config = {
    'co_adsorption': True,                         # True or False
    'isotherm_CO2': 'Toth',                         # 'Toth', 'SB'
    'Toth_data': 'Young',                             # 'Shi', 'Young, 'Chimani'
    'kinetic_CO2': 'TothRate',                      # 'TothRate', 'LDF_rate'
    'isotherm_H2O': 'GAB',                          # 'GAB' is the only option
    'kinetic_H2O': 'LDF_rate',                      # 'LDF_rate' is the only option
}

plot_config = {
    'update_plot_toggle': True,                     # Update axial/radial profiles in time
    'line_plot_toggle': True,                       # Use 1D plot in 2D simulations: shows axial profiles at r=0 and r=R
    'normalized_concentrations': False,              # normalization with feed conditions (adsorption) or initial conditions (desorption)
    'delta_T_max': 30,                              # factor to scale the temperature y-axis
    'y_max_multiplier': 2,                        # factor to scale the y-axis for clearer rapresentation
    're_ads_fact': 15,                              # factor to scale the axis accounting for re-adsorption effects
    'time_unit': 'hr'                                # factor to convert time of breakthrough plot: 's', 'min', 'hr'
}

# Initialize the column data
config = PackedBedFactory(data_config)

# Initialize therodynamics and kinetic models for CO2
isotherm_CO2 = config.get_isotherm_model('CO2')
kinetic_CO2 = config.get_kinetic_model('CO2', isotherm_CO2)

# Initialize therodynamics and kinetic models for CO2
isotherm_H2O = config.get_isotherm_model('H2O')
kinetic_H2O = config.get_kinetic_model('H2O', isotherm_H2O)


model = ParticleModelFick(init_params=init_params_PB, 
                          kinetics=[kinetic_CO2, kinetic_H2O],
                            thermo=[isotherm_CO2, isotherm_H2O],
                              co_adsorption=False)
# model.__init__(init_params=init_params_PB, 
#                           kinetics=[kinetic_CO2, kinetic_H2O],
#                             thermo=[isotherm_CO2, isotherm_H2O],
#                               co_adsorption=False)
model.solve_transient()