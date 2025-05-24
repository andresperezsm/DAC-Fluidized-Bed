import numpy as np
import matplotlib.pyplot as plt
from pymrm import (newton, construct_div, construct_coefficient_matrix,
                   construct_grad, numjac_local, interp_cntr_to_stagg_tvd, minmod, 
                   upwind, construct_convflux_upwind, clip_approach, non_uniform_grid)
from scipy.constants import gas_constant as R_gas
import scipy.sparse as sps
from scipy.linalg import norm
from scipy.sparse import csc_array
import math
from FluidizationHydrodynamics import Fluidization
import time


class Column:
    """
    Class for simulating adsorption-diffusion-convection systems in a 2D cylindrical geometry with axial and radial components.
    
    This class supports both 1D and 2D simulations depending on the number of radial grid points (Nr).
    The governing equations include components for both gas and solid phases, with customizable boundary conditions,
    initial conditions, and kinetic models.

    """

    def __init__(self, init_params=None, thermo=None, kin=None, model_type=None):
        """
        Initialize the ColumnModel class with specified parameters, thermodynamic models, and kinetic models.

        Parameters:
        - init_params (function, optional): Function to initialize model parameters.
        - thermo (tuple, optional): Thermodynamic models for CO₂ and H₂O (thermo_CO2).
        - kin (tuple, optional): Kinetic models for CO₂ and H₂O (kin_CO2).

        The constructor initializes the default physical properties such as molecular weights, diffusion coefficients, 
        heat capacities, and thermodynamic properties. It also sets up the concentration grid, initializes boundary conditions,
        diffusion coefficients, velocity fields, and Jacobian matrices for the system. Optionally, initial and feed conditions
        can be set via user-provided functions.
        """
        self.model_type = model_type['model']
        self.const_phase_frac = model_type['constant_phase_fractions']	
        self.mass_transf_corr = model_type['mass_transf_corr']	

        # Default parameters (constants)
        self.MW_CO2 = 44.01/1000            # [kg/mol]
        self.MW_H2O = 18.01528/1000         # [kg/mol]
        self.MW_N2 = 28.0134/1000           # [kg/mol]   

        self.sigma_CO2= 3.941             # [Å]
        self.sigma_H2O= 2.641             # [Å]
        self.sigma_N2=  3.798             # [Å]

        self.eps_k_CO2= 195.2             # [K] 
        self.eps_k_H2O= 809.1             # [K] 
        self.eps_k_N2= 71.4               # [K] 

        self.v_Fuller_CO2 = 26.9          # [cm3/mol]
        self.v_Fuller_H2O = 13.1          # [cm3/mol]
        self.v_Fuller_N2 = 18.5           # [cm3/mol]

        self.lambda_g_CO2= 0.0166         # [W/m/K] (298K, 1atm) The Engineering Toolbox
        self.lambda_g_H2O= 0.02457        # [W/m/K] (398K, 1atm) The Engineering Toolbox
        self.lambda_g_N2= 0.02583         # [W/m/K] (298K, 1atm) The Engineering Toolbox
        self.lambda_p =  0.43             # [W/m/K] (Bos 2019)

        self.cp_g_CO2= 0.84*1000          # [J/kg/K] (298K, 1atm) The Engineering Toolbox
        self.cp_g_H2O= 1.864*1000         # [J/kg/K] (298K, 1atm) The Engineering Toolbox
        self.cp_g_N2= 1.04*1000           # [J/kg/K] (298K, 1atm) The Engineering Toolbox  
        
        self.a_cp_p = -3.23e7             # [J/K/kg] (Low 2023)
        self.b_cp_p = 2.27                # [J/kg/K^2] (Low 2023)
        self.c_cp_p = -994                # [J/kg/K] (Low 2023)

        self.DH0_CO2 = - 70125            # [J/mol] (Low 2023)
        self.DH0_H2O =  - 46000           # [J/mol] (Low 2023)

        # Optional parameter initialization
        if init_params is not None:
            init_params(self)

        # Set the thermodynamic and kinetic model
        if thermo is not None:
            self.thermo_CO2 = thermo[0]
            self.thermo_H2O = thermo[1]

        if kin is not None:
            self.kin_CO2 = kin[0]
            self.kin_H2O = kin[1]

        if self.model_type == 'Two-Phase Model':
            self.Nc = 4                              # [CO2-solid]
        elif self.model_type == 'Three-Phase Model':
            raise ValueError('The Three-Phase model is still in development.')
        
        # Shape of the concentration field
        self.shape_c = (self.Nz, self.Nr, 4)

        # Other parameters
        self.d_p = self.R_p * 2
        self.rho_eps_p = self.rho_p / self.eps_p 
        self.inv_eps_p = 1/self.eps_p
        self.a_v_p = 3/self.R_p

        self.a_v_b = 4/(2*self.R_b)
        self.eps_fs = (1 - self.eps_b)/self.eps_b 
        self.eps_fs_rho_p = self.rho_p * (1 - self.eps_b)/self.eps_b 

        self.p_tot = self.p_tot_feed
        self.c_tot_feed = self.p_tot / R_gas / self.T_feed

        self.v_int = self.v_inlet / self.eps_b 
        self.v_sup = self.v_inlet

        self.A_inlet = (self.R_b ** 2) * np.pi
                                                       
        # Set initial conditions
        self.set_ic()

        # Set the feed conditions
        self.set_feed()

        # Initialize gas mixture properties
        self.init_gas_mix()

        # Calculate effective diffusion in the particle
        self.calc_diff()

        # Calcualte the MT Coeffs and Phase Fractions for FBR case
        self.fluidized_hydrodynamics()

        # Setup axial and radial grids after fluidization
        self.z_f = np.linspace(0, self.L_b, self.shape_c[0] + 1)
        self.z_c = 0.5 * (self.z_f[:-1] + self.z_f[1:])
        self.r_f = np.linspace(0, self.R_b, self.shape_c[1] + 1)
        self.r_c = 0.5 * (self.r_f[:-1] + self.r_f[1:])

        # Initialize Jacobian matrices for the system
        self.init_Jac()

    def set_ic(self):
        """
        Set the initial conditions for the concentration field.

        The initial conditions are set for each species in both the gas and solid phases.
        The method calculates the initial concentrations based on the initial pressures, 
        relative humidity, and temperature.

        The concentration field (`c`) is initialized as a 3D array with dimensions corresponding to 
        (axial grid points, radial grid points, species).
        """
        if self.model_type == 'Two-Phase Model':
            self.c = np.full(self.shape_c, 0, dtype='float')         
        elif self.model_type == 'Three-Phase Model':
            self.c = np.full(self.shape_c, 0, dtype='float')

    def set_feed(self):
        """
        Set the feed concentrations for the system based on the initial and feed conditions.

        This method calculates and sets the feed concentrations for the CO₂, H₂O, and N₂ species in both the 
        gas and solid phases. It uses the initial temperature and pressure conditions, along with the relative 
        humidity and thermodynamic models for CO₂ and H₂O.

        The concentration field (`c_feed`) is initialized as a 2D array with dimensions corresponding to 
        (1, species), where species include CO₂, H₂O, and N₂.
        """

        self.c_feed = np.zeros((1, self.Nc))

        c_CO2_feed = self.p_CO2_feed / R_gas / self.T_feed + 1e-32 + 1e-32
        q_CO2_feed = self.thermo_CO2.isotherm(self.p_CO2_feed, self.T_feed)[0] + 1e-32

        self.c_feed[..., 0] = c_CO2_feed
        self.c_feed[..., 2] = c_CO2_feed

        self.y_feed = np.zeros((1, 2))
        self.y_feed[..., 0] = self.c_feed[..., 0] / self.c_tot_feed
        self.y_feed[..., 1] = 1 - self.y_feed[..., 0]

        self.c_feed_plot = self.c_feed.copy()
        self.c_feed_plot[..., 1] = q_CO2_feed
        self.c_feed_plot[..., 3] = q_CO2_feed

    def init_gas_mix(self):
        """
        Initialize all gas mixture properties including density, specific heat capacity, 
        thermal conductivity, and viscosity.
        """
        print("\nInitializing gas mixture properties...")

        self.init_rho_mix()
        self.init_lambda_mix()
        self.init_visc_mix()

    def init_rho_mix(self):
        """
        Calculate and initialize the density of the gas mixture at feed conditions.
        This is based on the ideal gas law.
        """
        # Molecular weights [kg/mol]
        MW = np.array([self.MW_CO2, self.MW_N2])
        
        # Calculate the density [kg/m³]
        self.rho_g_mix = self.p_tot_feed * np.dot(self.y_feed, MW) / (R_gas * self.T_feed)
        
        print(f"Mixture gas density at T={self.T_feed}K: {self.rho_g_mix.item():.6f} [kg/m³]")

    def init_lambda_mix(self):
        """
        Calculate and initialize the thermal conductivity of the gas mixture at feed conditions.
        The thermal conductivity is calculated using mole fractions.
        """
        # Thermal conductivity [W/m/K]
        lambda_g = np.array([self.lambda_g_CO2, self.lambda_g_N2])
        
        # Calculate the average thermal conductivity [W/m/K] using mole fractions
        self.lambda_g_mix = np.dot(self.y_feed, lambda_g)
        
        print(f"Mixture gas thermal conductivity at T={self.T_feed}K: {self.lambda_g_mix.item():.6f} [W/m/K]")

    def init_visc_mix(self):
        """
        Calculate and initialize the viscosity of the gas mixture at feed conditions.
        This is done using the Wilke mixture rule based on Chapman-Enskog theory.
        """
        # Molecular weights [kg/mol], Lennard-Jones parameters
        MW = np.array([self.MW_CO2, self.MW_N2])
        eps_k = np.array([self.eps_k_CO2, self.eps_k_N2])
        
        # Calculate pure component viscosities [Pa·s]
        self.mu_i = 1.109e-6 * np.sqrt(self.T_feed) / self.collision_integral(np.array(eps_k), self.T_feed)
        
        # Wilke's method for mixture viscosity
        Mi_Mj = MW.reshape(-1, 1) / MW
        mui_muj = self.mu_i.reshape(-1, 1) / self.mu_i
        phi_ij = (8)**(-0.5) * (1 + Mi_Mj)**(-0.5) * (1 + mui_muj**0.5 * Mi_Mj**(-0.25))**2
        
        # Calculate mixture viscosity [Pa·s]
        self.mu_g_mix = np.sum((self.y_feed * self.mu_i)/([np.sum(yi * phi_ij, axis=1) for yi in self.y_feed]), axis=1).reshape(-1,1).item()        

        print(
            f"Viscosities at T={self.T_feed} K: CO₂ = {self.mu_i[0]:.2E} Pa·s, ")
        
        print(f"Mixture viscosity at T={self.T_feed}K: {self.mu_g_mix:.2E} Pa·s")

    def calc_diff(self):
        print("\nInitializing diffusion coefficients...")

        # Knudsen diffusion
        D_kn = 2/3 * self.r_pore * np.sqrt(8 * R_gas * self.T_feed/ np.array([self.MW_CO2,]) / np.pi)
        print(f'Knudsen diffusion coefficients m2/s: CO2 {D_kn[0]:0.2E}')   

        # Molecular diffusion in nitrogen (excess)
        binary_par_1n = [[self.MW_CO2, self.MW_N2], [self.v_Fuller_CO2, self.v_Fuller_N2]]
        D_1n0 = self.Fuller_binary_diff(*binary_par_1n, self.T_initial) / 1e4
        self.D_m = np.array([D_1n0]) 

        print(f'Molecular diffusion coefficients (Patm) m²/s: CO₂-N₂ {D_1n0:.2E}')

        # Effective particle diffusion coefficient of CO2 in Nitrogen excess (Bosanquet)
        self.D_eff = 1/self.tau_p * (1/D_kn[0] + 1/D_1n0)**(-1)

        print(f'Effective diluted particle diffusion coefficients m2/s: CO2 {self.D_eff.item():0.2E}')

    def fluidized_hydrodynamics(self):
        """
        Initializes phase fractions and mass transfer coefficients for a fluidized bed.
        
        Calculates the final bed height, `L_final`, and sets parameters based on Geldart 
        classification (A or B) for bubble, cloud-wake, and emulsion phases.
        """
        
        self.fbr = Fluidization(self)
        self.ub = self.fbr.u_b_avg
        self.u_ge = self.fbr.u_ge
        self.u_se = self.fbr.u_se
        self.L_b = self.fbr.H_f

        # Phase fractions per unit of reactor volume
        self.f_b = self.fbr.f_b
        self.f_w = self.fbr.f_w
        self.f_e = self.fbr.f_e

        # Gas and solid fraction per unit of reactor volume (per each phase)
        self.gamma_bw_gas = self.fbr.gamma_bw_gas                   
        self.gamma_e_gas = self.fbr.gamma_e_gas                    
        self.gamma_w_solid = self.fbr.gamma_w_solid  
        self.gamma_e_solid = self.fbr.gamma_e_solid

        self.bw_kin_par = self.rho_eps_p * self.gamma_w_solid / self.gamma_bw_gas
        self.e_kin_par = self.rho_eps_p * self.gamma_e_solid / self.gamma_e_gas

        # Mass transfer coefficeints
        self.K_we =  0.001*self.fbr.K_we
        self.K_be = self.fbr.K_be

    def init_Jac(self):
        """
        Initialize the Jacobian matrices for the system.
        """

        # Boundary conditions: Neumann
        self.bc =     { 'a': [ [[[0, 0, 0, 1]]],  [[[1, 1, 1, 0]]] ], 
                        'b': [ [[[1, 0, 1, 0]]],  [[[0, 0, 0, 0]]] ],
                        'd': [ [self.c_feed.tolist()],     [[[0, 0, 0, 0]]] ]}
        
        # Boundary conditions: Dirichlet (for the coupling)
        self.bc_w_in = { 'a': [ [[[0, 0, 0, 0]]],   [[[0, 0, 0, 0]]] ], 
                         'b': [ [[[0, 1, 0, 0]]],   [[[0, 0, 0, 0]]] ],
                         'd': [ [[[0, 1, 0, 0]]],   [[[0, 0, 0, 0]]] ]}  
        
        self.bc_e_in = { 'a': [ [[[0, 0, 0, 0]]],   [[[0, 0, 0, 0]]] ], 
                         'b': [ [[[0, 0, 0, 0]]],   [[[0, 0, 0, 1]]] ],
                         'd': [ [[[0, 0, 0, 0]]],   [[[0, 0, 0, 1]]] ]}
        
        # Accumulation term
        self.Jac_accum = construct_coefficient_matrix(1/self.dt, self.shape_c)

        # Divergence term
        self.Div_ax = construct_div(self.shape_c, self.z_f, nu=0, axis=0)
        
        # Construct the bw-e linking matrix
        self.size_c_f = math.prod(self.shape_c[1:])
        self.indx_j = np.arange(self.size_c_f)
        self.indx_i_out = self.size_c_f * self.shape_c[0] + self.indx_j
        self.indx_i_in = self.indx_j
        
        self.v_conv= [[[self.ub.item(), self.ub.item(), self.u_ge.item(), self.u_se.item()]]]
            
        Conv, conv_bc = construct_convflux_upwind(self.c.shape, self.z_f, self.z_c, self.bc, self.v_conv, axis=0)
        _, conv_solid_in_e = construct_convflux_upwind(self.c.shape, self.z_f, self.z_c, self.bc_e_in, self.v_conv, axis=0)
        _, conv_solid_in_w = construct_convflux_upwind(self.c.shape, self.z_f, self.z_c, self.bc_w_in, self.v_conv, axis=0)

        self.Flux_ax = Conv
        self.flux_ax_bc = conv_bc
        self.flux_solid_in_e = conv_solid_in_e
        self.flux_solid_in_w = conv_solid_in_w

        self.g_const = self.Div_ax @ (self.flux_ax_bc) #+  self.flux_diff_ax_bc)
        self.Jac_const = self.Jac_accum + self.Div_ax @ (self.Flux_ax)# + self.Flux_diff_ax)

        conv_bc_solid_in_e = csc_array((np.asarray(self.flux_solid_in_e[self.indx_i_out, [0]]), (self.indx_i_out, self.indx_j)),
                                        shape=(self.size_c_f * (self.shape_c[0] + 1), self.size_c_f))  
        conv_bc_solid_in_w = csc_array((np.asarray(self.flux_solid_in_w[self.indx_i_in, [0]]), (self.indx_i_in, self.indx_j)),
                                        shape=(self.size_c_f * (self.shape_c[0] + 1), self.size_c_f))   
        
        self.Jac_we = self.Div_ax @ conv_bc_solid_in_e
        self.Jac_ew = self.Div_ax @ conv_bc_solid_in_w

    def kin(self, c):
        """
        Calculate the reaction rates for CO₂  in both gas and solid phases.

        """
        r = np.zeros_like(c)

        ads_rate_w = self.kin_CO2.ads_rate(c[..., 0], c[..., 1], self.T_feed)
        ads_rate_e = self.kin_CO2.ads_rate(c[..., 2], c[..., 3], self.T_feed)

        r[..., 0] = - ads_rate_w * self.bw_kin_par - self.K_be[0] * (c[..., 0]  - c[..., 2] * self.gamma_e_gas / self.gamma_bw_gas) 
        r[..., 1] = + ads_rate_w - self.K_we[0] * (c[..., 1] - c[..., 3] * self.gamma_e_solid / self.gamma_w_solid)
        r[..., 2] = - ads_rate_e * self.e_kin_par + self.K_be[0] * (c[..., 0] * self.gamma_bw_gas / self.gamma_e_gas  - c[..., 2]) 
        r[..., 3] = + ads_rate_e + self.K_we[0] * (c[..., 1] * self.gamma_w_solid / self.gamma_e_solid - c[..., 3])

        return r
    
    def construct_g(self, c, c_old, q_mix_top, q_mix_bottom):
        """
        Construct the residual vector g and the Jacobian matrix for the system.
        """

        c_f, dc_f = interp_cntr_to_stagg_tvd(c, self.z_f, self.z_c, self.bc, self.v_conv, minmod, axis=0)
        dg_conv = self.Div_ax @ (self.v_conv * dc_f).reshape((-1, 1))
        g_react, Jac_react = numjac_local(lambda c_var: self.kin(c_var), c)

        g = (self.g_const + (self.Jac_const) @ c.reshape((-1, 1)) + dg_conv + 
            - self.Jac_accum @ c_old.reshape((-1, 1))  
            + self.Jac_we @ q_mix_top.reshape((-1, 1))  
            + self.Jac_ew @ q_mix_bottom.reshape((-1, 1))
            - g_react.reshape((-1, 1))) 

        Jac = self.Jac_const - Jac_react
        
        return g, Jac

    def solve_transient_coupled(self):
        """
        Solve the system for a specified number of time steps.
        """
        print(f"PDEs system: start solving")
  
        self.max_steps = int(self.t_end / self.dt) + 1

        # Initialize storage for outlet concentration
        self.c_out_bw = np.zeros((self.max_steps, 1))                  
        self.c_out_e = np.zeros((self.max_steps, 1))                        
        self.c_out = np.zeros((self.max_steps, 1))            

        c_iter = self.c.copy()
        self.q_mix_top = np.zeros((1, 1, 4))
        self.q_mix_bottom = np.zeros((1, 1, 4))

        start_time = time.time()

        for t_indx, t in enumerate(np.linspace(0, self.t_end, self.max_steps)):
            print(f"\ntime: {t:.5f}")

            c_old = self.c.copy()
            result = newton(
                lambda c_var: self.construct_g(c_var, c_old, self.q_mix_top, self.q_mix_bottom),
                c_old,
                maxfev=self.maxfev,
                tol=self.tol,
                callback=lambda x, g: self.update_solid_BC(x, g)
            )  

            self.c = np.clip(result.x, 0, None)  
            
            self.c_out[t_indx] = self.c[-1, 0, 2] * self.gamma_w_solid + self.c[-1, 0, 3] * self.gamma_e_solid

            print(f"Solid entering the emulsion: {self.q_mix_top[..., -1].item()} [mol/kg]")
            print(f"Solid entering the wake: {self.q_mix_bottom[..., 1].item()} [mol/kg]")

            if self.outer_message:
                print(result.message)
                
            # Update plotting 
            if self.update_plot_toggle:
                self.update_solid_plot(self.c, t)      

        self.plot_breakthrough(self.c_out)
        self.mass_conservation(self.c)

        end_time = time.time()
        print(f"PDEs system: end solving in {(end_time - start_time):0.2f} s\n")
        plt.show(block=True)

    def plot_breakthrough(self, c_out):
        plt.figure(4)
        time_steps = np.linspace(0, self.t_end, self.max_steps)
        self.time_scale = 60 * 60 # Time scale in hours
        plt.plot(time_steps / self.time_scale, self.c_out , label = 'Fluidized') # / self.c_tot_feed)
        # c_out_PBR = 1
        # plt.plot(time_steps / self.time_scale, c_out_PBR, label = 'Packed')
        plt.legend()
        plt.grid()
        plt.xlabel('Time [hr]')
        plt.ylabel('Concentration [-]')

    def mass_conservation(self, c):
        plt.figure(3)
        plt.plot(self.z_c, self.c[:, 0, 0] * self.gamma_bw_gas, label = 'Gas Bubble')
        plt.plot(self.z_c, self.c[:, 0, 1] * self.gamma_w_solid, label = 'Solid Wake')
        plt.plot(self.z_c, self.c[:, 0, 2] * self.gamma_e_gas, label = 'Gas Emulsion')
        plt.plot(self.z_c, self.c[:, 0, 3] * self.gamma_e_solid, label = 'Solid Emulsion')
        ctot = (self.c[:, 0, 0]* self.gamma_bw_gas + self.c[:, 0, 1] * self.gamma_w_solid + self.c[:, 0, 2] * self.gamma_e_gas + self.c[:, 0, 3] * self.gamma_e_solid)
        
        plt.plot(self.z_c, ctot, label = 'Total Concentration')
        # plt.axhline((self.c_tot_feed))
        plt.grid()
        plt.xlabel('Reactor Length [m]')
        plt.ylabel('Concentration [-]')
        plt.legend()

    def update_solid_BC(self, c, g=None):
        flux_w_out = c[-1, 0, 1] * self.ub * self.gamma_w_solid * self.rho_p           # [mol/m_R²/s]
        self.q_mix_top[..., 3] = - flux_w_out / (self.u_se * self.gamma_e_solid * self.rho_p)         
        flux_e_out = c[0, 0, -1] * self.u_se * self.gamma_e_solid * self.rho_p           # [mol/m_R²/s]
        self.q_mix_bottom[..., 1] = - flux_e_out / (self.ub * self.gamma_w_solid * self.rho_p) 

    def calc_adim_numbs(self):
        """
        Calculate adimensional numbers (Schmidt, Reynolds, and Sherwood numbers).

        These numbers are used to quantify transport phenomena in the system. The Schmidt number 
        depends on gas diffusivity, the Reynolds number on the gas flow, and the Sherwood number 
        is a function of the Reynolds and Schmidt numbers.
        """
        print("\nInitializing adimentional numbers...")
        
        self.Sc = self.mu_g_mix / self.D_m / self.rho_g_mix
        self.Pr = self.mu_g_mix * self.cp_g_mix / self.lambda_g_mix
        self.Re = self.rho_g_mix * self.v_int * (2*self.R_p) / self.mu_g_mix  
        
        self.Sh = 2 + 1.1 * self.Re**(0.6) * self.Sc**(1/3)                                     # Shi et al 2024
        
        self.Nu_overall = 0.813*self.Re*(0.9)*np.exp(-6*(self.R_p/self.R_b))                    # Shi et al 2024

        print(f'Reynolds number: {self.Re.item():.2f}')
        print(f'Schmidt number for CO2: {self.Sc[0].item():.2f}, H2O: {self.Sc[1].item():.2f}')
        print(f'Prandtl number: {self.Pr.item():.2f}')
        print(f'Sherwood number for CO2: {self.Sh[0].item():.2f}, H2O: {self.Sh[1].item():.2f}')
        print(f'Nusselt number: {self.Nu_overall.item():.2f}')

    def Fuller_binary_diff(self, MW, v, T):
        '''
        Compute the binary molecular diffusion coeffients using the Fuller model.

        Parameters:
        - MW (list): Molar masses of A and B species
        - T (numpy.ndarray): Temperature field.

        Return:
        - D_AB: Binary molecular diffusion coefficient.
        '''
        
        # Molar masses [g/mol]
        MW_A = MW[0]*1000        
        MW_B = MW[1]*1000          
        MW = (1/MW_A + 1/MW_B)

        # Diffusion volumes cm3 /mol
        v_A = v[0]
        v_B = v[1]

        # Calculate diffusion coefficient
        D_AB =  0.00143 * T**(1.75) * np.sqrt(MW) / (self.p_tot/1e5 * (v_A**(1/3) + v_B**(1/3))**2)

        return D_AB
    
    def set_plotter(self, plot):
        plot.column_model = self            # Ensure PlotClass can access column instance
        self.update_plot_toggle = False

        if plot.update_plot_toggle:
            self.update_plot_toggle = True
            
            if self.model_type == 'Two-Phase Model':        
                if plot.normalized_concentrations:
                    plot.init_solid_circulation_plot()              
                    self.update_solid_plot = plot.update_solid_circulation_plot    
                else:
                    plot.init_solid_circulation_plot()
                    self.update_solid_plot = plot.update_solid_circulation_plot

    def collision_integral(self, eps_k, T):

        # Neufield collision integral correlation parameters (1972) [A, B, C, D, E, F, G, H]
        o ={"A": 1.16145,
            "B": 0.14874,
            "C": 0.52487,
            "D": 0.77320,
            "E": 2.16178,
            "F": 2.43787}

        # Calculate Tstar
        Tstar = T/eps_k

        # Calculate the collision integral
        omega =  o["A"]/(Tstar**o["B"]) + o["C"]/np.exp(o["D"]*Tstar) + o["E"]/np.exp(o["F"]*Tstar) 

        return omega
           