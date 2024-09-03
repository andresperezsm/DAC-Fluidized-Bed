import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.optimize as sopt
import pymrm as mrm
import time

from particle_model_RWGS import *
from FluidizationHydrodynamics import *

class geldart_B_RWGS: 
    def __init__(self,Nodes, d_p, Lr, dt, gas_velocity, X_CO2_in, T_in, P_in, Maxwell_Stefan=False, c_dependent_diffusion=False):
        # Inlet conditions:
        self.T_in = T_in # Inlet temperature [K]
        self.P_in = P_in # Inlet pressure [Pa]
        self.R_gas = 8.314 # Gas constant [J/mol/K]

        # Initial concentrations validation case
        phi_CH4 = 373.4/3600 # (mol/s)
        phi_H2O = 2*phi_CH4
        self.inventory_cat = 0.592 # Catalyst inventory to obtain the reactor length
        u_mf = 0.014 # Try d_p = 2e-4 and rho_p = 2000 in fluidization_hydrodynamics

        self.X_CO2_in = X_CO2_in # X_CO2_in # Bulk CO2 mole fraction [-]
        self.X_CH4_in = 0 #(1 - self.X_CO2_in)*(phi_CH4/(phi_CH4 + phi_H2O)) # Bulk CH4 mole fraction [-]
        self.X_H2O_in = (1 - self.X_CO2_in)*(1 - self.X_CH4_in) # 1 - 0.5 # Bulk H2 mole fraction [-]

        """ THE FOLLOWING FRACTIONS ARE NOT EQUAL TO 1"""
        print(self.X_CH4_in,self.X_CO2_in,self.X_H2O_in)

        self.C_CO2_in = self.P_in*self.X_CO2_in/(self.R_gas*self.T_in) # Bulk CO2 concentration [mol/m3]
        self.C_H2O_in = self.P_in*self.X_H2O_in/(self.R_gas*self.T_in) # Bulk H2 concentration [mol/m3]
        self.C_CH4_in = self.P_in*self.X_CH4_in/(self.R_gas*self.T_in) # Bulk CH4 concentration [mol/m3]

        self.C_tot_in = self.C_CH4_in+self.C_H2O_in # Total inlet concentration [mol/m3]

        self.Maxwell_Stefan = Maxwell_Stefan
        self.c_dependent_diffusion = c_dependent_diffusion

        # Simulation and field parameters
        self.Nz = Nodes # Number of grid cells in the axial direction [-]
        self.dt = dt # Lengt of a time step [-]
        
        self.Nc = 5 # Number of components [-]

        """ CHANGE THIS TO 3"""
        self.Nph = 3 # Number of phases
        
        self.L_R = Lr # Reactor length [m]
        self.D_R = 0.5 # Reactor diameter [m]
        
        self.t_end = 10 # 10*self.L_R/gas_velocity # factor of 1.5 was used before
        
        if dt == np.inf:
            self.Nt = 1
        else:
            self.Nt = int(self.t_end/self.dt)
        
        self.dz = self.L_R/self.Nz # Width of a grid cell [m]
        
        self.z_f = np.linspace(0,self.L_R,self.Nz+1) # Axial coordinates of the faces of the grid cells
        self.z_c = 0.5 * (self.z_f[1:] + self.z_f[:-1]) # Axial coordinates of the centers of the grid cells

        # Gas constants
        self.rho_g = 1.225 # Density of air [kg/m3]
        self.eta_g = 4e-5 # Viscosity of air [Pas s]
        self.Cp_g = 5e3 # Molar heat capacity of air [J/kg/K]
        self.lam_g = 0.03 # Thermal conductivity of air [W/m/K]
        self.P_in = P_in # Reactor pressure [Pa]
        self.eps_b = 0.4 # Bed porosity [-]
        self.u_g = gas_velocity # Velocity of the gas [m/s]

        # Heat capacity constants
        self.Cp_CO2 = 1.2e3 # Heat capacity of CO2 [J/kg/K]
        self.Cp_H2 = 15e3 # Heat capacity of H2 [J/kg/K]
        self.Cp_CO = 1.1e3 # Heat capacity of CO [J/kg/K]
        self.Cp_H2O = 2e3 # Heat capcity of water [J/kg/K]
        self.Cp_CH4 = (69.14/16)*1000 # # Heat capcity of Methane [J/kg/K]
        self.Cp_g = self.X_CH4_in*self.Cp_CH4 + self.X_H2O_in*self.Cp_H2O # Mean heat capcity of the feed gas [J/kg/K]
        
        # Catalyst parameters: Validation case
        self.lam_s = 70 # Thermal diffusivity NEEDED
        self.rho_s = 2000  # Solid density
        self.d_p = d_p # Diameter of the catalyst particle [m]
        self.Cp_s = 880 # Al2O3 support heat capacity # Lewatit ((-3.23e4)/(self.T_in**2) + (0.00227)*(self.T_in) - (-0.994))*(10**3) # Heat capacity 
        self.Dax_s = 0.0
        self.eps_b = 0.4438 # Packing in the bed at minimum fluidization -> should be epsilon_mf from FluidizationHydrodynamics
        self.eps_s = 0.338 # Porosity of the Lewatit particle
        self.eps_mf = 0.4438
        self.tauw = np.sqrt(2)
        self.d_pore = 25e-9 # Average pore diameter of the Lewatit particle 
        
        # Reaction related parameters:
        self.H_r = 0 # 42e3 # Heat of the reaction in [J/mol]
        self.T_in = T_in # Inlet temperature [K]
        
        # Maxwell-Stefan related parameters:
        self.M_CO2 = 44 # Molecular weight of CO2 [g/mol]
        self.M_H2 = 2 # Molecular weight of H2 [g/mol]
        self.M_CO = 28 # Molecular weight of CO [g/mol]
        self.M_H2O = 48 # Molecular weight of H2O [g/mol]
        self.M_CH4  = 16 # Molecular weight of CH4 [g/mol]

        self.V_CO2 = 26.7 # Diffusion volume of CO2 [m3]
        self.V_H2 = 6.12 # Diffusion volume of H2 [m3]
        self.V_CO = 18.0 # Diffusion volume of CO [m3]
        self.V_H2O = 13.1 # Diffusion volume of H2O [m3]
        self.V_CH4 = 2*self.V_H2 + 15.9 # Diffusion volume of CH4 [m3]

        self.D_CO2, self.D_H2, self.D_CO, self.D_H2O, self.D_CH4 = self.calculate_average_diffusion_coefficients() # Diffusion coefficients [m2/s]
        self.D_eff_p_avg = np.array(self.calculate_average_diffusion_coefficients()) # Effective diffusion coefficients [m2/s]
        
        self.km_12, self.km_13, self.km_14, self.km_23, self.km_24, self.km_34 = self.k_ms()


        # Fluidization hydrodynamics
        model = FluidizationHydrodynamics()
        model.init(d_p, gas_velocity, Nodes)
        z = model.z

        # Phase fractions varying along the axial direction
        self.epsilon_cw = model.epsilon_cloud_wake
        self.epsilon_bubble = model.fb
        self.epsilon_emulsion = model.epsilon_emulsion

        # Phase velocities as arrays and minimum fludization velocity
        self.umf = model.u_mf
        self.ub = model.u_b
        #ub_second_option = model.u_b_1
        self.ue = model.u_emulsion
        
        # Gas to solid mass transfer coefficients
        self.kgs = model.kgs

        # kgas[Reactor Position, Coefficient type (Kbc, Kce, Kbe, Kov),Component Index]
        if model.particletype == 'Particle is Geldart A':
            # Bubble to cloud transfer coefficients
            self.Kbc_CO2 = model.kgas[0,:,0]
            self.Kbc_CO = model.kgas[0,:,1]
            self.Kbc_H2 = model.kgas[0,:,2]
            self.Kbc_H2O = model.kgas[0,:,3]

            # Cloud to emulsion transfer coefficients
            self.Kce_CO2 = model.kgas[1,:,0]
            self.Kce_CO= model.kgas[1,:,1]
            self.Kce_H2 = model.kgas[1,:,2]
            self.Kce_H2O = model.kgas[1,:,3]

        elif model.particletype == 'Particle is Geldart B':
            # Bubble to cloud transfer coefficients
            self.Kbe_CO2 = model.kgas[2,:,0]
            self.Kbe_CO = model.kgas[2,:,1]
            self.Kbe_H2 = model.kgas[2,:,2]
            self.Kbe_H2O = model.kgas[2,:,3]

        # Bubble diameter profile along the column
        self.db = model.d_b

        # Dimensionless numbers:
        self.Re = self.rho_g*self.u_g*self.d_p/self.eta_g # Reynolds number [-]
        self.Pr = self.Cp_g*self.eta_g/self.lam_g # Prandtl number [-]
        self.Sc_CO2 = self.eta_g / (self.rho_g * self.D_CO2) # Schmidt number CO2 [-]
        self.Sc_H2 = self.eta_g / (self.rho_g * self.D_H2) # Schmidt number H2 [-]
        self.Sc_CO = self.eta_g / (self.rho_g * self.D_CO) # Schmidt number CO [-]
        self.Sc_H2O = self.eta_g / (self.rho_g * self.D_H2O) # Schmidt number H2O [-]
        self.Sc_CH4 = self.eta_g / (self.rho_g * self.D_CH4) # Schmidt number CH4 [-]

        # Correlations:
        self.D_ax_g = np.mean([self.D_CO2, self.D_H2, self.D_CO, self.D_H2O])/np.sqrt(2) + 0.5*self.u_g*self.d_p # Dispersion coefficient gas [m2/s]

        self.A = self.lam_s/self.lam_g 
        self.B = 1.25*((1-self.eps_b)/self.eps_b)**(10/9)
        self.gamma = 2/(1-self.B/self.A)*((self.A-1)/(1-self.B/self.A)**2*self.B/self.A*(np.log(self.A/self.B))-0.5*(self.B+1))
        self.lam_stat = self.lam_g*((1-np.sqrt(1-self.eps_b))+np.sqrt(1-self.eps_b)*self.gamma)
        
        self.D_Thermal = self.lam_g/self.rho_g/self.Cp_g # Thermal diffusivity of air [m2/s]
        self.Dax_Thermal = self.D_Thermal*0.5*self.Re*self.Pr # Axial dispersion coefficient temperature [m2/s]
        self.h_w = self.lam_g/self.d_p*(1.3+5/self.D_R/self.d_p)*self.lam_stat/self.lam_s + 0.19*self.Re**(0.75)*self.Pr**(1/3) # Bed to wall heat transfer [?]
        
        # Gas solid mass and heat transfer coefficient
        self.k_gs_CH4 = self.calculate_mass_transfer_coefficient(self.D_CH4, self.Sc_CH4)
        self.k_gs_H2O = self.calculate_mass_transfer_coefficient(self.D_H2O, self.Sc_H2O)
        self.k_gs_H2 = self.calculate_mass_transfer_coefficient(self.D_H2, self.Sc_H2)
        self.k_gs_CO2 = self.calculate_mass_transfer_coefficient(self.D_CO2, self.Sc_CO2)
        self.k_gs_CO = self.calculate_mass_transfer_coefficient(self.D_CO, self.Sc_CO)
        
        # Fluidized bed mass transfer coefficients
        self.k_bc_CH4, self.k_ce_CH4, self.D_sv_CH4 = self.calculate_mass_transfer_fluidized(self.D_CH4, self.Sc_CH4)
        self.k_bc_H2O, self.k_ce_H2O, self.D_sv_H2O = self.calculate_mass_transfer_fluidized(self.D_H2O, self.Sc_H2O)
        self.k_bc_H2, self.k_ce_H2, self.D_sv_H2 = self.calculate_mass_transfer_fluidized(self.D_H2, self.Sc_H2)
        self.k_bc_CO2, self.k_ce_CO2, self.D_sv_CO2 = self.calculate_mass_transfer_fluidized(self.D_CO2, self.Sc_CO2)
        self.k_bc_CO, self.k_ce_CO, self.D_sv_CO = self.calculate_mass_transfer_fluidized(self.D_CO, self.Sc_CO)
        
        self.h_gs = self.lam_g/self.d_p*((7-10*self.eps_b+5*self.eps_b**2)*(1+0.7*self.Re**0.2*self.Pr**0.33)+(1.33-2.4*self.eps_b+1.2*self.eps_b**2)*self.Re**0.7*self.Pr**0.33) # Gunn correlations
        
        self.k_gs = np.array([self.k_gs_CO2, self.k_gs_H2, self.k_gs_CO, self.k_gs_H2O, self.h_gs])

        self.u_s = 1 # Solid velocities
        self.vel = np.concatenate([np.ones(self.Nc)*self.u_g, np.ones(self.Nc)*self.u_mf, np.ones(self.Nc)*self.u_s,[1]])

        if dt == np.inf: 
            # Initial guess (steady state model)
            self.c_0 = np.zeros(self.Nc*self.Nph+1)
            self.c_0[-1] = self.P_in

        else: 
            # Initial conditions (transient model)
            self.c_0 =   np.zeros(self.Nc*self.Nph+1) #np.ones(self.Nc*self.Nph+1)*1e-3
            self.c_0[-1] = P_in
        
        # Boundary conditions
        self.c_in = np.zeros(self.Nc*self.Nph + 1) # Initial conditions field
        
        # c[3] and c[8] are the initial hydrogen concentrations taken as self.C_H2O_in
        self.c_in[0], self.c_in[2], self.c_in[3] = self.C_CH4_in, self.C_H2O_in, self.C_CO2_in
        self.c_in[-1] = self.P_in
        
        # self.Dax = np.array([self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, 0, 0, 0, 0, 0, 0], dtype = 'float')
        self.Dax = np.zeros(self.Nc*self.Nph + 1)
        self.Dax[10], self.Dax[11], self.Dax[12], self.Dax[13], self.Dax[14] = self.D_sv_CH4, self.D_sv_H2O, self.D_sv_H2, self.D_sv_CO2, self.D_sv_CO

        self.bc_ax = {
                    'a': [[[self.Dax]], 1], # Dirichlet boundary conditions
                    'b': [[[self.vel]], 0], # Neumann boundary conditions
                    'd': [[[self.vel*self.c_in]] , 0.0], # Values
                     }

        # Functions
        self.init_field() # Calls the function in the initial call
        self.init_Jac() # Calls the function in the initial call

    def Fuller_correlation(self, M_i, M_j, V_i, V_j):
        C = 1.013e-2
        D_ij = C*self.T_in**1.75/self.P_in * np.sqrt(1/M_i + 1/M_j) / (V_i**(1/3) + V_j**(1/3))**2
        return D_ij
    
    def k_ms(self, correlation='gunn'):
        pairs = [(self.M_CO2, self.M_H2, self.V_CO2, self.V_H2),    # 12
                    (self.M_CO2, self.M_CO, self.V_CO2, self.V_CO),    # 13
                    (self.M_CO2, self.M_H2O, self.V_CO2, self.V_H2O),  # 14
                    (self.M_H2, self.M_CO, self.V_H2, self.V_CO),      # 23
                    (self.M_H2, self.M_H2O, self.V_H2, self.V_H2O),    # 24
                    (self.M_CO, self.M_H2O, self.V_CO, self.V_H2O)]    # 34

        k_ms = []
        
        for M_i, M_j, V_i, V_j in pairs:
            D_ij = self.Fuller_correlation(M_i, M_j, V_i, V_j)

            # Calculate Reynolds number
            Re = self.rho_g * self.u_g * self.d_p / self.eta_g

            # Calculate Schmidt number
            Sc = self.eta_g / (self.rho_g * D_ij)

            if correlation == 'ranz-marshall':
                k_mt = D_ij / self.d_p * (2 + 0.06 * Re**0.5 * Sc**(1/3))
            elif correlation == 'gunn':
                k_mt = D_ij / self.d_p * ((7 - 10 * self.eps_b + 5 * self.eps_b**2) * (1 + 0.7 * Re**0.2 * Sc**0.33) + (1.33 - 2.4 * self.eps_b + 1.2 * self.eps_b**2) * Re**0.7 * Sc**0.33)
            else:
                raise ValueError("Use 'ranz-marshall' or 'gunn'.")

            k_ms.append(k_mt)

        return k_ms
    
    def calculate_average_diffusion_coefficients(self):
        # Check old script in case temperature balance is required
        diffusion_coeff = np.zeros((self.Nc,self.Nc))
        M = np.array([self.M_CO2,self.M_H2,self.M_CO,self.M_H2O, self.M_CH4])
        V = np.array([self.V_CO2,self.V_H2,self.V_CO,self.V_H2O, self.V_CH4])
        
        for i in range(self.Nc): # Takes component i
            for j in range(self.Nc): # Calculates binary diffusion of i with each other component j
                if i != j:
                    diffusion_coeff[i,j] = self.Fuller_correlation(M[i], M[j], V[i], V[j]) # Calculates Dij

        avg_diffusion_coeff = np.zeros(self.Nc)

        for i in range (self.Nc):
            avg_diffusion_coeff[i] = np.sum(diffusion_coeff[i,:])/(self.Nc)
        
        return avg_diffusion_coeff[0], avg_diffusion_coeff[1], avg_diffusion_coeff[2], avg_diffusion_coeff[3] , avg_diffusion_coeff[4]
 
    def calculate_mass_transfer_coefficient(self, D_i, Sc_i):
        # Gunn correlation:
        k_mt = D_i/self.d_p*((7 - 10*self.eps_b + 5*self.eps_b**2)*(1 + 0.7*self.Re**0.2 * Sc_i**0.33) + (1.33 - 2.4*self.eps_b + 1.2*self.eps_b**2)*self.Re**0.7*Sc_i**0.33)
        return k_mt

    def calculate_mass_transfer_fluidized(self, D_i, Sc_i):
        self.g = 9.81
        self.u_mf = 0.014

        self.d_b0 = 0.376*((self.u_g - self.u_mf)**2)
        self.A_R = np.pi*(self.D_R**2)/4 # Area of the reactor
        self.d_b_max = np.minimum(0.65*((self.A_R*(self.u_g - self.u_mf)**(0.4))), self.D_R)

        self.H_mf = self.H_r*((1 - 0.4)/(1-0.42))

        # Average values taken in the middle of the column
        self.d_b_avg = self.d_b_max - (self.d_b_max - self.d_b0)*(np.exp(-(0.15*self.H_r)/(self.D_R)))
        self.u_b_avg = self.u_g - self.u_mf + 0.711*(self.g*self.d_b_avg)
        self.u_br = 0.711*((self.g*self.d_b_avg)) #0.711*(np.sqrt(self.g*self.d_b_avg))

        # Phase fractions per m3 of bubble volume
        self.fb = (self.u_g - self.u_mf)/self.u_b_avg
        self.fcw = 3*self.fb*(self.u_mf/self.eps_mf)/((0.711*(self.g*self.d_b_avg)) - (self.u_mf/self.eps_mf))
        self.femulsion = 1 - self.fb - self.fcw

        # Define phase fractions assumed constant
        self.eps_bubble = self.fb 
        self.eps_cw = self.fcw 
        self.eps_emulsion = self.femulsion 
        self.eps_solids = (self.fcw + self.femulsion)*(1 - self.eps_mf)

        # Solid phase fractions
        self.gamma_bubble = 0.001
        self.gamma_cw = (1 - self.eps_mf)*(((3*self.u_mf/self.eps_mf)/(0.711*(np.sqrt(self.g*self.d_b_avg)) -self.u_mf/self.eps_mf)) + 0.5)
        self.gamma_emulsion  = 1 - self.eps_mf

        # Define specific surface areas in m3 per reactor volume
        self.a_bubble = (6/(self.d_b_avg))*self.eps_bubble
        self.a_gs = (6/self.d_p)*(1-self.eps_mf)

        # Bubble to cloud
        k_bc = 4.5*(self.u_mf/self.d_b_avg) + 5.85*(((D_i**(1/2))*(self.g**(1/4)))/(self.d_b_avg**(5/4)))
        
        # Cloud to emulsion
        k_ce = 13.56*((D_i*self.eps_mf*self.u_b_avg/(self.d_b_avg**3))**(1/2))

        # Solid vertical dispersion
        Dsv = ((self.fcw**2)*self.eps_mf*self.fb*self.d_b_avg*(self.u_b_avg**2))/(3*self.u_mf)

        return k_bc, k_ce, Dsv

    def init_field(self):
        self.c = np.full([self.Nz, self.Nc*self.Nph+1], self.c_0, dtype='float')
    
    def mass_conservation(self):
        
        self.initial_m_tot = (self.M_H2*self.c_in[:,3] + self.M_CO*self.c_in[:,0])*self.vel*self.A_R
        self.m_tot_array = np.ones(np.size(self.c))*(self.initial_m_tot)

        self.m_bubble = (self.M_CO2*self.c[:,0] + self.M_H2O*self.c[:,1] + self.M_CO*self.c[:,2]  + self.M_H2*self.c[:,3])*self.vel*self.A_R
        self.m_cloud_wake = (self.M_CO2*self.c[:,5] + self.M_H2O*self.c[:,6] + self.M_CO*self.c[:,7]  + self.M_H2*self.c[:,8])*self.vel*self.A_R
        self.m_emulsion = (self.M_CO2*self.c[:,10] + self.M_H2O*self.c[:,11] + self.M_CO*self.c[:,12]  + self.M_H2*self.c[:,13])*self.vel*self.A_R
        
        self.m_tot = self.m_bubble + self.m_cloud_wake + self.m_emulsion
        plt.plot(self.m_tot)
        return 
    
    def reaction(self, c): 
        f = np.zeros_like(c)
       
        P_CH4 = c[:,10]*self.R_gas*self.T_in
        P_H2O = c[:,11]*self.R_gas*self.T_in
        P_H2  = c[:,12]*self.R_gas*self.T_in
        P_CO2 = c[:,13]*self.R_gas*self.T_in
        P_CO  = c[:,14]*self.R_gas*self.T_in
              
        # self.r1 = 4.3e-3*(c[:,4]*c[:,1] - c[:,3]*c[:,2]) #self.r1_real# 4.3e-3# Average SMR rate
        # self.r2 = 6.8e-3*(c[:,2]*c[:,1] - c[:,3]*c[:,0] ) #self.r2_real #6.8e-3 # Average WGS rate

        # # # WGS Kinetics: Kinetic constant
        # k_WGS_0 = 9.74e4
        # E_WGS = 83.6e3 
        # k_WGS = k_WGS_0*np.exp(-(E_WGS/self.R_gas*self.T_in))

        # # # Equillibrium constant
        # self.Keq_WGS_empirical = 10**((-2.4198 + 0.0003855*self.T_in + (2180.6/self.T_in)))

        # self.deltaG_WGS = -1.9e3 # J/mol
        # self.Keq_WGS = np.exp(self.deltaG_WGS/(self.R_gas*self.T_in)) 
        # r_WGS = (k_WGS/P_H2O)*(1000)*(P_CO*P_H2O - (P_H2*P_CO2/self.Keq_WGS))
        # r_WGS[np.isnan(r_WGS)] = 0

        # Reverse WGS kinetics
        k = 11101.2*np.exp(-117432/(self.R_gas*self.T_in)) # Rate constant [mol/s/gcat/bar]
        self.Keq_RWGS_empirical = 1/(10**((-2.4198 + 0.0003855*self.T_in + (2180.6/self.T_in))))
        self.Keq_RWGS = np.exp(12.11 - 5319/self.T_in- 1.012*np.log(self.T_in) + 1.144*10**(-4*self.T_in))
        K_H2O = 96808*np.exp(-51979/(self.R_gas*self.T_in))
        r_RWGS = k*(P_CO2*P_H2 - P_CO*P_H2O/self.Keq_RWGS_empirical)/(P_H2 + K_H2O*P_H2O) * self.rho_s * 1000 / 101325 # Reaction rate [mol/m3cat/s]
        r_RWGS[np.isnan(r_RWGS)] = 0

        # Validation case
        # -H2O - H2 - CO2 - CO -CH4 for 3 PHASES -> 15 balances
        # self.pCH4 = self.R_gas*self.T_in*(self.c[:,10])/101325
        # self.pH2O = self.R_gas*self.T_in*(self.c[:,11])/101325
        # self.pH2 = self.R_gas*self.T_in*(self.c[:,12])/101325
        # self.pCO2 = self.R_gas*self.T_in*(self.c[:,13])/101325
        # self.pCO = self.R_gas*self.T_in*(self.c[:,14])/101325

        # # Steam-Methane Reforming
        # self.deltaG_SMR = 21.3e3 # J/mol
        # self.Keq_SMR = np.exp(self.deltaG_SMR/(self.R_gas*self.T_in))
        # self.A1 = 2.45e2 # Preexponential factor
        # self.Eact_1 = 106.9e3 # Activation Energy (J/mol)
        # self.k1 = self.A1*np.exp((-self.Eact_1)/(self.R_gas*self.T_in))
        # self.SMR_kinetics = self.k1*(self.pCH4*self.pH2O - (self.pH2**3)*(self.pCO/self.Keq_SMR))/((self.pH2O**(1.596) + 0.001)) # Power is 1.5986
        # self.r1_real = self.SMR_kinetics*self.rho_s*1e3 # /101325

        # # Water-Gas Shift
        # self.deltaG_WGS = -1.9e3 # J/mol
        # self.Keq_WGS = np.exp(self.deltaG_WGS/(self.R_gas*self.T_in))
        # self.Keq_WGS_empirical = 10**((-2.4198 + 0.0003855*self.T_in + (2180.6/self.T_in)))
        # self.A2 = 2.62e5 # Preexponential factor
        # self.Eact_2 = 54.5e3 # Activation Energy (J/mol)
        # self.k2 = self.A2*np.exp((-self.Eact_2)/(self.R_gas*self.T_in))
        # self.WGS_kinetics = self.k2*(self.pCO*self.pH2O - (self.pH2)*(self.pCO2/self.Keq_WGS_empirical))/(self.pH2O + 0.001)
        # self.r2_real = self.WGS_kinetics*self.rho_s*1e3 #/101325

        # self.r1_real[np.isnan(self.r1_real)]=0
        # self.r2_real[np.isnan(self.r2_real)]=0


        # T_p = c[:,9]
        # T_g = c[:,4]

        # c_g = c[:,:5]
        # c_s_I = c[:,6:11]
        # P = c[:,-1]

        # xavg = 0.5*(c_g+c_s_I)/self.C_tot_in
        # dc = c_s_I - c_g

        # r = []
        
        # for i in range(0, self.Nz):
        #     # Particle model is called, in the reactor model the particle model is in steady state
        #     part = particle_model_adsorption_RWGS_GOOD(40,self.d_p,np.inf,c[i,:],self.T_in,self.P_in,c_dependent_diffusion=self.c_dependent_diffusion)
        #     eta, phi, rapp = part.solve()
        #     r.append(rapp)

        # r = np.array(r)*(1-self.eps_b)
        # print(r)
        # # r = 0

        # rho_g = self.rho_g
        # Cp_g = self.Cp_g
        
        # # rho_g = (c[:,0]*self.M_CO2+c[:,1]*self.M_H2+c[:,2]*self.M_CO+c[:,3]*self.M_H2O)*1e-3
        # # Cp_g =  (c[:,0]*self.Cp_CO2+c[:,1]*self.Cp_H2+c[:,2]*self.Cp_CO+c[:,3]*self.Cp_H2O)/(c[:,0]+c[:,1]+c[:,2]+c[:,3])
            
        # if self.Maxwell_Stefan == True:
        #     flx = np.zeros((self.Nz, 4))

        #     flx[:,0] = - r/self.a_gs
        #     flx[:,1] = - r/self.a_gs
        #     flx[:,2] = + r/self.a_gs
        #     flx[:,3] = + r/self.a_gs 

        #     f[:,0] = flx[:,0]*self.a_gs
        #     f[:,1] = flx[:,1]*self.a_gs
        #     f[:,2] = flx[:,2]*self.a_gs
        #     f[:,3] = flx[:,3]*self.a_gs

        #     f[:,4] =  self.h_gs*self.a_gs*(T_p-T_g)/(rho_g*Cp_g)    
            
        #     f[:,5] = dc[:,0] + (xavg[:,0]*flx[:,1] - xavg[:,1]*flx[:,0])/self.km_12 \
        #                      + (xavg[:,0]*flx[:,2] - xavg[:,2]*flx[:,0])/self.km_13 \
        #                      + (xavg[:,0]*flx[:,3] - xavg[:,3]*flx[:,0])/self.km_14
            
        #     f[:,6] = dc[:,1] + (xavg[:,1]*flx[:,0] - xavg[:,0]*flx[:,1])/self.km_12 \
        #                     + (xavg[:,1]*flx[:,2] - xavg[:,2]*flx[:,1])/self.km_23 \
        #                     + (xavg[:,1]*flx[:,3] - xavg[:,3]*flx[:,1])/self.km_24 

        #     f[:,7] = dc[:,2] + (xavg[:,2]*flx[:,0] - xavg[:,0]*flx[:,2])/self.km_13 \
        #                     + (xavg[:,2]*flx[:,1] - xavg[:,1]*flx[:,2])/self.km_23 \
        #                     + (xavg[:,2]*flx[:,3] - xavg[:,3]*flx[:,2])/self.km_34 
            
        #     f[:,8] = dc[:,3] + (xavg[:,3]*flx[:,0] - xavg[:,0]*flx[:,3])/self.km_14 \
        #                     + (xavg[:,3]*flx[:,1] - xavg[:,1]*flx[:,3])/self.km_24 \
        #                     + (xavg[:,3]*flx[:,2] - xavg[:,2]*flx[:,3])/self.km_34 

        #     f[:,9] =  -self.h_gs*self.a_gs*(T_p-T_g) -self.H_r*r+self.Q_joule*(1-self.eps_b)
           
        #     # Ergun equation 
        #     f[:,10] = -(150*self.eta_g*self.u_g*(1-self.eps_b)**2/(self.d_p**2*self.eps_b**3)+1.75*self.rho_g*(1-self.eps_b)/(self.d_p*self.eps_b**2)*self.u_g**2)

        # else:

        # Bubble phase:
        f[:,0] = self.Kbe_CO2*self.a_bubble*(c[:,5]-c[:,0])*self.eps_bubble
        f[:,1] = self.Kbe_CO2*self.a_bubble*(c[:,6]-c[:,1])*self.eps_bubble 
        f[:,2] = self.Kbe_CO2*self.a_bubble*(c[:,7]-c[:,2])*self.eps_bubble 
        f[:,3] = self.Kbe_CO2*self.a_bubble*(c[:,8]-c[:,3])*self.eps_bubble 
        f[:,4] = self.Kbe_CO2*self.a_bubble*(c[:,9]-c[:,4])*self.eps_bubble 
        
        # # Cloud-Wake phase:
        # f[:,5] = -self.k_bc_CH4*self.a_bubble*(c[:,5]-c[:,0]) + self.k_ce_CH4*self.a_gs*(c[:,10]-c[:,5])  + self.k_gs_CH4*self.a_gs*(c[:,15] - c[:,5]) # CH4
        # f[:,6] = -self.k_bc_H2O*self.a_bubble*(c[:,6]-c[:,1])  + self.k_ce_H2O*self.a_gs*(c[:,11]-c[:,6]) + self.k_gs_H2O*self.a_gs*(c[:,16] - c[:,6]) # H2O
        # f[:,7] = -self.k_bc_H2*self.a_bubble*(c[:,7]-c[:,2])  + self.k_ce_H2*self.a_gs*(c[:,12]-c[:,7]) + self.k_gs_H2*self.a_gs*(c[:,17] - c[:,7]) # H2 
        # f[:,8] = -self.k_bc_CO2*self.a_bubble*(c[:,8]-c[:,3]) + self.k_ce_CO2*self.a_gs*(c[:,13]-c[:,8]) + self.k_gs_CO2*self.a_gs*(c[:,18] - c[:,8]) # CO2
        # f[:,9] = -self.k_bc_CO*self.a_bubble*(c[:,9]-c[:,4]) + self.k_ce_CO*self.a_gs*(c[:,14]-c[:,9]) + self.k_gs_CO*self.a_gs*(c[:,19] - c[:,9])  # CO

        # Emulsion phase:
        f[:,5] = -self.Kbe_CO2*self.a_gs*(c[:,5]-c[:,0])*self.eps_bubble + self.k_gs_CH4*self.a_gs*(c[:,10] - c[:,5])*self.eps_emulsion*self.gamma_emulsion # Methane 
        f[:,6] = -self.Kbe_CO2*self.a_gs*(c[:,6]-c[:,1])*self.eps_bubble + self.k_gs_H2O*self.a_gs*(c[:,11] - c[:,6])*self.eps_emulsion*self.gamma_emulsion # Water
        f[:,7] = -self.Kbe_CO2*self.a_gs*(c[:,7]-c[:,2])*self.eps_bubble + self.k_gs_H2*self.a_gs*(c[:,12] - c[:,7])*self.eps_emulsion*self.gamma_emulsion # Hydrogen
        f[:,8] = -self.Kbe_CO2*self.a_gs*(c[:,8]-c[:,3])*self.eps_bubble + self.k_gs_CO2*self.a_gs*(c[:,13] - c[:,8])*self.eps_emulsion*self.gamma_emulsion # Carbon dioxide 
        f[:,9] = -self.Kbe_CO2*self.a_gs*(c[:,9]-c[:,4])*self.eps_bubble + self.k_gs_CO*self.a_gs*(c[:,14] - c[:,9])*self.eps_emulsion*self.gamma_emulsion # Carbon Monoxide

        # Solid phase:
        f[:,10] = - self.k_gs_CH4*self.a_gs*(c[:,10] - c[:,5])*self.eps_emulsion*self.gamma_emulsion # + r[:,0] #- self.r1_real # Methane 
        f[:,11] =  - self.k_gs_H2O*self.a_gs*(c[:,11] - c[:,6])*self.eps_emulsion*self.gamma_emulsion  + r_RWGS # + r[:,1] #+ 3*self.r1_real + self.r2_real # Water
        f[:,12] = - self.k_gs_H2*self.a_gs*(c[:,12] - c[:,7])*self.eps_emulsion*self.gamma_emulsion  - r_RWGS# + r[:,2] #- self.r1_real -self.r2_real # Hydrogen
        f[:,13] =  - self.k_gs_CO2*self.a_gs*(c[:,13] - c[:,8])*self.eps_emulsion*self.gamma_emulsion - r_RWGS# + r[:,3] #+ self.r2_real # Carbon dioxide 
        f[:,14] = - self.k_gs_CO*self.a_gs*(c[:,14] - c[:,9])*self.eps_emulsion*self.gamma_emulsion  + r_RWGS# + r[:,4] #+ self.r1_real - self.r2_real # Carbon Monoxide

        # Pressure with Ergun equation:
        f[:,15] = -(150*self.eta_g*self.u_g*(1-self.eps_b)**2/(self.d_p**2*self.eps_b**3)+1.75*self.rho_g*(1-self.eps_b)/(self.d_p*self.eps_b**2)*self.u_g**2)

        return f
    
    def init_Jac(self):
        if self.Maxwell_Stefan == True:
            self.Jac_accum = sps.diags([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]*self.Nz, dtype='float', format='csc')/self.dt
        else: 
            self.Jac_accum = sps.diags([self.eps_bubble,self.eps_bubble, self.eps_bubble, self.eps_bubble, self.eps_bubble,self.eps_emulsion, self.eps_emulsion, self.eps_emulsion, self.eps_emulsion, self.eps_emulsion, self.eps_solids, self.eps_solids, self.eps_solids, self.eps_solids, self.eps_solids, 1.0]*self.Nz, dtype='float', format='csc')/self.dt

        Grad, grad_bc = mrm.construct_grad(self.c.shape, self.z_f, self.z_c, self.bc_ax, axis=0)
        Conv, conv_bc = mrm.construct_convflux_upwind(self.c.shape, self.z_f, self.z_c, self.bc_ax, self.vel, axis=0)
        self.Div_ax = mrm.construct_div(self.c.shape, self.z_f, nu=0, axis=0)
        #Dax_m = mrm.construct_coefficient_matrix([[self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g, self.D_ax_g ,0.0]], self.c.shape, axis=0)
        Dax_m = mrm.construct_coefficient_matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.D_sv_CH4 ,self.D_sv_H2O, self.D_sv_H2, self.D_sv_CO2, self.D_sv_CO, 0.0]], self.c.shape, axis=0)
        self.Flux = Conv-Dax_m@Grad
        self.flux_bc = conv_bc -Dax_m@grad_bc
        self.g_const = self.Div_ax@self.flux_bc
        self.Jac_const = self.Jac_accum + self.Div_ax@self.Flux

    def lin_pde(self, c, c_old):
        f_react, Jac_react = mrm.numjac_local(self.reaction, c)
        c_f, dc_f = mrm.interp_cntr_to_stagg_tvd(c, self.z_f, self.z_c, self.bc_ax, self.vel, mrm.minmod)
        dg_conv = self.Div_ax@(self.vel*dc_f).reshape(-1,1)
        g = self.g_const + self.Jac_const@c.reshape(-1,1) + dg_conv - self.Jac_accum@c_old.reshape(-1,1) - f_react.reshape(-1,1)
        Jac = self.Jac_const-Jac_react 
        return g, Jac 
    
    def plot_pre(self):
            self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(10, 7))  # Adjust the figsize as needed
            self.bubble = self.ax1.grid()
            self.cloudwake = self.ax2.grid()
            self.emulsion = self.ax3.grid()
            self.pressure = self.ax4.grid()
            self.ax1.set_xlabel('Reactor Length [m]')
            self.ax1.set_ylabel('Concentration [mol m-3]')
            self.ax1.set_title('Bubble Phase Concentrations')
            self.ax2.set_xlabel('Reactor Length [m]')
            self.ax2.set_ylabel('Concentration [mol m-3]')
            self.ax2.set_title('Emulsion Phase Concentrations')
            self.ax3.set_xlabel('Reactor Length [m]')
            self.ax3.set_ylabel('Concentration [mol m-3]')
            self.ax3.set_title('Solid Phase Concentrations')
            self.ax4.set_xlabel('Reactor Length [m]')
            self.ax4.set_ylabel('Concentration [mol m-3]')
            self.ax4.set_title('Solid Phase Concentrations')

    def solve(self): 
        # self.plot_pre()
        self.freq_out = 100
        self.t = 0

        for i in range(self.Nt):
            start = time.time()
            c_old = self.c.copy()
            self.c  = mrm.newton(lambda c: self.lin_pde(c, c_old), c_old, tol = 1e-4, maxfev=500).x
            #self.c = mrm.newton(lambda c: self.lin_pde(self.c, c_old), c_old, maxfev =50, callback=lambda x,f:mrm.clip_approach(x, f)).x

            self.t += self.dt
            # if (i % self.freq_out == 0):
            self.plot()

        self.Conversion_CO2 = 1 - (self.c[-1,3])/(self.C_CO2_in)      
        # print(f'Conversion CO2 = {self.Conversion_CO2*100:.1f}%')

        self.pdrop = self.c[0,10] - self.c[-1,10]     
        # print(f'Pressure drop = {self.pdrop:.1f} bar')

        self.C_tot = (self.c[-1,0] + self.c[-1,1] + self.c[-1,2] + self.c[-1,3] + self.c[-1,4])
        self.Fraction_CO2 = self.c[-1,3]/self.C_tot
        self.Fraction_CO = self.c[-1,4]/self.C_tot

        return self.Conversion_CO2, self.pdrop, self.Fraction_CO, self.Fraction_CO2 
    
    def plot(self):
        labels = ['CH4','H2O','H2','CO2','CO']

        if self.dt == np.inf: # Steady state plot
            plt.suptitle('Validation Case - Steady State')
        
            bubble_index = np.array([0,1,2,3,4])
            for i in bubble_index:
                self.ax1.plot(self.z_c,self.c[:,i],label=labels[i]+' Bubble')                  
            self.ax1.legend()

            # cw_index = np.array([5,6,7,8,9])
            # for i in cw_index:
            #     self.ax2.plot(self.z_c,self.c[:,i],'-',label=labels[i - self.Nc]+' Cloud-Wake') 
            # self.ax2.legend()

            emulsion_index = np.array([5,6,7,8,9])
            for i in emulsion_index:
                self.ax2.plot(self.z_c,self.c[:,i],'-.',label=labels[i - self.Nc] +' Emulsion')
            self.ax2.legend()

          
            solid_index = np.array([10,11,12,13,14])
            for i in solid_index:
                self.ax3.plot(self.z_c,self.c[:,i],'--',label=labels[i - self.Nc*2] +' Solid')
            self.ax3.legend()

            self.ax4.plot(self.z_c,self.c[:,15],label='Pressure')
            self.ax4.legend()
            
            plt.show()
            
        else: # Transient plot
            
            plt.clf()
            plt.suptitle(f'Time: {self.t:.2f} s')
            start = time.time()
            plt.subplot(221)
            labels = ['CH4','H2O','H2','CO2','CO']
            bubble_index = np.array([0,1,2,3,4])
            for i in bubble_index:
                plt.plot(self.z_c,self.c[:,i],label=labels[i]+' Bubble')            
            plt.legend()
            plt.grid()
            plt.xlabel('Reactor length [m]') 
            plt.ylabel('Concentration [mol m-3]') 
            
            # plt.subplot(222)
            # cw_index = np.array([5,6,7,8,9])
            # for i in cw_index:
            #     plt.plot(self.z_c,self.c[:,i],'--', label=labels[i - self.Nc]+' Cloud-Wake')
            # plt.legend()
            # plt.grid()
            # plt.xlabel('Reactor length [m]') 
            # plt.ylabel('Concentration [mol m-3]') 
            

            plt.subplot(222)
            emulsion_index = np.array([5,6,7,8,9])
            for i in emulsion_index:
                plt.plot(self.z_c,self.c[:,i],'--',label=labels[i - self.Nc]+' Emulsion')
            plt.legend()
            plt.grid()
            plt.xlabel('Reactor length [m]') 
            plt.ylabel('Concentration [mol m-3]') 
            

            plt.subplot(223)
            solid_index = np.array([10,11,12,13,14])
            for i in solid_index:
                plt.plot(self.z_c,self.c[:,i],'--',label=labels[i - self.Nc*2] +' Solid')
            plt.legend()
            plt.grid()
            plt.xlabel('Reactor length [m]') 
            plt.ylabel('Concentration [mol m-3]') 

            plt.subplot(224)
            plt.plot(self.z_c,self.c[:,15]/101325,label='Pressure')
            plt.legend()
            plt.grid()
            plt.xlabel('Reactor length [m]') 
            plt.ylabel('Pressure [Bar]') 
            
            # print(f'Time to solve time step: {time.time()-start:.2f} s')

            plt.pause(0.01)

        return
            
# if __name__ == "__main__":
# Steady state:
# reactor = fluidized_bed_RWGS(100, 2e-4, 1.0 ,np.inf, 0.22, 0, 900, 1*101325, Maxwell_Stefan=True, c_dependent_diffusion=True)
# reactor.solve()
# print(reactor.Conversion_CO2)
# print(f'CO2 Conversion in Fluidized Bed: {reactor.Conversion_CO2:.2f}')

#    Transient:
Temperature = 900
X_CO2_in = 0.35
reactor = geldart_B_RWGS(100, 3e-4, 1.0, 0.01, 0.22, X_CO2_in, Temperature, 1*101325, Maxwell_Stefan=False, c_dependent_diffusion=False)
reactor.solve()

# Temp = []
# Conversion_CO2 = []
# temps = [500, 550, 600, 650, 700, 750, 800]
# X_CO_in = 0.35

# for temp in temps:
#     reactor = geldart_B_RWGS(100, 3e-4, 1.0, np.inf, 0.22, X_CO_in, temp, 1*101325, Maxwell_Stefan=False, c_dependent_diffusion=False)
#     reactor.solve()
#     Conversion_CO2.append(reactor.Conversion_CO2)
#     Temp.append(temp)

# plt.figure(figsize = (8,6))
# plt.plot(temps,Conversion_CO2, label = 'CO2 conversion')
# plt.grid()
# plt.xlabel('Reactor Temperature [K]') 
# plt.ylabel('Conversion [-]') 
# plt.title('RWGS Equillibrium Conversion Obtained')
# plt.show()


# Temp = []
# # Inlet has CO2/H2
# Composition_CO2 = []

# Composition_CO = [] 
# temps = [100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
# X_CO_in = 0.5

# for temp in temps:
#     reactor = geldart_B_RWGS(100, 3e-4, 1.0, np.inf, 0.22, X_CO_in, temp, 1*101325, Maxwell_Stefan=False, c_dependent_diffusion=False)
#     reactor.solve()

#     Composition_CO2.append(reactor.Fraction_CO2)
#     Composition_CO.append(reactor.Fraction_CO)
#     Temp.append(temp)

# plt.figure(figsize = (8,6))
# plt.plot(temps,Composition_CO2, label = 'CO2/H2 equillibrium composition')
# plt.plot(temps,Composition_CO, label = 'CO/H2O equillibrium composition' )
# plt.grid()
# plt.xlabel('Reactor Temperature [K]') 
# plt.ylabel('Conversion [-]') 
# plt.legend()
# plt.title('RWGS Equillibrium Compositions Obtained')
# plt.show()