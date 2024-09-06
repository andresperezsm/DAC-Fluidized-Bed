import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.optimize as sopt
import pymrm as mrm
import time

# from particle_model_RWGS import *
from FluidizationHydrodynamics import *

class geldart_A_WGS: 
    def __init__(self,Nodes, d_p, Lr, dt, gas_velocity, X_CO_in, T_in, P_in,c_dependent_diffusion=False, plot=False):
        # Inlet conditions:
        self.T_in = T_in # Inlet temperature [K]
        self.P_in = P_in # Inlet pressure [Pa]
        self.R_gas = 8.314 # Gas constant [J/mol/K]

        self.plotting = plot 

        # Initial concentrations
        self.inventory_cat = 0.592 # Catalyst inventory to obtain the reactor length
        self.X_CO_in = X_CO_in # X_CO2_in # Bulk CO2 mole fraction [-]
        self.X_CH4_in = 0 # (1 - self.X_CO2_in)*(phi_CH4/(phi_CH4 + phi_H2O)) # Bulk CH4 mole fraction [-]
        self.X_H2O_in = (1 - self.X_CO_in)*(1 - self.X_CH4_in) # 1 - 0.5 # Bulk H2 mole fraction [-]
        self.C_CO_in = 0.0877 # self.P_in*self.X_CO_in/(self.R_gas*self.T_in) # Bulk CO2 concentration [mol/m3]
        self.C_H2O_in = 0.4830 # self.P_in*self.X_H2O_in/(self.R_gas*self.T_in) # Bulk H2 concentration [mol/m3]
        self.C_CH4_in = 0 # self.P_in*self.X_CH4_in/(self.R_gas*self.T_in) # Bulk CH4 concentration [mol/m3]

        self.C_tot_in = self.C_CO_in+self.C_H2O_in # Total inlet concentration [mol/m3]

        self.c_dependent_diffusion = c_dependent_diffusion

        # Simulation and field parameters
        self.Nz = Nodes # Number of grid cells in the axial direction [-]
        self.dt = dt # Lengt of a time step [-]
        
        self.Nc = 5 # Number of components (temperature included) [-]
        self.Nph = 4 # Number of phases
        
        self.L_R = Lr # Reactor length [m]
        self.D_R = 0.035 # 0.5 # Reactor diameter [m]
        
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
        # z = model.z

        # Phase fractions varying along the axial direction
        self.epsilon_cw = model.epsilon_cloud_wake
        self.epsilon_bubble = model.fb
        self.epsilon_emulsion = model.femulsion*model.fb
        self.epsilon_solids = model.epsilon_solids

        # print(self.epsilon_emulsion)
        # The solid phase fractions (gamma) should be defined in the Fluidization Hydrodynamics
        self.gamma_bubble = model.gamma_b
        self.gamma_cw = model.gamma_cw
        self.gamma_emulsion = model.gamma_e
        self.epsilon_solids = self.gamma_bubble*self.epsilon_bubble + self.gamma_cw*self.epsilon_cw + self.gamma_emulsion*self.epsilon_emulsion

        # Phase velocities as arrays and minimum fludization velocity
        self.umf = model.u_mf
        self.ub = model.u_b
        #ub_second_option = model.u_b_1
        self.ue = model.u_emulsion
        self.us = model.us_down

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
            self.Kbc_CO2 = model.kgas[0,:,0]
            self.Kbc_CO = model.kgas[0,:,1]
            self.Kbc_H2 = model.kgas[0,:,2]
            self.Kbc_H2O = model.kgas[0,:,3]

            # Cloud to emulsion transfer coefficients
            self.Kce_CO2 = model.kgas[1,:,0]
            self.Kce_CO = model.kgas[1,:,1]
            self.Kce_H2 = model.kgas[1,:,2]
            self.Kce_H2O = model.kgas[1,:,3]

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

        self.u_s = (model.fw[0]*model.fb[0]*model.u_b[0])/(1 - model.fb[0] - model.fw[0]*model.fb[0]) # Solid velocities downwards
        self.vel = np.concatenate([np.ones(self.Nc)*self.u_g, np.ones(self.Nc)*self.u_mf, np.ones(self.Nc)*self.u_mf, np.ones(self.Nc)*self.u_s,[1]])
        # self.vel = np.concatenate([np.ones(self.Nc)*self.u_g, np.ones(self.Nc)*self.u_mf, np.ones(self.Nc)*self.u_mf, np.ones(self.Nc)*self.u_s,[1]])

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
        self.c_in[0], self.c_in[1], self.c_in[4] = self.C_CH4_in, self.C_H2O_in, self.C_CO_in
        self.c_in[-1] = self.P_in
        
        self.Dax = np.zeros(self.Nc*self.Nph + 1)
        self.Dax[15], self.Dax[16], self.Dax[17], self.Dax[18], self.Dax[19] = self.D_sv_CH4, self.D_sv_H2O, self.D_sv_H2, self.D_sv_CO2, self.D_sv_CO

        self.bc_ax = {
                    'a': [[[self.Dax]], 1], # Dirichlet boundary conditions
                    'b': [[[self.vel]], 0], # Neumann boundary conditions
                    'd': [[[self.vel*self.c_in]] , 0.0], # Values
                     } # Mixed boundary condition
        
        # self.bc_ax = {
        #             'a': [0, 1], # Dirichlet boundary conditions
        #             'b': [1, 0], # Neumann boundary conditions
        #             'd': [[[self.c_in]] , 0.0], # Values
        #              } # Neumann Boundary condition
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
        self.femulsion = 1 - self.fb - self.fcw*self.fb

        # Define phase fractions assumed constant
        self.eps_bubble = self.fb 
        self.eps_cw =  self.fcw*self.fb
        self.eps_emulsion =  self.femulsion*self.fb 
        self.eps_solids = (self.fcw + self.femulsion)*(1 - self.eps_mf)

        # print(self.eps_bubble + self.eps_cw + self.eps_emulsion)

        # """ THIS IS TO CHECK IF PHASE FRACTIONS ARE THE ISSUE"""
        # self.epsilon_bubble=np.ones(self.Nz)*0.3 # self.eps_bubble*
        # self.epsilon_cw = np.ones(self.Nz)*0.05 #self.eps_cw*
        # self.epsilon_emulsion = np.ones(self.Nz)*0.6 #  self.eps_emulsion*
        # self.epsilon_solids = np.ones(self.Nz)*0.05  # self.eps_solids*
        # print(self.eps_emulsion, self.epsilon_cw, self.eps_bubble)

        # Solid phase fractions
        # self.gamma_bubble = 0.001
        # self.gamma_cw = (1 - self.eps_mf)*(((3*self.u_mf/self.eps_mf)/(0.711*(np.sqrt(self.g*self.d_b_avg)) -self.u_mf/self.eps_mf)) + 0.5)
        # self.gamma_emulsion  = 1 - self.eps_mf

        # Define specific surface areas in m3 per reactor volume
        self.a_bubble = (6/(self.db))*self.eps_bubble #(6/(self.d_b_avg))*self.eps_bubble
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
    
    def concentration_conservation(self):
        self.initial_c_tot = (self.c_in[1] + self.c_in[0] + self.c_in[4]) 
        self.initial_p = self.initial_c_tot*self.R_gas*self.T_in
        self.c_tot_array = np.ones(self.Nz)*(self.initial_c_tot) # Array for plotting

        # Concentrations in each phase
        self.c_bubble = (self.c[:,0] + self.c[:,1] + self.c[:,2]  + self.c[:,3] + self.c[:,4]) # *self.eps_bubble #*self.vel*self.A_R
        self.c_cloud_wake = (self.c[:,5] + self.c[:,6] + self.c[:,7]  + self.c[:,8] + self.c[:,9]) # *self.eps_cw #*self.vel*self.A_R
        self.c_emulsion = (self.c[:,10] + self.c[:,11] + self.c[:,12]  + self.c[:,13] + self.c[:,14]) # *self.eps_emulsion #*self.vel*self.A_R
        self.c_solid = (self.c[:,15] + self.c[:,16] + self.c[:,17]  + self.c[:,18] + self.c[:,19]) #*self.eps_solids #*self.vel*self.A_R

        # The following are for mass conservation
        # self.c_bubble = (self.M_CO2*self.c[:,0] + self.M_H2O*self.c[:,1] + self.M_CO*self.c[:,2]  + self.M_H2*self.c[:,3])*self.eps_bubble #*self.vel*self.A_R
        # self.c_cloud_wake = (self.M_CO2*self.c[:,5] + self.M_H2O*self.c[:,6] + self.M_CO*self.c[:,7]  + self.M_H2*self.c[:,8])*self.eps_cw #*self.vel*self.A_R
        # self.c_emulsion = (self.M_CO2*self.c[:,10] + self.M_H2O*self.c[:,11] + self.M_CO*self.c[:,12]  + self.M_H2*self.c[:,13])*self.eps_emulsion #*self.vel*self.A_R
        
        # Pressures in all the gas phases
        self.p_bubble = (self.c[:,0]*self.R_gas*self.T_in + self.c[:,1]*self.R_gas*self.T_in + self.R_gas*self.T_in*self.c[:,2]  + self.R_gas*self.T_in*self.c[:,3])
        self.p_cloud_wake = (self.R_gas*self.T_in*self.c[:,5] + self.R_gas*self.T_in*self.c[:,6] + self.R_gas*self.T_in*self.c[:,7]  + self.R_gas*self.T_in*self.c[:,8])
        self.p_emulsion = (self.R_gas*self.T_in*self.c[:,10] + self.R_gas*self.T_in*self.c[:,11] + self.R_gas*self.T_in*self.c[:,12]  + self.R_gas*self.T_in*self.c[:,13])
        
        self.c_total = self.c_bubble + self.c_cloud_wake + self.c_emulsion + self.c_solid 

        # # Plotting pressures against inlet pressure
        plt.figure(figsize=(8,5))
        plt.plot(self.z_c, self.c_tot_array*self.R_gas*self.T_in, label = 'Total Pressure at Inlet')
        plt.plot(self.z_c, self.c_bubble*self.R_gas*self.T_in, label = 'Pressure in Bubble Phase')
        plt.plot(self.z_c, self.c_cloud_wake*self.R_gas*self.T_in, label = 'Pressure in Cloud-Wake Phase')
        plt.plot(self.z_c, self.c_emulsion*self.R_gas*self.T_in, label = 'Pressure in Emulsion Phase')
        plt.plot(self.z_c, self.c_solid*self.R_gas*self.T_in, label = 'Pressure in Solid Phase')
        plt.plot(self.z_c, self.c_tot_conservation*self.R_gas*self.T_in, label = 'Sum of Pressures in phases')
        plt.legend()
        plt.xlabel('Reactor Length [m]')
        plt.ylabel('Pressure [Pascal]')
        plt.grid()
        plt.title('Total pressure axial profile')
        plt.show()
    
        # # Concentration plot
        # plt.plot(self.z_c, self.c_tot_array*self.R_gas*self.T_in, label = 'Total Concentration at Inlet')
        # plt.plot(self.z_c, self.c_bubble, label = 'Concentration in Bubble Phase')
        # plt.plot(self.z_c, self.c_cloud_wake, label = 'Concentration in Cloud-Wake Phase')
        # plt.plot(self.z_c, self.c_emulsion, label = 'Concentration in Emulsion Phase')
        # plt.plot(self.z_c, self.c_solid, label = 'Concentration in Solid Phase')
        # plt.plot(self.z_c, self.c_tot, label = 'Sum of concentrations in phases')

        # return self.c_tot_conservation
    
    def reaction(self, c): 
        f = np.zeros_like(c)
       
        P_H2O = c[:,11]*self.R_gas*self.T_in/101325
        P_H2  = c[:,12]*self.R_gas*self.T_in/101325
        P_CO2 = c[:,13]*self.R_gas*self.T_in/101325
        P_CO  = c[:,14]*self.R_gas*self.T_in/101325

        # # # WGS Kinetics: Kinetic constant
        self.k_WGS = (3.08e-4)*np.exp(-(1855.5/self.T_in) + 12.88)

        # # Equillibrium constant
        self.Keq_WGS = np.exp((4577.8/self.T_in) - 4.33)
        rho_s = 4580 # Catalyst density assumed to be iron
        r_WGS = (self.k_WGS)*(P_CO*P_H2O - (P_H2*P_CO2/self.Keq_WGS))*rho_s*self.epsilon_emulsion*self.gamma_emulsion #*0.5 # *self.eps_s
        r_WGS[np.isnan(r_WGS)] = 0

        # # Particle model coupling removed to find issue in conversion
        # r = []

        # for i in range(0, self.Nz):
        #     # Particle model is called, in the reactor model the particle model is in steady state
        #     part = particle_model_RWGS(40,self.d_p,np.inf,c[i,:],self.T_in,self.P_in,c_dependent_diffusion=self.c_dependent_diffusion)
        #     eta, phi, rapp = part.solve()
        #     r.append(rapp)

        # r = np.array(r)*(1-self.eps_b) # self.fb(self.fc + self.fw + self.fe)
        #print(r)

        # Bubble phase:
        # self.a_bubble = 1 # Kbc does not require an a_bubble according to balances in literature
        f[:,0] = self.Kbc_CO2*self.a_bubble*(c[:,5]-c[:,0])*self.epsilon_bubble
        f[:,1] = self.Kbc_H2O*self.a_bubble*(c[:,6]-c[:,1])*self.epsilon_bubble
        f[:,2] = self.Kbc_H2*self.a_bubble*(c[:,7]-c[:,2])*self.epsilon_bubble
        f[:,3] = self.Kbc_CO2*self.a_bubble*(c[:,8]-c[:,3])*self.epsilon_bubble
        f[:,4] = self.Kbc_CO*self.a_bubble*(c[:,9]-c[:,4])*self.epsilon_bubble
        
        # Cloud-Wake phase:
        f[:,5] = -self.Kbc_CO2*self.a_bubble*(c[:,5]-c[:,0])*self.epsilon_bubble + self.Kce_CO2*self.a_bubble*(c[:,10]-c[:,5])*self.epsilon_cw   + self.k_gs_CH4*self.a_gs*(c[:,15] - c[:,5])*self.epsilon_cw *self.gamma_cw  # CH4
        f[:,6] = -self.Kbc_H2O*self.a_bubble*(c[:,6]-c[:,1])*self.epsilon_bubble + self.Kce_H2O*self.a_bubble*(c[:,11]-c[:,6])*self.epsilon_cw   + self.k_gs_H2O*self.a_gs*(c[:,16] - c[:,6])*self.epsilon_cw *self.gamma_cw  # H2O
        f[:,7] = -self.Kbc_H2*self.a_bubble*(c[:,7]-c[:,2])*self.epsilon_bubble  + self.Kce_H2*self.a_bubble*(c[:,12]-c[:,7])*self.epsilon_cw  + self.k_gs_H2*self.a_gs*(c[:,17] - c[:,7])*self.epsilon_cw *self.gamma_cw  # H2 
        f[:,8] = -self.Kbc_CO2*self.a_bubble*(c[:,8]-c[:,3])*self.epsilon_bubble+ self.Kce_CO2*self.a_bubble*(c[:,13]-c[:,8])*self.epsilon_cw  + self.k_gs_CO2*self.a_gs*(c[:,18] - c[:,8])*self.epsilon_cw *self.gamma_cw  # CO2
        f[:,9] = -self.Kbc_CO*self.a_bubble*(c[:,9]-c[:,4])*self.epsilon_bubble + self.Kce_CO*self.a_bubble*(c[:,14]-c[:,9])*self.epsilon_cw   + self.k_gs_CO*self.a_gs*(c[:,19] - c[:,9])*self.epsilon_cw *self.gamma_cw  # CO

        # Emulsion phase:
        f[:,10] = -self.Kce_CO2*self.a_bubble*(c[:,10]-c[:,5])*self.epsilon_cw + self.k_gs_CH4*self.a_gs*(c[:,15] - c[:,10])*self.epsilon_emulsion*self.gamma_emulsion # Methane 
        f[:,11] = -self.Kce_H2O*self.a_bubble*(c[:,11]-c[:,6])*self.epsilon_cw  + self.k_gs_H2O*self.a_gs*(c[:,16] - c[:,11])*self.eps_emulsion*self.gamma_emulsion # Water
        f[:,12] = -self.Kce_H2*self.a_bubble*(c[:,12]-c[:,7])*self.epsilon_cw + self.k_gs_H2*self.a_gs*(c[:,17] - c[:,12])*self.eps_emulsion*self.gamma_emulsion # Hydrogen
        f[:,13] = -self.Kce_CO2*self.a_bubble*(c[:,13]-c[:,8])*self.epsilon_cw + self.k_gs_CO2*self.a_gs*(c[:,18] - c[:,13])*self.eps_emulsion*self.gamma_emulsion # Carbon dioxide 
        f[:,14] = -self.Kce_CO*self.a_bubble*(c[:,14]-c[:,9])*self.epsilon_cw  + self.k_gs_CO*self.a_gs*(c[:,19] - c[:,14])*self.eps_emulsion*self.gamma_emulsion # Carbon Monoxide

        # Solid phase:
        f[:,15] = -self.k_gs_CH4*self.a_gs*(c[:,15] - c[:,10])*self.epsilon_emulsion*self.gamma_emulsion - self.k_gs_CH4*self.a_gs*(c[:,15] - c[:,5])*self.gamma_cw*self.epsilon_cw  #- self.r1_real # Methane 
        f[:,16] = -self.k_gs_H2O*self.a_gs*(c[:,16] - c[:,11])*self.epsilon_emulsion*self.gamma_emulsion - self.k_gs_H2O*self.a_gs*(c[:,16] - c[:,6])*self.gamma_cw*self.epsilon_cw - r_WGS # + r[3] #  #+ 3*self.r1_real + self.r2_real # Water
        f[:,17] = -self.k_gs_H2*self.a_gs*(c[:,17] - c[:,12])*self.epsilon_emulsion*self.gamma_emulsion - self.k_gs_H2*self.a_gs*(c[:,17] - c[:,7])*self.gamma_cw*self.epsilon_cw + r_WGS  # + r[1] ##- self.r1_real -self.r2_real # Hydrogen
        f[:,18] = -self.k_gs_CO2*self.a_gs*(c[:,18] - c[:,13])*self.epsilon_emulsion*self.gamma_emulsion - self.k_gs_CO2*self.a_gs*(c[:,18] - c[:,8])*self.gamma_cw*self.epsilon_cw + r_WGS # + r[0] # #+ self.r2_real # Carbon dioxide 
        f[:,19] = -self.k_gs_CO*self.a_gs*(c[:,19] - c[:,14])*self.epsilon_emulsion*self.gamma_emulsion - self.k_gs_CO*self.a_gs*(c[:,19] - c[:,9])*self.gamma_cw*self.epsilon_cw  - r_WGS # + r[2] #+ self.r1_real - self.r2_real # Carbon Monoxide # + r_RWGS

        # Pressure with Ergun equation:
        f[:,20] = -(150*self.eta_g*self.u_g*(1-self.eps_b)**2/(self.d_p**2*self.eps_b**3)+1.75*self.rho_g*(1-self.eps_b)/(self.d_p*self.eps_b**2)*self.u_g**2)

        return f
    
    def init_Jac(self):
        # Constant phase fraction profile
        # self.Jac_accum = sps.diags([self.eps_bubble,self.eps_bubble, self.eps_bubble, self.eps_bubble, self.eps_bubble, self.eps_cw, self.eps_cw, self.eps_cw, self.eps_cw, self.eps_cw, self.eps_emulsion, self.eps_emulsion, self.eps_emulsion, self.eps_emulsion, self.eps_emulsion, self.eps_solids, self.eps_solids, self.eps_solids, self.eps_solids, self.eps_solids, 0.0]*self.Nz, dtype='float', format='csc')/self.dt
        
        # Changing phase fractions profile
        eps_profile = np.concatenate([self.epsilon_bubble,self.epsilon_bubble, self.epsilon_bubble, self.epsilon_bubble, 
                            self.epsilon_bubble, self.epsilon_cw, self.epsilon_cw, self.epsilon_cw, self.epsilon_cw, self.epsilon_cw, 
                            self.epsilon_emulsion, self.epsilon_emulsion, self.epsilon_emulsion, self.epsilon_emulsion, self.epsilon_emulsion,
                            self.epsilon_solids, self.epsilon_solids, self.epsilon_solids, self.epsilon_solids, self.epsilon_solids, np.ones(self.Nz)])

        self.Jac_accum = sps.diags(eps_profile, dtype='float', format='csc')/self.dt

        Grad, grad_bc = mrm.construct_grad(self.c.shape, self.z_f, self.z_c, self.bc_ax, axis=0)
        self.Div_ax = mrm.construct_div(self.c.shape, self.z_f, nu=0, axis=0)
        self.Dax_m_flux = mrm.construct_coefficient_matrix([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, self.D_sv_CH4 ,self.D_sv_H2O, self.D_sv_H2, self.D_sv_CO2, self.D_sv_CO, 0.0]], self.c.shape, axis=0)
        
        # # Changing velocities profile (Not working)
        # vel_profile = np.concatenate([self.ub, self.ub, self.ub, self.ub, self.ub,
        #                        np.ones(self.Nz)*self.umf, np.ones(self.Nz)*self.umf, np.ones(self.Nz)*self.umf, np.ones(self.Nz)*self.umf, np.ones(self.Nz)*self.umf, 
        #                       np.ones(self.Nz)*self.umf, np.ones(self.Nz)*self.umf, np.ones(self.Nz)*self.umf, np.ones(self.Nz)*self.umf, np.ones(self.Nz)*self.umf, 
        #                       np.ones(self.Nz)*self.u_s, np.ones(self.Nz)*self.u_s, np.ones(self.Nz)*self.u_s, np.ones(self.Nz)*self.u_s, np.ones(self.Nz)*self.u_s, np.ones(self.Nz)])
        # vel_operator = sps.csc_array(sps.diags(vel_profile.flatten(), format='csc'))

        """ This convective term makes use of self.vel which is set to 1 in the init function. Therefore, the vel_operator should be constructed"""
        # # The following uses only a constant value for the phase velocities given by the following array: vel_coefficients

        # # Low u_mf value causes issues
        self.u_mf = 1
        vel_coefficients = np.concatenate([np.ones(self.Nc)*self.ub[0], np.ones(self.Nc)*self.u_mf, np.ones(self.Nc)*self.u_mf, np.ones(self.Nc)*self.u_s,[1]])

        # self.ub[0],self.ue[0]
        # vel_operator = mrm.construct_coefficient_matrix([vel_coefficients], self.c.shape, axis=0)
        # Conv, conv_bc = mrm.construct_convflux_upwind(self.c.shape, self.z_f, self.z_c, self.bc_ax, self.vel, axis=0)
        # self.Flux = vel_operator@Conv-self.Dax_m_flux@Grad
        # self.flux_bc = vel_operator@conv_bc -self.Dax_m_flux@grad_bc

        """ vel_operator uses the array of velocities profiles so vel should be 1 when constructing the Conv"""
        Conv, conv_bc = mrm.construct_convflux_upwind(self.c.shape, self.z_f, self.z_c, self.bc_ax, vel_coefficients, axis=0)
        self.Flux = Conv-self.Dax_m_flux@Grad
        self.flux_bc = conv_bc -self.Dax_m_flux@grad_bc

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
            self.ax2.set_title('Cloud-Wake Phase Concentrations')
            self.ax3.set_xlabel('Reactor Length [m]')
            self.ax3.set_ylabel('Concentration [mol m-3]')
            self.ax3.set_title('Emulsion Phase Concentrations')
            self.ax4.set_xlabel('Reactor Length [m]')
            self.ax4.set_ylabel('Concentration [mol m-3]')
            self.ax4.set_title('Solid Phase Concentrations')

    def solve(self): 
        if self.plotting == True:
            self.plot_pre()
        self.freq_out = 100
        self.t = 0

        for i in range(self.Nt):
            start = time.time()
            c_old = self.c.copy()
            self.c  = mrm.newton(lambda c: self.lin_pde(c, c_old), c_old, tol = 1e-4, maxfev=500).x
            #self.c = mrm.newton(lambda c: self.lin_pde(self.c, c_old), c_old, maxfev =50, callback=lambda x,f:mrm.clip_approach(x, f)).x

            self.t += self.dt
            # if (i % self.freq_out == 0):
            if self.plotting == True:
                self.plot()

        # self.c_tot_conservation = self.concentration_conservation()
        # self.concentration_conservation()

        # # Conversion 
        """ THIS DEFINITION TAKES INTO CONSIDERATION THE CO CONCENTRATION IN ALL THE PHASES"""
        # self.Conversion_CO2 = (1 - (self.c[-1,4]*self.eps_bubble + self.c[-1,9]*self.eps_cw + self.c[-1,14]*self.eps_emulsion + self.c[-1,19]*self.eps_solids)/(self.C_CO2_in*self.eps_bubble)) 
        
        """ THIS DEFINITION TAKES INTO CONSIDERATION THE CO CONCENTRATION ONLY IN THE BUBBLE PHASE"""
        self.Conversion_CO = (1 - (self.c[-1,4])/(self.C_CO_in))  
        # print(f'Conversion CO2 = {self.Conversion_CO2*100:.1f}%')

        self.pdrop = self.c[0,10] - self.c[-1,10]     
        # print(f'Pressure drop = {self.pdrop:.1f} bar')

        self.C_tot = (self.c[-1,0] + self.c[-1,1] + self.c[-1,2] + self.c[-1,3] + self.c[-1,4])
        self.Fraction_CO2 = self.c[-1,3]/self.C_tot
        self.Fraction_CO = self.c[-1,4]/self.C_tot
        return self.Conversion_CO, self.pdrop, self.Fraction_CO, self.Fraction_CO2 
    
    def plot(self):
        labels = ['$\mathrm{CH_4}$','$\mathrm{H_2O}$','$\mathrm{H_2}$','$\mathrm{CO_2}$','CO']

        if self.dt == np.inf: # Steady state plot
            plt.suptitle('Validation Case - Steady State')
        
            bubble_index = np.array([1,2,3,4])
            for i in bubble_index:
                self.ax1.plot(self.z_c,self.c[:,i],label=labels[i]+' Bubble')                  
            self.ax1.legend()

            cw_index = np.array([6,7,8,9])
            for i in cw_index:
                self.ax2.plot(self.z_c,self.c[:,i],'-',label=labels[i - self.Nc]+' Cloud-Wake') 
            self.ax2.legend()

            emulsion_index = np.array([11,12,13,14])
            for i in emulsion_index:
                self.ax3.plot(self.z_c,self.c[:,i], label=labels[i - self.Nc*2] +' Emulsion')
           
            solid_index = np.array([16,17,18,19])
            for i in solid_index:
                self.ax3.plot(self.z_c,self.c[:,i],'--',label=labels[i - self.Nc*3] +' Solid')
            self.ax3.legend()

            self.c_bubble = (self.c[:,0] + self.c[:,1] + self.c[:,2]  + self.c[:,3] + self.c[:,4])*self.eps_bubble #*self.vel*self.A_R
            self.c_cloud_wake = (self.c[:,5] + self.c[:,6] + self.c[:,7]  + self.c[:,8] + self.c[:,9])*self.eps_cw #*self.vel*self.A_R
            self.c_emulsion = (self.c[:,10] + self.c[:,11] + self.c[:,12]  + self.c[:,13] + self.c[:,14])*self.eps_emulsion #*self.vel*self.A_R
            self.c_solid = (self.c[:,15] + self.c[:,16] + self.c[:,17]  + self.c[:,18] + self.c[:,19])*self.eps_solids #*self.vel*self.A_R
            self.c_total = self.c_bubble + self.c_cloud_wake + self.c_emulsion + self.c_solid 

            self.ax4.plot(self.z_c,self.c[:,20],label='Pressure')
            self.ax4.plot(self.z_c, self.c_total*self.R_gas*self.T_in, label = 'Sum of Pressures in phases')
            self.ax4.legend()
            
            plt.show()
            
        else: # Transient plot
            labels = ['$\mathrm{CH_4}$','$\mathrm{H_2O}$','$\mathrm{H_2}$','$\mathrm{CO_2}$','CO']
            
            plt.clf()   
            plt.suptitle(f'Time: {self.t:.2f} s')
            start = time.time()

            plt.subplot(221)
            bubble_index = np.array([1,2,3,4])
            for i in bubble_index:
                plt.plot(self.z_c,self.c[:,i],label=labels[i]+' Bubble')            
            plt.legend()
            plt.grid()
            plt.xlabel('Reactor length [m]') 
            plt.ylabel('Concentration [mol m-3]') 

            plt.subplot(222)
            cw_index = np.array([6,7,8,9])
            for i in cw_index:
                  plt.plot(self.z_c,self.c[:,i],'-.', label=labels[i - self.Nc]+' Cloud-Wake')
            plt.legend()
            plt.grid()
            plt.xlabel('Reactor length [m]') 
            plt.ylabel('Concentration [mol m-3]') 

            plt.subplot(223)
            emulsion_index = np.array([11,12,13,14])
            solid_index = np.array([16,17,18,19])
            for i in emulsion_index:
                plt.plot(self.z_c,self.c[:,i], label=labels[i - self.Nc*2]+' Emulsion')
            for i in solid_index:
                plt.plot(self.z_c,self.c[:,i],'--',label=labels[i - self.Nc*3]+' Solid')
            plt.legend()
            plt.grid()
            plt.xlabel('Reactor length [m]') 
            plt.ylabel('Concentration [mol m-3]') 

            self.c_bubble = (self.c[:,0] + self.c[:,1] + self.c[:,2]  + self.c[:,3] + self.c[:,4]) #*self.eps_bubble #*self.vel*self.A_R
            self.c_cloud_wake = (self.c[:,5] + self.c[:,6] + self.c[:,7]  + self.c[:,8] + self.c[:,9])#*self.eps_cw #*self.vel*self.A_R
            self.c_emulsion = (self.c[:,10] + self.c[:,11] + self.c[:,12]  + self.c[:,13] + self.c[:,14])# *self.eps_emulsion #*self.vel*self.A_R
            self.c_solid = (self.c[:,15] + self.c[:,16] + self.c[:,17]  + self.c[:,18] + self.c[:,19]) #*self.eps_solids #*self.vel*self.A_R
            self.c_total = self.c_bubble + self.c_cloud_wake + self.c_emulsion # + self.c_solid 

            plt.subplot(224)
            plt.plot(self.z_c,self.c[:,20]/101325,label='Pressure')
            plt.plot(self.z_c, self.c_total*self.R_gas*self.T_in/101325, label = 'Sum of Pressures in phases')
            plt.legend()
            plt.grid()
            
            plt.pause(0.01)
        return
            
# if __name__ == "__main__":
# Steady state:
# reactor = fluidized_bed_RWGS(100, 2e-4, 1.0 ,np.inf, 0.22, 0, 900, 1*101325, Maxwell_Stefan=True, c_dependent_diffusion=True)
# reactor.solve()
# print(reactor.Conversion_CO2)
# print(f'CO2 Conversion in Fluidized Bed: {reactor.Conversion_CO2:.2f}')

# Transient:
# Temperature = 700
# X_CO_in = 0.35
# dt = 0.01
# Nodes = 100
# reactor = geldart_A_WGS(Nodes, 3e-4, 1.0, dt, 0.22, X_CO_in, Temperature, 1*101325, c_dependent_diffusion=True, plot=True)
# reactor.solve()

# Temp = []
# Conversion_CO = []
# temps = [400,450,500,550,600,650,700,750]
# # X_CO_in = 0.35

# X_CO_in = 0.35

# for temp in temps:
#     reactor = geldart_A_WGS(100, 3e-4, 1.0, np.inf, 0.22, X_CO_in, temp, 1*101325, c_dependent_diffusion=True, plot=False)
#     reactor.solve()
#     Conversion_CO.append(reactor.Conversion_CO)
#     Temp.append(temp)

# plt.figure(figsize = (8,6))
# plt.plot(temps,Conversion_CO, label = 'CO2 conversion')
# plt.grid()
# plt.xlabel('Reactor Temperature [K]') 
# plt.ylabel('Conversion [-]') 
# plt.title('WGS Equillibrium Conversion Obtained')
# plt.show()




# Temp = []
# # Inlet has CO2/H2
# Composition_CO2 = []

# Composition_CO = [] 
# temps = [100, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
# X_CO_in = 0.5

# for temp in temps:
#     reactor = geldart_A_RWGS(100, 3e-4, 1.0, np.inf, 0.22, X_CO_in, temp, 1*101325,  c_dependent_diffusion=False)
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
# plt.title('WGS Equillibrium Compositions Obtained')
# plt.show()