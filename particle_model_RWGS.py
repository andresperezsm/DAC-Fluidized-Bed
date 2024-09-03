import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import scipy.optimize as sopt
import pymrm as mrm
import time

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
###############################################################
###############################################################
####                                                       ####
####     ############  ####    ####      ###               ####
####     ############  ####    ####     ###  ######        ####
####         ####      ####    ####    ###  ###  ###       ####
####         ####      ####    ####   ###  ##########      ####
####         ####      ############  ###    ###            ####
####         ####       ##########  ###      ######        ####
####                                                       ####
###############################################################
###############################################################
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class particle_model_RWGS:
    
    def __init__(self,Nodes,d_p,dt,c_surface,T_surface,P_surface,c_dependent_diffusion=False):
        
        self.c_dependent_diffusion = c_dependent_diffusion
        
        # Simulation and field parameters
        self.Nr = Nodes # Number of grid cells in the radial direction [-]
        self.dt = dt # Lengt of a time step [-]
        self.t_end = 1.0 # End time of the simulation [s]
        
        if dt == np.inf:
            self.Nt = 1
        else:
            self.Nt = int(self.t_end/self.dt) # Number of time steps
            
        self.Nc = 5 # Number of components (temperature included) [-]
        
        self.d_p = d_p # Diameter of the particle [m]
        self.R_p = d_p/2 # Radius of the particle
        
        self.dr = self.R_p/self.Nr # Width of a grid cell [m]
        
        self.dr_large = 0.1 * self.R_p # Parameter needed for non uniform grid
        self.refinement_factor = 0.75 # Parameter needed for non uniform grid
        
        self.r_f = mrm.non_uniform_grid(0, self.R_p, self.Nr + 1, self.dr_large, self.refinement_factor) # Creates non uniform grid
        self.r_c = 0.5 * (self.r_f[1:] + self.r_f[:-1]) # Radial coordinates of the centers of the grid cells
        
        self.V_c = 4/3*np.pi*(self.r_f[1:]**3-self.r_f[:-1]**3) # Volume of a grid cell (shell of a sphere)
        self.V_c_validation = np.sum(self.V_c) # Sum of the shells of each grid cell (sum = total volume)
        self.V_p = 4/3*np.pi*self.R_p**3 # Volume of the sphere

        # Gas constants
        self.rho_g = 1.225 # Density of air [kg/m3]
        self.eta_g = 4e-5 # Viscosity of air [Pas s]
        self.Cp_g = 5e3 # Molar heat capacity of air [J/kg/K]
        self.lam_g = 0.03 # Thermal conductivity of air [W/m/K]
        self.R_gas = 8.314 # Gas constant [J/mol/K]
        self.D_Thermal = self.lam_g/self.rho_g/self.Cp_g # Thermal diffusivity of air [m2/s]
        self.P_surface = P_surface # Pressure in the reactor [Pa]
        
        # Catalyst parameters
        self.Cp_s = 451 # Molar heat capacity of iron [J/kg/K]
        self.Drad_s = 0.0 # Radial dispersion coefficient in the particle [m2/s]
        self.lam_s = 73 # Thermal conductivity of iron [W/m/K]
        self.rho_s = 4580 # Density of iron [kg/m3]
        self.eps_s = 0.5 # Catalyst porosity [-]
        self.tau = np.sqrt(2) # Catalyst tortuosity [-]
        
        # Reaction related parameters
        self.H_r = 42e3 # Heat of the reaction in [J/mol]
        self.T_surface = T_surface # Surface temperature [K]
        
        # Maxwell-Stefan related parameters
        self.M_CO2 = 44 # Molecular weight of CO2 [g/mol]
        self.M_H2 = 2 # Molecular weight of H2 [g/mol]
        self.M_CO = 28 # Molecular weight of CO [g/mol]
        self.M_H2O = 48 # Molecular weight of H2O [g/mol]

        self.V_CO2 = 26.7 # Diffusion volume of CO2 [m3]
        self.V_H2 = 6.12 # Diffusion volume of H2 [m3]
        self.V_CO = 18.0 # Diffusion volume of CO [m3]
        self.V_H2O = 13.1 # Diffusion volume of H2O [m3]

        self.D_CO2, self.D_H2, self.D_CO, self.D_H2O = self.calculate_average_diffusion_coefficients() # Diffusion coefficients [m2/s]
        
        self.D12, self.D13, self.D14, self.D23, self.D24, self.D34 = self.calculate_binary_diffusion_coefficients()
        self.D_eff_p_ms = np.array(self.calculate_binary_diffusion_coefficients())
        
        # Surface conditions
        self.C_CO2_surface = c_surface[0] # Surface concentration of CO2 [mol/m3]
        self.C_H2_surface = c_surface[1] # Surface concentration of H2 [mol/m3]
        self.C_CO_surface = c_surface[2] # Surface concentration of CO [mol/m3]
        self.C_H2O_surface = c_surface[3] # Surface concentration of H2O [mol/m3]
        
        self.T_surface = T_surface # Surface temperature [K]
        self.P_surface = P_surface # Surface pressure [Pa]
        
        # Initial conditions
        self.c_0 = np.array([1e-2, 1e-2, 1e-2, 1e-2, self.T_surface], dtype='float') # Initial conditions field
        
        # Boundary conditions
        self.c_in = np.array([self.C_CO2_surface, self.C_H2_surface, self.C_CO_surface, self.C_H2O_surface, self.T_surface], dtype='float') # surface conditions in an array for Dirichlet bc
        
        self.bc = {'a':[1,0], # Dirichlet boundary conditions
                   'b':[0,1], # Neumann boundary conditions
                   'd':[0,[self.c_in]]} # Values
    
        # Functions
        self.init_field() # Calls the function in the initial call
        self.init_Jac() # Calls the function in the initial call
    
    
    def Fuller_correlation(self,M_i, M_j, V_i, V_j):
        C = 1.013e-2
        D_ij = C*self.T_surface**1.75/self.P_surface * np.sqrt(1/M_i + 1/M_j) / (V_i**(1/3) + V_j**(1/3))**2
        return D_ij
    
    
    def calculate_average_diffusion_coefficients(self):
        diffusion_coeff = np.zeros((self.Nc-1,self.Nc-1))

        M = np.array([self.M_CO2,self.M_H2,self.M_CO,self.M_H2O])
        V = np.array([self.V_CO2,self.V_H2,self.V_CO,self.V_H2O])
        
        for i in range(self.Nc-1): # Takes component i
            for j in range(self.Nc-1): # Calculates binary diffusion of i with each other component j
                if i != j:
                    diffusion_coeff[i,j] = self.Fuller_correlation(M[i], M[j], V[i], V[j])*self.eps_s/self.tau # Calculates Dij

        avg_diffusion_coeff = np.zeros(self.Nc-1)

        for i in range (self.Nc-1):
            avg_diffusion_coeff[i] = np.sum(diffusion_coeff[i,:])/(self.Nc-2)
        
        return avg_diffusion_coeff[0], avg_diffusion_coeff[1], avg_diffusion_coeff[2], avg_diffusion_coeff[3]
        
    
    def calculate_binary_diffusion_coefficients(self):
        diffusion_coeff = np.zeros((self.Nc-1,self.Nc-1))
        M = np.array([self.M_CO2,self.M_H2,self.M_CO,self.M_H2O])
        V = np.array([self.V_CO2,self.V_H2,self.V_CO,self.V_H2O])
        
        diffusion_coeffs = []
        
        for i in range(self.Nc-1): # Takes component i
            for j in range(self.Nc-1): # Calculates binary diffusion of i with each other component j
                if i < j:
                    diffusion_coeff = self.Fuller_correlation(M[i], M[j], V[i], V[j])*self.eps_s/self.tau # Calculates Dij
                    diffusion_coeffs.append(diffusion_coeff)

        return diffusion_coeffs
    
    
    def init_field(self):        
        self.c = np.full([self.Nr, self.Nc], self.c_0, dtype='float')
    
    
    def reaction(self,c):
        f = np.zeros_like(c)
        T_p = c[:,-1]
        
        P_CO2 = c[:,0]*self.R_gas*T_p
        P_H2  = c[:,1]*self.R_gas*T_p
        P_CO  = c[:,2]*self.R_gas*T_p
        P_H2O = c[:,3]*self.R_gas*T_p
        
        k = 11101.2*np.exp(-117432/(self.R_gas*T_p)) # Rate constant [mol/s/gcat/bar]
        k_eq = np.exp(12.11 - 5319/T_p - 1.012*np.log(T_p) + 1.144*10**(-4*T_p))
        K_H2O = 96808*np.exp(-51979/(self.R_gas*T_p))
         
        r_RWGS = k*(P_CO2*P_H2 - P_CO*P_H2O/k_eq)/(P_H2 + K_H2O*P_H2O) * self.rho_s * 1000 / 101325 # Reaction rate [mol/m3cat/s]
        
        r_RWGS[np.isnan(r_RWGS)] = 0
        f[:,0] = - r_RWGS
        f[:,1] = - r_RWGS
        f[:,2] = + r_RWGS
        f[:,3] = + r_RWGS
        
        # f[:,-1] = (r_RWGS * (-self.H_r) + self.Q_joule) / (self.rho_s*self.Cp_s)
        
        return f
        
    
    def diffusion(self, c):
        
        x_1 = c[:,0]/(c[:,0]+c[:,1]+c[:,2]+c[:,3])
        x_2 = c[:,1]/(c[:,0]+c[:,1]+c[:,2]+c[:,3])
        x_3 = c[:,2]/(c[:,0]+c[:,1]+c[:,2]+c[:,3])
        x_4 = c[:,3]/(c[:,0]+c[:,1]+c[:,2]+c[:,3])

        D_field = np.empty_like(c)

        D_field[:,0] = (1-x_1)/(x_2/self.D12+x_3/self.D13+x_4/self.D14)
        D_field[:,1] = (1-x_2)/(x_1/self.D12+x_3/self.D23+x_4/self.D24)
        D_field[:,2] = (1-x_3)/(x_1/self.D13+x_2/self.D23+x_4/self.D34)
        D_field[:,3] = (1-x_4)/(x_1/self.D14+x_2/self.D24+x_3/self.D34)
        D_field[:,4] = self.D_Thermal

        self.D_field_ax = mrm.interp_cntr_to_stagg(D_field, self.r_f, self.r_c, axis=0)

        D_m = mrm.construct_coefficient_matrix(self.D_field_ax)
    
        return D_m    
    
    def init_Jac(self):
        if self.c_dependent_diffusion == True:
            self.Jac_accum= self.eps_s/self.dt*sps.eye_array(self.Nr*self.Nc)
            self.Grad, self.grad_bc = mrm.construct_grad(self.c.shape, self.r_f, self.r_c, self.bc, axis=0)
            self.Div = mrm.construct_div(self.c.shape, self.r_f, nu=2, axis =0)
        else:
            self.Jac_acc = self.eps_s*sps.eye_array(self.Nr*self.Nc)/self.dt
            self.Div_rad = mrm.construct_div(self.c.shape, self.r_f, nu=2, axis=0)
            self.Grad_rad, self.Grad_rad_bc = mrm.construct_grad(self.c.shape, self.r_f, self.r_c, self.bc, axis=0)
        
            self.D_rad_M = mrm.construct_coefficient_matrix([[self.D_CO2, self.D_H2, self.D_CO, self.D_H2O, self.D_Thermal]], self.c.shape, axis=0)

            self.Diff_rad, self.Diff_rad_bc = - self.D_rad_M @ self.Grad_rad, - self.D_rad_M @ self.Grad_rad_bc
            self.Flux_rad, self.Flux_rad_bc = self.Diff_rad, self.Diff_rad_bc
            self.Jac_Flux_rad, self.g_Flux_rad = - self.Div_rad @ self.Flux_rad, - self.Div_rad @ self.Flux_rad_bc
            
            self.Jac_const = self.Jac_acc - self.Jac_Flux_rad
            self.g_const = - self.g_Flux_rad
        
    
    def linearized_pde(self, c, c_old):
        if self.c_dependent_diffusion == True:
            f_react, Jac_react = mrm.numjac_local(self.reaction, c)
            Dax_m = self.diffusion(c)
            self.Flux = -Dax_m@self.Grad
            self.flux_bc = -Dax_m@self.grad_bc
            self.g_const = self.Div@self.flux_bc
            self.Jac_const = self.Jac_accum + self.Div@self.Flux

            g = self.g_const + self.Jac_const@c.reshape(-1,1)  - self.Jac_accum@c_old.reshape(-1,1) -f_react.reshape(-1,1)
            Jac = self.Jac_const-Jac_react 
        else:
            rea, Jac_rea = mrm.numjac_local(lambda c: self.reaction(c), c)

            Jac = self.Jac_const - Jac_rea
            g = - c_old.reshape(-1,1)/self.dt + self.Jac_const @ c.reshape(-1,1) + self.g_const - rea.reshape(-1,1) 
        return g, Jac


    def solve(self):
        if __name__ == "__main__":
            plt.figure(figsize=(8,5))
            
        t = 0
        
        for _ in range(self.Nt):
            c_old = self.c.copy()
            self.c = mrm.newton(lambda c: self.linearized_pde(c,c_old),c_old).x
            t += self.dt
            if __name__ == "__main__":
                if self.dt == np.inf: # Transient plot
                    plt.suptitle(f'Steady state',fontsize=20)
                    
                    plt.subplot(121)
                    labels = ['CO2','H2','CO','H2O']
                    for i in range(self.Nc-1):
                        plt.plot(self.r_c,self.c[:,i],label=labels[i])
                    plt.xlim(0,self.R_p)
                    plt.ylim(0,None)
                    plt.grid()
                    plt.legend()
                    
                    plt.subplot(122)
                    plt.plot(self.r_c,self.c[:,4],label='Temperature')
                    plt.xlim(0,self.R_p)
                    # plt.ylim(self.T_surface-2,self.T_surface)
                    plt.legend()

                    plt.grid()
                    plt.legend()
                    plt.show()
                    
                    # labels = ['$\mathrm{CO_2}$','$\mathrm{H_2}$','CO','$\mathrm{H_2O}$']
                    # for i in range(self.Nc-1):
                    #     plt.plot(self.r_c,self.c[:,i],label=labels[i])
                    # plt.xlim(0,self.R_p)
                    # plt.ylim(0,80)
                    
                    # plt.xlabel('Radius [m]',fontsize=16)
                    # plt.ylabel('Concentration [mol $\mathrm{m^{-3}}$]',fontsize=16)
                    # plt.xticks(np.linspace(0,0.0025,6),fontsize=12)
                    # plt.yticks(np.linspace(0,80,5),fontsize=12)
                    # plt.minorticks_on()

                    # plt.grid(which='major',linestyle='-')
                    # plt.grid(which='minor',linestyle=':')
                    # plt.legend(loc='upper left',fontsize=12)
                    # plt.show()
                    
                else:
                    plt.clf()
                    plt.suptitle(f'Time: {t:.2f} s',fontsize=20)
                    
                    plt.subplot(121)
                    labels = ['CO2','H2','CO','H2O']
                    for i in range(self.Nc-1):
                        plt.plot(self.r_c,self.c[:,i],label=labels[i])
                    plt.xlim(0,self.R_p)
                    plt.ylim(0,None)
                    plt.grid()
                    plt.legend()
                    
                    plt.subplot(122)
                    plt.plot(self.r_c,self.c[:,4],label='Temperature')
                    plt.xlim(0,self.R_p)
                    # plt.ylim(self.T_surface-2,self.T_surface)
                    plt.legend()
                    
                    plt.grid()
                    plt.legend()
                    plt.pause(0.01)
                
        # Calculate the flux for each component
        if self.c_dependent_diffusion == True:
            N_CO2 = - self.D_field_ax[-1,0] * (self.c[-1,0] - self.c[-2,0]) / (self.r_c[-1] - self.r_c[-2])
        else:
            N_CO2 = - self.D_CO2 * (self.c[-1,0] - self.c[-2, 0]) / (self.r_c[-1] - self.r_c[-2])
        
        # Determine the apparent reaction rate for each component
        self.r_app_CO2 = 6 * -N_CO2 / self.d_p

        f = self.reaction(self.c)
        self.r_CO2_surface = np.abs(f[-1,0])
        
        self.efficiency = self.r_app_CO2 / self.r_CO2_surface

        self.thiele_modulus = self.d_p/2*np.sqrt(self.r_CO2_surface/(self.D_CO2*self.c[-1, 0]))
        
        return self.efficiency, self.thiele_modulus, self.r_app_CO2
    
if __name__ == "__main__":
    # Concentration and temperature profile
    Nodes = 40
    dp = 5e-3
    
    # dt = 0.01 # Transient simulation
    dt = np.inf # Steady state simulation
    
    T_surface = 1000
    P_surface = 10*101325
    
    X_CO2_surface = 0.35
    X_H2_surface = 1 - X_CO2_surface
    
    C_CO2_surface = X_CO2_surface*P_surface/T_surface/8.314
    C_H2_surface = X_H2_surface*P_surface/T_surface/8.314
    
    c_surface = np.array([C_CO2_surface,C_H2_surface,0,0])
    
    Q_Joule = 0
    
    part = particle_model_RWGS(Nodes,dp,dt,c_surface,T_surface,P_surface,c_dependent_diffusion=False)
    eta, phi, rapp = part.solve()