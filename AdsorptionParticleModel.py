import numpy as np 
import matplotlib.pyplot as plt 
import scipy.sparse as sps 
import pymrm  as mrm
import math
from FluidizationHydrodynamics import *

# Heterogeneous particle model for adsorption
class AdsorptionParticleModel:
    def __init__(self,Tin, dp):
        ## Gas constants
        self.Rho_g = 1  #Density of the gas
        self.eta_g = 4e-5 #Viscosity gas 
        self.Cp_g = 5e3 #Heat capacity gas
        self.lam_g = 0.03 #Thermal diffusivity
        self.Rg = 8.314  #Gas constant

        # For the Lewatit VP OC 1065 adsorbent particle
        """ IMPORTANT: TO BE CHANGED """
        self.lam_s = 70 # Thermal diffusivity NEEDED
        self.rho_s = 4580 # Solid density
        self.d_p = dp # Diameter of the particle 
        self.Cp_s = 0.45e3 # Heat capacity 
        self.Dax_s = 0.0
        self.eps_b = 0.4 # Packing in the bed at minimum fluidization -> should be epsilon_mf from FluidizationHydrodynamics
        self.eps_s = 0.5 # Porosity of the particle
        self.tauw = np.sqrt(2)
        self.Df_therm = self.lam_g/self.Rho_g/self.Cp_g # Thermal diffusivity of the particle
        self.d_pore = 25e-9 # Average pore diameter of the Lewatit particle 

        ## Maxwell-Stefan constants
        #Molecular weights 
        self.M_CO2 = 44  # 1 
        self.M_N2 = 2    # 2
        self.M_O2 = 32  # 3
        self.M_Ar = 39.95  # 4

        #Diffusion volumes
        self.V_CO2 = 26.7 
        self.V_N2 = 6.12
        self.V_O2 = 16.3 
        self.V_Ar = 16.2

        # Check how to define the heat transfer coefficient. Also, the use of epsilon values.
        self.lam_g = 0.03 # Thermal conductivity of air [W/m/K]
        self.R_gas = 8.314 # Gas constant [J/mol/K]
        self.lam_s = 73 # Thermal conductivity of Lewatit [W/m/K] REQUIRED FOR PARTICLE MODEL
        self.A = self.lam_s/self.lam_g 
        self.B = 1.25*((1-self.eps_b)/self.eps_b)**(10/9)
        self.gamma = 2/(1-self.B/self.A)*((self.A-1)/(1-self.B/self.A)**2*self.B/self.A*(np.log(self.A/self.B))-0.5*(self.B+1))
        self.lam_stat = self.lam_g*((1-np.sqrt(1-self.eps_b))+np.sqrt(1-self.eps_b)*self.gamma)
        
        self.Df_Thermal = self.lam_g/(self.Rho_g*self.Cp_g) # Thermal diffusivity of air [m2/s]
        self.Ds_Thermal = self.lam_s/(self.rho_s*self.Cp_s)


        ## Calculate the average diffusion coefficient for each molecule
        self.Tin = Tin # 298
        self.P = 1*101325 
        self.Dm_CO2, self.Dm_N2, self.Dm_O2, self.Dm_Ar = self.calculate_average_diffusion_coefficients(self.Tin, self.P)
        
        #Heat of reaction
        self.H_r = -81e3 #Heat of reaction 
        
        # These values were used for the verification
        self.D_val = 1e-6
        self.k_val = 5 

        #Simulation parameters
        self.Nr = 40
        self.Nc = 10 # 4 components and 1 temperatures for each solid and gas phase
        self.Tin = 298
        self.lam_s = 70 

        self.R = dp/2  # Radius of the particle
        self.dp = 2*self.R
        self.dt = np.inf  # Time step (infinity for steady state)
        self.dr_large = 0.1 * self.R  # Initial large grid spacing
        self.refinement_factor = 0.75  # Factor for refining the grid

        self.r_f = mrm.non_uniform_grid(0, self.R, self.Nr + 1, self.dr_large, self.refinement_factor)
        self.r_c = 0.5 * (self.r_f[:-1] + self.r_f[1:])

        self.P = 1*101325 
        self.X_CO2 = 0.04 #The actual figure is 0.0004
        self.X_O2 = 0.2095
        self.X_Ar = 0.0093 
        self.X_N2 = 1 - self.X_CO2 - self.X_O2 #- self.X_Ar
        self.Cin_N2 = self.P*self.X_N2/(self.Rg*self.Tin)
        self.Cin_CO2 = self.P*self.X_CO2/(self.Rg*self.Tin)

        self.c0 = np.array([0,0,0,0, self.Tin,0,0,0,0, self.Tin], dtype='float')
        self.cin = np.array([self.Cin_CO2, self.Cin_N2, 0, 0,  self.Tin,0, 0, 0, 0,  self.Tin], dtype='float')
    
        # Change to a mixed boundary condition the file is in the Analytical solution
        self.bc = {'a':[1,0],'b':[0,1], 'd':[0.0, [self.cin]]}

        self.init_field(self.c0)

        self.init_Jac()

        self.freq_out=100

    def fuller(self, T, M1, M2, V1, V2, P):
        D_ij = 1.013e-2 * T**1.75 / P * np.sqrt(1/M1 + 1/M2) / (V1**(1/3) + V2**(1/3))**2
        return D_ij

    def calculate_average_diffusion_coefficients(self, T, P):
        # Define the molecules and their properties
        molecules = [
            ('CO2', self.M_CO2, self.V_CO2),
            ('N2', self.M_N2, self.V_N2),
            ('O2', self.M_O2, self.V_O2),
            ('Ar', self.M_Ar, self.V_Ar)
        ]
        
        # Calculate diffusion coefficients for each molecule with every other molecule
        diffusion_coeffs = {mol[0]: [] for mol in molecules}
        
        for i, (name1, M1, V1) in enumerate(molecules):
            for j, (name2, M2, V2) in enumerate(molecules):
                if i != j:
                    diffusion_coeff = self.fuller(T, M1, M2, V1, V2, P)*self.eps_s/self.tauw
                    diffusion_coeffs[name1].append(diffusion_coeff)
        
        # Calculate average diffusion coefficient for each molecule
        avg_diffusion_coeffs = {name: np.mean(coeffs) for name, coeffs in diffusion_coeffs.items()}
        # print(f"Average Diffusion Coefficients: {avg_diffusion_coeffs}")
        
        return avg_diffusion_coeffs['CO2'], avg_diffusion_coeffs['N2'], avg_diffusion_coeffs['O2'], avg_diffusion_coeffs['Ar']


    def init_field(self, c0):
        self.c = np.full([self.Nr, self.Nc], c0, dtype='float')

    ## Reaction rate equations
    def reaction(self, c):
        # Mass transfer coefficients obtained from the Gunn correlation
        model = FluidizationHydrodynamics()
        k_gs = model.kgs
        kCO2 = k_gs[0]
        kN2 = k_gs[1]

        Ea = 0.6e3 # J/mol
        deltaH = 105.5e3 # J/mol
        deltaHc = 13.5e3 # J/mol
        b0 = 1.05e-13 # CHECK ORDER OF MAGNITUDE Unit is Pascal
        b0c = 1.03/10 # CHECK ORDER OF MAGNITUDE Unit is Pascal
        Rg = 8.31
        T0 = 298

        pCO2 =  self.R_gas*self.Tin*self.c[:,0] # Pressure in Pascal

        T = self.Tin
        q_inf0_constant = 2.56 # mmol/g
        q_infc_constant = 1.05 # mmol/g 

        b = b0*np.exp(deltaH/(Rg*T))
        bc = b0c*np.exp(deltaH/(Rg*T))

        tau0 = 0.343
        alpha = 1.39
        tau = tau0 + alpha*(1 - T0/T)
        
        chi = 1.63
        q_inf_0 = q_inf0_constant*np.exp(chi*(1 - (T/T0)))
        q_inf_c = q_infc_constant*np.exp(chi*(1 - (T/T0)))

        nphys = (q_inf_0*b*pCO2)/((1 + ((b*pCO2)**(tau)))**(1/tau))
        nchem = (q_inf_c*bc*pCO2)/(1 + bc*pCO2)
        eta = np.exp(-Ea/Rg*T)

        # print(tau,q_infc_constant,q_inf0_constant,b,bc)  
        # print(nphys,nchem,eta)

        q_equillibrium = nphys + nchem*eta
        
        self.r_p = self.d_p/2 # Particle radius
        self.r_c = self.d_pore/2 # Pore radius

        self.q0 = 2.56e-3 # Equillibrium surface concentration of the adsorbent particle (mol/g)
        self.k_f_CO2 = kCO2
        self.k_f_N2 = kN2
        self.c0 = pCO2/(self.R_gas*self.Tin)
        self.eps_p = self.eps_s

        self.Dknudsen_CO2 = (self.d_pore)*np.sqrt((8*self.R_gas*self.Tin)/(np.pi*self.M_CO2))
        self.DpCO2 = self.tauw*((1/self.Dm_CO2)+(1/self.Dknudsen_CO2))
        self.DmuCO2 = 1e5 # Micropore diffusion is currently neglected

        self.Dknudsen_N2 = (self.d_pore)*np.sqrt((8*self.R_gas*self.Tin)/(np.pi*self.M_CO2))
        self.DpN2 = self.tauw*((1/self.Dm_N2)+(1/self.Dknudsen_N2))/(self.eps_p)
        self.DmuN2 = 1e5 # Micropore diffusion is currently neglected

        self.k_LDF_CO2 = 1/((((self.r_p**2)*self.q0)/(15*self.eps_p*self.DpCO2*self.c0)) + ((self.r_c**2)/(15*self.DmuCO2)))
        # c[5] is the solid phase concentration
        r = self.k_LDF_CO2*(q_equillibrium - c[:,5])
        # r = 0
    
        T_p_gas = c[:,4]
        T_p_solid = c[:,9]


        # CO2 and N2 are the 2 adsorbed species
        pCO2 = c[:,0]*self.Rg*T_p_gas
        pN2 = c[:,1]*self.Rg*T_p_gas
        pO2 = c[:,2]*self.Rg*T_p_gas
        pAr = c[:,3]*self.Rg*T_p_gas
 
        # How is the intra particle mass and heat transfer between the different phases obtained? 
        Sh = 2
        Nu = 2

        #  Check if these are defined correctly? Also if the effective values should be considered
        
        """ THIS IS PROBABLY NOT DONE CORRECTLY"""
        kCO2 = Sh*self.d_p/self.Dm_CO2
        kN2 = Sh*self.d_p/self.Dm_N2
        mtCO2 = kCO2*(6/self.d_p)*(c[:,1] - c[:,0])
        mtN2 = kN2*(6/self.d_p)

        ht = Nu*self.d_p/self.Df_Thermal

        f = np.zeros_like(c)
        
        # r = k*(Pco2-(Pco*Ph2o)/(Ke*Ph2))/(1+K_H2O*Ph2o/Ph2)*self.rho_s*1e3/101325

        # r[np.isnan(r)]=0
        r = 0 # self.k_val*c
        # f = -r

        f[:,0] = +mtCO2
        f[:,1] = -mtCO2
        f[:,2] = 0    
        f[:,3] = 0
        f[:,4] = ht # -self.H_r*r/self.rho_s/self.Rho_g
 
        f[:,5] = -r - mtCO2
        f[:,6] = -r - mtN2
        f[:,7] = 0
        f[:,8] = 0
        f[:,9] = -self.H_r*r - ht# -self.H_r*r/self.rho_s/self.Rho_g

        return f 

    def init_Jac(self):
        self.Jac_accum = sps.diags([self.eps_s, self.eps_s, self.eps_s, self.eps_s, self.eps_s, (1-self.eps_s), (1-self.eps_s), (1-self.eps_s), (1-self.eps_s), self.rho_s*self.Cp_s*(1-self.eps_s)]*self.Nr, dtype='float', format='csc')/self.dt
        self.Grad, grad_bc = mrm.construct_grad(self.c.shape, self.r_f, self.r_c, self.bc, axis=0)
        self.Div = mrm.construct_div(self.c.shape, self.r_f, nu=2, axis =0)

        Dax_m = mrm.construct_coefficient_matrix([[self.Dm_CO2, self.Dm_CO2, self.Dm_CO2, self.Dm_CO2, self.Df_therm]], self.c.shape, axis=0)
        # Dax_m = mrm.construct_coefficient_matrix([[self.D_val]], self.c.shape, axis=0)
        self.Flux = -Dax_m@self.Grad
        self.flux_bc = -Dax_m@grad_bc
        self.g_const = self.Div@self.flux_bc
        self.Jac_const = self.Jac_accum + self.Div@self.Flux

    def lin_pde(self, c, c_old):
        f_react, Jac_react = mrm.numjac_local(self.reaction, c)
        g = self.g_const + self.Jac_const@c.reshape(-1,1)  - self.Jac_accum@c_old.reshape(-1,1) -f_react.reshape(-1,1)
        Jac = self.Jac_const-Jac_react 

        return g, Jac 
    
    def solve(self, nt):
        # self.plot_pre()
        for i in range(1, nt + 1):
            c_old = self.c.copy()
            result = mrm.newton(lambda c: self.lin_pde(c, c_old), c_old)
            self.c = result.x
            # if (i% self.freq_out==0):
            #     self.plot(i*self.dt)


        # # Calculate the flux for each component
        # N_CO2 = -self.Dm_CO2 * (self.c[-1, 0] - self.c[-2, 0]) / (self.r_c[-2] - self.r_c[-1])
        # N_H2 = -self.Dm_H2 * (self.c[-1, 1] - self.c[-2, 1]) / (self.r_c[-2] - self.r_c[-1])

        # # Determine the apparent reaction rate for each component
        # rapp_CO2 = 6 * N_CO2 / self.dp
        # rapp_H2 = 6 * N_H2 / self.dp


        # # Determine Weitz-Pater criterion for each component
        # Cwp_CO2 = rapp_CO2 * self.R**2 / self.c[-1, 0] / self.Dm_CO2
        # Cwp_H2 = rapp_H2 * self.R**2 / self.c[-1, 1] / self.Dm_H2

        # print(f"Weitz-Pater criterion for CO2: {Cwp_CO2}")
        # print(f"Weitz-Pater criterion for H2: {Cwp_H2}")

        # T_p = self.c[-1, 4]
        # K_H2O = 96808 * np.exp(-51979 / (self.Rg * T_p))
        # k = 11101.2 * np.exp(-117432 / (self.Rg * T_p))
        # Ke = np.exp(-(-12.11 + 5319 * T_p**(-1) + 1.012 * np.log(T_p) + 1.144 * 10**(-4 * T_p)))

        # Pco2 = self.c[-1, 0] * self.Rg * T_p
        # Ph2 = self.c[-1, 1] * self.Rg * T_p
        # Pco = self.c[-1, 2] * self.Rg * T_p
        # Ph2o = self.c[-1, 3] * self.Rg * T_p

        # # Calculate the rate at the surface of the particle
        # r = k * (Pco2*Ph2 - (Pco * Ph2o) / (Ke)) / (Ph2 + K_H2O * Ph2o ) * self.rho_s * 1e3 / 101325

        # eta = rapp_CO2 / r  # Calculate the efficiency for CO2
        # print(eta)

        N = -self.D_val*(self.c[-1, 0] - self.c[-2, 0]) / (self.r_c[-2] - self.r_c[-1])
        rapp = 6*N/self.dp 

        r = self.k_val*self.c[-1]

        self.thiele = self.d_p / 2 * np.sqrt(r / (self.D_val * self.c[-1, 0]))
        #self.thiele = self.R*np.sqrt(self.k_val/self.D_val)

        eta = rapp/r

        return eta, self.thiele

    def plot_pre(self):
        # plt.ion()
        self.fig1, self.ax1 = plt.subplots(figsize=(10,5))
        self.line1 = []
        labels = ['CO_2', 'N_2', 'O2', 'Ar']
        for i in range(0, 4):
            self.line1 += self.ax1.plot(self.r_c, self.c[:,i], label=labels[i])
        
        self.ax1.set_title(f'Time: {0.0:5.3f} s', fontsize=16)
        self.ax1.set_xlabel('Position [m]')
        self.ax1.set_ylabel('Concentration [mol/m3]')
        self.ax1.set_xlim(0, self.R)
        self.ax1.set_ylim(0, self.cin[0]*1.2)
        self.ax1.grid()
        self.ax1.legend()


        self.fig2, self.ax2 = plt.subplots(figsize=(10,5))
        self.line2 = []
        
        self.line2 += self.ax2.plot(self.r_c, self.c[:,4], '-', label='T')
        
        self.ax2.set_title(f'Time: {0.0:5.3f} s', fontsize=16)
        self.ax2.set_xlabel('Position [m]')
        self.ax2.set_ylabel('Temperature [k]')
        self.ax2.set_xlim(0, self.R)
        self.ax2.set_ylim()
        self.ax2.grid()
        self.ax2.legend()


    def plot(self, t):
        self.ax1.set_title(f'Time: {t:5.3f} s', fontsize=16)
        for i in range(0, 4): 
            self.line1[i].set_ydata(self.c[:,i])
        self.fig1.canvas.draw()
        self.fig1.canvas.flush_events()

        self.ax2.set_title(f'Time: {t:5.3f} s', fontsize=16)
        self.line2[0].set_ydata(self.c[:,4])    
        self.fig2.canvas.draw()
        self.fig2.canvas.flush_events()    



particle = AdsorptionParticleModel(298, 1e-3)
particle.solve(110)
print(np.size(particle.c[:,0:2]))
plt.plot(particle.r_c,particle.c[:,0:2])
plt.show()
# plt.pause(1e5)



# # T_in_list = [500, 550, 600, 650, 700, 750, 800, 900, 1000, 1100]
# T_in_list = [298.15, 353.15]
# dp_list = np.logspace(-5, -1, 20)

# print(dp_list)

# results = {T_in: {'efficiencies': [], 'thiele_moduli': []} for T_in in T_in_list}

# # Loop over the temperatures and particle diameters
# for Tin in T_in_list:
#     for dp in dp_list:
#         particle = AdsorptionParticleModel(Tin, dp)
#         efficiency, thiele = particle.solve(110)
#         results[Tin]['efficiencies'].append(efficiency)
#         results[Tin]['thiele_moduli'].append(thiele)

# def analytical(thiele):
#     eta = 3/thiele*(1/np.tanh(thiele)-1/thiele)

#     return eta 

# eta_analytical = []

# thiele_list = np.logspace(-2, 3, 100)
# for thiele in thiele_list:
#     eff = analytical(thiele)

#     eta_analytical.append(eff) 



# # Plot the efficiency vs. Thiele modulus for different temperatures
# plt.figure(figsize=(10, 6))
# for Tin in T_in_list:
#     plt.plot(results[Tin]['thiele_moduli'], results[Tin]['efficiencies'], marker='o', linestyle='-', label=f'T_in = {Tin} K',color = 'orange')

# plt.plot(thiele_list, eta_analytical, '--', label='Analytical', color = 'red')

# plt.xlabel('Thiele Modulus',fontsize = 16)
# plt.ylabel('Efficiency',fontsize = 16)
# plt.title('Efficiency vs. Thiele Modulus for Different Inlet Temperatures', fontsize = 20)
# plt.grid(True)
# plt.xscale('log')  # Use logarithmic scale for Thiele modulus
# # plt.xlim(0.01, 18)
# plt.legend()
# plt.show()




# thiele_array = np.logspace(-2,2,20)
# Bi_m = particle.k_mt*particle.d_p/particle.Deff
# #analytical_internal = (np.tanh(thiele_array)/thiele_array)

# # for i in range(np.size(Bi_m)):
# analytical_external = (3/(thiele_array*thiele_array))*((thiele_array*(1/np.tanh(thiele_array)) - 1) / (1 + (((thiele_array*(1/np.tanh(thiele_array)) - 1))/Bi_m)))
# #plt.loglog(thiele_array,analytical_internal, label = 'Internal Resistance')
# plt.loglog(thiele_array,analytical_external, label = 'Analyitical - External Resistance')
# plt.xlabel('Thiele Modulus')
# plt.ylabel('Efficiency')
# plt.title('Efficiency vs. Thiele Modulus for Different Inlet Temperatures')
# plt.grid(True)
# plt.xscale('log')  # Use logarithmic scale for Thiele modulus
# plt.legend()
# plt.show()
