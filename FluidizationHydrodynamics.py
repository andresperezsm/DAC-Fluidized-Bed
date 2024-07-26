import numpy as np 
import scipy.sparse as sps 
import pymrm as mrm  
import matplotlib.pyplot as plt 
import scipy.optimize as opt
# from UmfClass import solve_umf

class FluidizationHydrodynamics:
    def init(self, d_p, u0, nz):
        # Reactor properties
        self.Tin = 298
        self.P = 1*101325 
        self.Lr = 1
        self.Dr = 0.5 # 10/self.Lr # Based on John's suggestion for area
        self.u0 = u0 # Superficial gas velocity at the inlet -> Check if interstitial velocity is relevant
        self.epsilon_b = 0.4 # Minimum fluidization porosity (-) assumed to be the same for a packed bed reactor
      
        self.L = self.Lr # Length of the reactor
        self.nz = nz # Number of grid points
        self.z = np.linspace(0,self.L,self.nz)

        # Gas phase properties
        self.mu_g = 1.825e-5 # Dynamic viscosity of Air at (298 K, 1 bar) (kg m-1 s-1)
        self.rho_g = 1.204 # Density of air at (293.15 K, 1 atm) (kg m-3)
        self.g = 9.81 # Gravitational acceleration (m^2/s)
        self.R = 8.31 # Gas Constant J mol-1 K-1

        # Particle phase properties from (Low 2023)
        self.d_p =  d_p # Lewatit VP OC 1065
        self.rho_s = 744 # Particle density (kg/m3)
        self.epsilon_s = 0.338 # Particle porosity (-)
        self.phi_p = 1 # Particle sphericity assumed to be 1 
        self.tauw = 2 # Particle tortuosity

        # Archimedes number
        self.Ar = (self.g*self.rho_g*(self.rho_s - self.rho_g)*(self.d_p**3))/((self.mu_g)**2)

        # Archimedes limits
        self.Ar_AB = (1.03e6)*(((self.rho_g)/(self.rho_s - self.rho_g))**(1.275))
        self.Ar_BD = (1.25e5)

        if self.Ar <= self.Ar_AB :
            self.particletype = 'Particle is Geldart A'
            print(self.Ar,'Particle is Geldart A type')

        elif self.Ar <= self.Ar_BD :
            self.particletype = 'Particle is Geldart B'
            print(self.Ar,'Particle is Geldart B type')

        else:
            raise ValueError("Particle is D type. Check Lewatit properties.")

        # Fluidization regimes
        self.Re = (self.rho_g*self.u0*self.d_p)/(self.mu_g)
        # print(self.Re)

        # Limits for bubbling fluidization (KL page 88-89)
        self.Re_bub_low = np.exp(-3.218 + 0.274*np.log(self.Ar))
        self.Re_bub_high = np.exp(-0.357 + 0.149*np.log(self.Ar))
        # print(self.Re, self.Re_bub_low ,self.Re_bub_high)

        # if self.Re >= self.Re_bub_high:
        #     raise ValueError("Reactor is not in bubbling fluidization regime (Re > Re_high). Check flow rate/particle size.")
        # elif self.Re <= self.Re_bub_low:
        #     raise ValueError("Reactor is not in bubbling fluidization regime (Re < Re_low). Check flow rate/particle size.")
        
        # Terminal particle velocity
        self.Re_terminal = (self.Ar**(1/3))/(((18)/(self.Ar**(2/3))) + ((2.335 - 1.744*self.phi_p)/(self.Ar**(1/6))))
        self.u_t = self.Re_terminal*self.mu_g/(self.d_p*self.rho_g)

        # Bed voidage at minimum fluidization
        self.T = 293.15 # Move to reactor properties after
        self.T0 = 298
        self.epsilon_mf = 0.382*(((self.Ar**(-0.196))*((self.rho_s/self.rho_g)**(-0.143))) + 1)*((self.T/self.T0)**(0.083))
        #[self.Re_mf, self.u_mf] = solve_umf(self.d_p)
        self.Re_mf = np.sqrt((33.7**2 + 0.408*self.Ar)) - 37 # From KL
        self.u_mf = (self.Re_mf*self.mu_g)/(self.rho_g*self.d_p)

        self.d_h = self.Dr # Hydraulic diameter of the column

        # The following are two options for the bubble diameter correlation

        """ OPTION 1"""
        # The following is the correlation from Cai et al. obtained from KL 
        # Porous plate
        self.db0_porous = (3.67 * 10**(-3)) * (self.u0 - self.u_mf)**2

        # Perforated plate
        Ac = 55e-6 # Size of the perforated holes from KL
        nd = 100
        self.db0_perforated = 0.347 * ((Ac * (self.u0 - self.u_mf) / nd) ** 0.4)

        self.d_b_func_1 = 0.138*(self.z**0.8)*((self.u0 - self.u_mf)**0.42)*np.exp(((self.u0 - self.u_mf)**2)*(-0.25/(10**5)) - ((self.u0 - self.u_mf)/(10**3)))
        # print(db0_porous,db0_perforated)
        self.d_b = np.maximum(self.d_b_func_1, self.db0_porous)


        """ OPTION 2"""
        # self.d_b_max = np.minimum(1.638*(((np.pi/4)*(self.d_h**2)*(self.u0 - self.u_mf))**(0.4)),self.d_h)
        # self.d_b0 = 0.376*((self.u0 - self.u_mf)**2)
        # self.d_b = self.d_b_max - (self.d_b_max - self.d_b0)*np.exp((-0.3*self.z)/(self.R*self.T)) # Mean bubble diameter from Mori & Wen KL

        # Bubble rise velocity
        self.u_br = np.zeros_like(self.d_b)

        for i in range(np.size(self.d_b)):
            if (self.d_b[i]/self.d_h) <= 0.125:
                self.u_br[i] = 0.711*(np.sqrt(self.g*self.d_b[i]))

            elif (self.d_b[i]/self.d_h) <= 0.6:
                self.u_br[i] = 0.8532*(np.sqrt(self.g*self.d_b[i]))*(np.exp(-1.49*(self.d_b[i]/self.d_h)))
            
            elif (self.d_b[i]/self.d_h) >= 0.6:
                self.u_br[i] = 0.35*(np.sqrt(self.g*self.d_b[i]))

            else:
                raise ValueError("The bubble to hydraulic diameter ratio is out of correlation bounds. This corresponds to less than 0.125 or more than 0.6. Check column dimensions.")

        # Bubble velocity
        # # First option specific for different types
        # u_b_type_A = 1.55*((Dt)**(0.32))*((u0 - u_mf) + 14.1*(d_b_func_FIRST_OPTION + 0.005))+ u_br # For Geldart A
        self.u_b_1 = 1.6*((self.Dr)**(1.35))*((self.u0 - self.u_mf) + 1.13*(self.d_b**(0.5))) + self.u_br # For Geldart B

        self.u_b = self.u0 - self.u_mf + self.u_br # 
        # print(self.d_b,self.u_b)

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

        self.Dm_CO2, self.Dm_N2, self.Dm_O2, self.Dm_Ar = self.calculate_average_diffusion_coefficients(self.Tin, self.P)
        
        self.kgs = self.k_gas_to_solid(correlation='Gunn')

        # print(self.u_b, self.d_b,self.Dm_CO2,self.Dm_N2,self.kgs)

        # self.Darray = np.array([self.Dm_CO2, self.Dm_N2, self.Dm_O2, self.Dm_Ar]) 
        self.Darray = np.array([self.Dm_CO2, self.Dm_N2])
        
        self.Kbc = np.zeros([self.nz, np.size(self.Darray)])
        self.Kce = np.zeros([self.nz, np.size(self.Darray)])
        self.Kov = np.zeros([self.nz, np.size(self.Darray)])
        self.Kbe = np.zeros([self.nz, np.size(self.Darray)])

        # Fix for index -> 4 initial indexes for Kbe, Kbc, Kce, Kov
        self.kgas = np.zeros([4,self.nz, np.size(self.Darray)])

        for i in range(np.size(self.Darray)):

            if self.particletype == 'Particle is Geldart A':
                self.Kbc[:,i] = 4.5*(self.u_mf/self.d_b[:]) + 5.85*(((self.Darray[i]**(1/2))*(self.g**(1/4)))/(self.d_b[:]**(5/4)))

                ## THE FACTOR 6.77 IS 13.56 IN ANOTHER CORRELATION
                self.Kce[:,i] = 6.77*((self.Darray[i]*self.epsilon_mf*self.u_br[:]/(self.d_b[:]**3))**(1/2))

            elif  self.particletype == 'Particle is Geldart B':
                self.Kbe[:,i] = np.sqrt((self.Darray[i]*self.u_b[:]*4)/(self.d_b[:]*np.pi)) + self.u_mf/3 # Correlation for a 2 phase Bubble to Emulsion model from ACRE exam

                # Using a combination and an overall transfer term
                self.Kbc[:,i] = 4.5*(self.u_mf/self.d_b[:]) + 5.85*(((self.Darray[i]**(1/2))*(self.g**(1/4)))/(self.d_b[:]**(5/4)))
                self.Kce[:,i] = 6.77*((self.Darray[i]*self.epsilon_mf*self.u_br[:]/(self.d_b[:]**3))**(1/2))
                self.Kov[:,i] = 1/((1/self.Kbc[:,i]) + (1/self.Kce[:,i]))
            else:
                raise ValueError("Paricle is D or C type. Spouting and jet formation is undesired, therefore, check Lewatit CP 1065 properties.")
            
        if self.particletype == 'Particle is Geldart A':
            self.kgas = np.array([self.Kbc, self.Kce])
        elif self.particletype == 'Particle is Geldart B':
            self.kgas[0,:,:] = self.Kbc[:,:]
            self.kgas[1,:,:] = self.Kce[:,:]
            self.kgas[2,:,:] = self.Kbe[:,:]
            self.kgas[3,:,:] = self.Kov[:,:]
       
        self.u_f = self.u_mf/self.epsilon_mf # Interstitial gas velocity at minimum fluidization

        self.fb = np.zeros_like(self.u_b)

        # Bubble phase fraction
        for i in range(np.size(self.u_b)):
            if (self.u_b[i]/self.u_f) <= 1:
                self.psi = 2

            elif (self.u_b[i]/self.u_f) == 1:
                self.psi = 1
            
            elif (self.u_b[i]/self.u_f) <= 5:
                self.psi = 0

            elif (self.u_b[i]/self.u_f) >= 5:
                self.psi = -1

            else:
                raise ValueError("The bubble to emulsion velocity ratio is out of correlation bounds. Check superficial (self.u0) and minimum fluidization velocity (self.u_mf)")
            
            self.fb[i] = (self.u0 - self.u_mf)/(self.u_b[i] + self.psi*self.u_mf)

        # Wake fraction inside of Bubble
        self.alpha_w = 1 - np.exp(-4.92*self.d_b)
        self.fw = self.alpha_w*self.fb

        # Cloud fraction 
        # There are two correlations for the cloud fraction per bubble volume. The following is from Medrano et al.
        self.RcRb = np.minimum(np.maximum(((self.u_br + 2*self.u_f)/(self.u_br - self.u_f)), 0), 1 - self.fb - self.fw)
        self.alpha_c = (self.RcRb)**3 - 1
        self.fc = self.alpha_c*self.fb # From Medrano et. al
        # print(self.fw,self.fb, self.alpha_c)

        # The following option is from KL
        # self.fc = 3/((self.u_br*self.epsilon_mf - self.u_mf)/self.u_mf)
        self.epsilon_cloud = self.fc*self.fb

        # Emulsion fraction 
        self.femulsion = (1 - self.fb)/self.fb # - self.fc - self.fw
        self.epsilon_emulsion = self.femulsion*self.fb # To obtain units per m3 of reactor volume

        """ # THIS ONE IS NOT ACCURATE """
        self.femulsion2 = (1 - self.fb - (self.fb*self.fc*self.fb*self.fw))/self.fb

        # Alternative to emulsion phase velocity from KL
        self.us_wake = self.u_b
        self.us_down = (self.fw*self.fb*self.u_b)/(1 - self.fb - self.fb*self.fw) 
        self.u_emulsion2 = ((self.u_mf)/(self.epsilon_mf)) - self.us_down

        # Emulsion phase velocity from Martin
        self.u_emulsion = (self.u0 - (((self.fb*(1 - self.epsilon_s)) + (self.fw*self.epsilon_mf))*self.u_b))/(self.femulsion*self.epsilon_mf)

        # print(self.u_emulsion, self.u_emulsion2)

        # Solid phase fractions per unit of bubble volume
        self.gamma_b = 0.005
        self.gamma_cw = (1 - self.epsilon_mf)*(self.fc + self.fw) # This one is from KL the alternative is the following (1 - self.epsilon_mf)*(self.fc + 0.5) 
        self.gamma_e = (1 - self.epsilon_mf)*((1 - self.fb)/self.fb) - self.gamma_b - self.gamma_cw

        # Solid axial dispersion from KL
        self.Dsv = ((self.fw**2)*self.epsilon_mf*self.fb*self.d_b*(self.u_b**2))/(3*self.u_mf)
        self.alpha = 0.77 # 1 # for fine Geldart A and AB solids from KL
        self.Dsh = (3*self.fb*self.u_mf*self.d_b*(self.alpha))/(16*(1 - self.fb)*self.epsilon_mf)


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
                    diffusion_coeff = self.fuller(T, M1, M2, V1, V2, P)*self.epsilon_s/self.tauw
                    diffusion_coeffs[name1].append(diffusion_coeff)
        
        # Calculate average diffusion coefficient for each molecule
        avg_diffusion_coeffs = {name: np.mean(coeffs) for name, coeffs in diffusion_coeffs.items()}
        # print(f"Average Diffusion Coefficients: {avg_diffusion_coeffs}")
        
        return avg_diffusion_coeffs['CO2'], avg_diffusion_coeffs['N2'], avg_diffusion_coeffs['O2'], avg_diffusion_coeffs['Ar']

    
    def k_gas_to_solid(self, correlation='Gunn'):
        pairs = [(self.M_CO2, self.M_O2, self.V_CO2, self.V_O2),    # 12
                    (self.M_N2, self.M_O2, self.V_N2, self.V_O2)]    # 13
        
        k_gs = []
        
        for M_i, M_j, V_i, V_j in pairs:
            D_ij = self.fuller(self.T,M_i, M_j, V_i, V_j,self.P)

            # Calculate Reynolds number
            Re = self.rho_g * self.u0 * self.d_p / self.mu_g

            # Calculate Schmidt number
            Sc = self.mu_g / (self.rho_g * D_ij)

            if correlation == 'Ranz-Marshall':
                k_mt = D_ij / self.d_p * (2 + 0.06 * Re**0.5 * Sc**(1/3))
            elif correlation == 'Gunn':
                k_mt = D_ij / self.d_p * ((7 - 10 * self.epsilon_b + 5 * self.epsilon_b**2) * (1 + 0.7 * Re**0.2 * Sc**0.33) + (1.33 - 2.4 * self.epsilon_b + 1.2 * self.epsilon_b**2) * Re**0.7 * Sc**0.33)
            else:
                raise ValueError("Use 'Ranz-Marshall' or 'Gunn'.")

            k_gs.append(k_mt)

        return k_gs
    