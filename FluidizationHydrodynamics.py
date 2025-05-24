import numpy as np 
from scipy.optimize import fsolve

class Fluidization:
    """
    Class for modeling fluidization hydrodynamics in two-phase or three-phase fluidized bed reactors (FBRs).

    Attributes:
        Various physical and hydrodynamic properties of the reactor,
        gas, and particle phases are initialized and used for calculations.
    """
    
    def __init__(self, column_model=None):
        print(f"\n... Initializing fluidization hydrodynamics")

        # Gas phase properties
        self.g = 9.81 # Gravitational acceleration [m^2/s]

        # Acces column model class
        if not column_model:
            raise Exception('No valid column model provided') 
        else: 
            self.cm = column_model

        # Archimedes number and limits
        self.geldart_limit()

        # Calculate phase hold-up fractions
        if self.cm.const_phase_frac:
            self.constant_phase_fractions()
        else:
            raise Exception('varying phase fractions not implemented yet')
        
        # Calculate mass transfer coefficients
        if self.cm.const_phase_frac:
            self.calc_mass_tranf_coeffs_constant()

        else:  
            raise Exception('varying phase fractions not implemented yet')
        
    def geldart_limit(self):
        """
        Determine particle Geldart classification and calculate fluidization parameters.
        
        This method calculates the Archimedes number (Ar) to classify particle types based on 
        Geldart's classification (A, B, or D), and determines key fluidization properties such as 
        the minimum fluidization velocity, bed voidage at minimum fluidization, and terminal velocity.
        
        Raises:
        -------
        ValueError:
            - If the particle type falls into Geldart's 'D' category.
            - If the bed is not in the bubbling fluidization regime (u0 < u_mf).
        """
        print(f"\n... Calculating Geldart Limits")
        # Archimedes number
        self.Ar = (self.g * self.cm.rho_g_mix / self.cm.mu_g_mix ** 2) * (self.cm.rho_p - self.cm.rho_g_mix) * (self.cm.d_p ** 3)
        
        # Define Archimedes number limits for classification
        self.Ar_AB = 1.03e6 * ((self.cm.rho_g_mix / (self.cm.rho_p - self.cm.rho_g_mix)) ** 1.275)
        self.Ar_BD = 1.25e5

        if self.Ar <= self.Ar_AB :
            self.particletype = 'Geldart A'
            print(f'Particle is {self.particletype}')
        elif self.Ar <= self.Ar_BD :
            self.particletype = 'Geldart B' 
            print(f'Particle is {self.particletype}')
        else:
            raise ValueError("Particle is Geldart D type. Check particle properties.")

        # Calculate fluidization regimes
        self.Re = (self.cm.rho_g_mix * self.cm.v_sup * self.cm.d_p) / self.cm.mu_g_mix
        
        # Terminal particle velocity
        self.phi_sph = 1                                         
        self.u_t_star = 1 / ((18 / (self.Ar ** (2 / 3))) + ((2.335 - 1.744 * self.phi_sph) / (self.Ar ** (1 / 6))))
        self.u_t_KL = self.u_t_star * ((self.g * self.cm.mu_g_mix * (self.cm.rho_p - self.cm.rho_g_mix) / (self.cm.rho_g_mix ** 2)) ** (1 / 3))
        
        # Terminal velocity based on correlation
        self.Re_terminal = (self.Ar ** (1 / 3)) / (((18 / (self.Ar ** (2 / 3))) + ((2.335 - 1.744 * self.phi_sph) / (self.Ar ** (1 / 6)))))
        self.u_t = self.Re_terminal * self.cm.mu_g_mix / (self.cm.d_p * self.cm.rho_g_mix)
        print(f'Terminal velocity {self.u_t.item():0.2f} m/s')
        
        # Calculate bed voidage and minimum fluidization velocity 
        self.eps_mf = 0.586 * (self.Ar ** -0.029) * (self.cm.rho_g_mix / self.cm.rho_p) ** 0.021
        self.u_mf = self.cm.mu_g_mix / self.cm.rho_g_mix / self.cm.d_p * ( np.sqrt(27.2**2 + 0.0408 * self.Ar) - 27.2)  # Minimum fluidization vel. [m/s]

        print(f'Minimum fluidization velocity {self.u_mf.item():0.4f} m/s')
        print(f'Bed porosity at minimum fluidization {self.eps_mf.item():0.3f}')

        # Check if the bed is in the bubbling fluidization regime
        if (self.cm.v_sup - self.u_mf) <= 0:
            raise ValueError('The velocity is below the minimum fluidization value. The bed is not in the bubbling fluidization regime.')
        
        # Calculate hydraulic diameter of the column
        self.d_hyd = self.cm.R_b * 2
    
    def constant_phase_fractions(self):
        """
        Calculate phase and solid fractions in a three-phase fluidized bed model.
        
        Determines bed height, bubble velocity and diameter, and phase fractions 
        (bubble, cloud, and emulsion) based on fluidization conditions.
        """
        print(f"\n... Calculating constant phase fractions")

        # Solve for final bed height
        self.solve_H_final()
        print(f'Bed final height: {self.H_f.item():0.2f} m')
        
        # Average values taken at mid-column height
        self.d_b_avg = self.d_b_max - (self.d_b_max - self.d_b0) * np.exp(-0.3 * self.H_f/2 / (self.cm.R_b * 2))
        self.u_b_avg = self.cm.v_sup - self.u_mf + 0.711 * np.sqrt(self.g * self.d_b_avg)
        print(f'Average bubble velocity: {self.u_b_avg.item():0.3f} m/s')

        if self.cm.model_type == 'Two-Phase Model':
            # Calculate phase fractions   [m3phase/m3reactor]
            self.alpha_w = 1 - np.exp(-4.92*self.d_b_avg)    

            self.f_b = (self.cm.v_sup - self.u_mf) / self.u_b_avg  
            self.f_w = self.alpha_w * self.f_b
            self.f_e = 1 - self.f_b - self.f_w  
            
            print(f'Bubble fraction respect to reactor volume: {100*self.f_b.item():0.2f} %')
            print(f'Wake fraction respect to reactor volume: {100*self.f_w.item():0.2f} %')
            print(f'Emulsion fraction respect to reactor volume: {100*self.f_e.item():0.2f} %')

            # Gas fraction per unit of reactor
            self.gamma_bw_gas = self.f_b + self.f_w * self.eps_mf                     # [gas in the bubble + wake]
            self.gamma_e_gas = self.f_e * self.eps_mf                                 # [gas in the emulsion]

            print(f'Gas in bubble + wake respect to reactor volume: {self.gamma_bw_gas.item():0.3f}')
            print(f'Gas in emuslion respect to reactor volume: {self.gamma_e_gas.item():0.3f}')

            # Solid fraction per unit of reactor
            self.gamma_w_solid = self.f_w * (1 - self.eps_mf)
            self.gamma_e_solid = self.f_e * (1 - self.eps_mf)

            print(f'Solid in wake  respect to reactor volume: {self.gamma_w_solid.item():0.3f}')
            print(f'Solid in emulsion  respect to reactor volume: {self.gamma_e_solid.item():0.3f}')

            # Gas and solid phase velocities
            self.u_ge = (self.cm.v_sup - ( self.f_b + self.f_w * (self.eps_mf)) * self.u_b_avg)/((1 -  self.f_b -  self.f_w) * (self.eps_mf))

            self.u_se =  - self.f_w * (1 - self.eps_mf) * self.u_b_avg /((1 -  self.f_b -  self.f_w) * (1 - self.eps_mf))
            
            print(f'Gas emulsion velocity: {self.u_ge.item():0.3f} m/s')
            print(f'Solid emulsion velocity: {self.u_se.item():0.3f} m/s')

    def solve_H_final(self):
        """
        Calculate the final bed height (H_final) in a fluidized bed model.
        
        This method estimates `H_final` by solving a non-linear equation for the average bubble
        diameter and velocity at mid-height of the bed, using an iterative approach. The maximum 
        bubble diameter (`d_b_max`) and initial bubble parameters (`d_b0`, `u_b_0`) are calculated 
        based on input fluid and bed properties.
        
        Returns:
        --------
        float
            Final bed height, H_final [m].
        """
        self.cm.A_in = np.pi * (self.cm.R_b**2)
        
        # Height at minimum fluidization [m]
        self.H_mf = self.cm.L_b * (1 - 0.4) / (1 - self.eps_mf)
        print(f'Height at minimum fluidization {self.H_mf.item():0.2f} m')
        
        # Maximum bubble diameter [m], limited by the column diameter
        self.d_b_max = np.minimum((0.65 * (self.cm.A_in * (self.cm.v_sup - self.u_mf))**0.4), self.cm.R_b * 2)

        # Initial bubble diameter at porous distributor and bubble initial velocity
        self.d_b0 = 0.376 * (self.cm.v_sup - self.u_mf) ** 2
        print(f'Bubble initial diameter {self.d_b0.item():0.5f} m')
        print(f'Bubble max diameter: {self.d_b_max.item():0.5f} m')

        self.u_b_0 = self.cm.v_sup - self.u_mf + 0.711 * np.sqrt(self.g * self.d_b0)
        print(f'Bubble initial velocity {self.u_b_0.item():0.2f} m/s')

        # Function to solve for H_final
        def f(H_f):
                # Average bubble swarm diameter and velocity as function of H final 
                d_b_avg = self.d_b_max - (self.d_b_max - self.d_b0) * np.exp(-0.3 * H_f/2 /(self.cm.R_b * 2))
                u_b_avg = self.cm.v_sup - self.u_mf + 0.711 * np.sqrt(self.g * d_b_avg)

                c1 = 1 - (self.u_b_0 / u_b_avg) * np.exp(-0.275 / (self.cm.R_b * 2))
                c2 = ((self.cm.v_sup - self.u_mf) / u_b_avg) * (1 - np.exp(-0.275 / (self.cm.R_b * 2)))

                # Equation to solve: f(H) = 0
                return H_f - self.H_mf * (c1 / (c1 - c2))

        # Solve for H_final using an initial guess based on minimum fluidization height
        H_initial_guess = self.H_mf
        self.H_f = fsolve(f, H_initial_guess)[0]
            
    def calc_mass_tranf_coeffs_constant(self):
        """
        Calculate mass transfer coefficients for a three-phase fluidized bed reactor (FBR).
        
        Computes component-wise mass transfer coefficients (K_bc, K_ce) and stores them in `self.k_gas`
        for each component in `self.Darray`. The results are stored in a matrix where each row
        corresponds to different mass transfer coefficients: Kbe, Kbc, Kce, and Kov.
        """
        print(f"\n... Calculating constant mass transfer coefficients")

        # Initialize arrays to store mass transfer coefficients for each component
        num_components = np.size(self.cm.D_m)
        self.K_be = np.zeros(num_components)
        self.K_we = np.zeros(num_components)

        # Matrix to store all mass transfer coefficients (rows: [Kbe, Kbc, Kce, Kov])
        self.k_gas = np.zeros((4, num_components))
        
        # Loop over each component to calculate mass transfer coefficients
        for i in range(num_components):
            if self.cm.model_type == 'Two-Phase Model':
                # Bubble to emulsion mass transfer
                if self.cm.mass_transf_corr == 'Medrano':
                    self.K_be[i] = (4*2.6*self.cm.v_sup)/(np.pi*self.d_b_avg) 
                if self.cm.mass_transf_corr == 'Grace':
                    self.K_be[i]  == 1.5*(self.u_mf/self.d_b_avg) + 12*(((self.cm.D_m[i]*self.eps_mf*self.u_b_avg)/(self.d_b_avg**3))**(1/2))
                if self.cm.mass_transf_corr == 'Xie':
                    self.K_be[i] = np.sqrt(self.u_b_avg*(self.d_b_avg**(1.7)))*0.492*self.eps_mf # Xie correlation for 3D columns
                if self.cm.mass_transf_corr == 'Chiba':
                    alpha = 1 - np.exp(-4.92*self.d_b_avg)
                    self.fw = alpha*self.epsilon_b
                    self.K_be[i]  = (6.78/(1 - self.fw))*(((self.cm.D_m[i]*self.eps_mf*self.eps_mf*self.u_b_avg)/(self.d_b_avg**3))**(1/2))
                if self.cm.mass_transf_corr == 'Hernandez-Jimenez':
                    self.K_be[i] = (9*self.cm.v_sup)/(4*self.d_b_avg) # Correlation for a 2 phase Bubble to Emulsion model from ACRE exam
                if self.cm.mass_transf_corr == 'K&L':
                    # Using a combination and an overall transfer term
                    Kbc_KL= 4.5*(self.u_mf/self.d_b_avg) + 5.85*(((self.cm.D_m[i]**(1/2))*(self.g**(1/4)))/(self.d_b_avg**(5/4)))
                    Kce_KL= 6.77 * ((self.cm.D_m[i] * self.eps_mf * (0.711 * np.sqrt(self.g * self.d_b_avg) ) / (self.d_b_avg**3))**0.5)
                    self.K_be[i] = 1/((1/Kbc_KL) + (1/Kce_KL))
                
                # Wake to emulsion mass transfer
                if (self.cm.v_sup/self.u_mf) <= 3:
                    self.K_we[0] = (0.075 * (self.cm.v_sup - self.u_mf)) / (self.d_b_avg * self.u_mf)
                elif (self.cm.v_sup/self.u_mf) > 3:
                    self.K_we[0] = 0.15 / self.d_b_avg

        # Store K_bc and K_ce in the corresponding rows of k_gas
        self.k_gas = self.K_be  
        self.k_solid = self.K_we

        component_names = ["CO2"] 

        # Print Kbc, Kce, and Kov for each component
        print("Mass transfer coefficients (1/s) for each component:")
        for i, name in enumerate(component_names):
            print(
                f"{name}: "
                f"Kbe = {self.K_be[i]:0.2f}, "
                f"Kwe = {self.K_we[i]:0.2f}, "
            )


    