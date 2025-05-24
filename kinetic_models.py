import numpy as np
from abc import ABC, abstractmethod
from scipy.constants import gas_constant as R_gas

class Kinetics(ABC):
    @abstractmethod
    def __init__(self, thermo):
        """Initialize the kinetics class."""
        if thermo is None:
            raise ValueError("Thermodynamic model must be provided.")
        self.thermo = thermo

    @abstractmethod
    def ads_rate(self, p, q, T=None):
        """Compute the reaction rate."""
        pass

class TothRate(Kinetics):
    def __init__(self, thermo, ):
        """Initialize the Toth kinetics model with parameters."""
        super().__init__(thermo)  
        
        print("Selected adsorption rate equation: Toth rate")
        
        # Set kinetic parameters
        self.k0 = np.exp(15.915867)  # [mol/kg/Pa/s]
        self.Eact = 7.459580 * R_gas * 1000  # [J/mol]

    def ads_rate(self, c, q, T=None):
        """Compute the adsorption rate using the Toth rate equation."""
        
        q_eq, b, th = self.thermo.isotherm(c * R_gas * T, T)

        k = self.k0 * np.exp(-self.Eact / R_gas / T)
        r =  k * (c * R_gas * T * (1 - (q / self.thermo.qm) ** th) ** (1 / th) - q / self.thermo.qm / b)

        return r
    
class LDFRateCO2(Kinetics):
    def __init__(self, thermo=None):
        """
        Initialize the LDF kinetic model parameters.
        """
        super().__init__(thermo)
        self.k_LDF = 100 # [1/s]

        print(f'Selected H2O rate equation: LDF rate\n')

    def ads_rate(self, c, q, T=None):
        """ Compute the adsorption rate using the LDF rate equation model. """       

        q_eq, _, _ = self.thermo.isotherm(c * R_gas * T, T)
        q_eq = np.clip(q_eq, 0, self.thermo.qm)
        
        r = self.k_LDF * (q_eq - q)

        return r

    

    
