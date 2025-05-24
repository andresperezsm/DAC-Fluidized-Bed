import numpy as np
from abc import ABC, abstractmethod
from scipy.constants import gas_constant as R_gas

class Thermodynamics(ABC):
    @abstractmethod
    def __init__(self):
        """Initialize the thermodynamics class."""
        pass

    @abstractmethod
    def isotherm(self, p, T):
        """Compute the adsorption isotherm."""
        pass

class TothIsotherm(Thermodynamics):
    def __init__(self, data='Shi'):
        """Initialize the Toth isotherm model with parameters."""
        print("Selected CO2 isotherm equation: Toth")
        if data == 'Chimani':
            print(f"Selected Toth isotherm data: {data}")
            # Chimani 2024 (15 - 30 C) - (400 - 1300 ppm)
            self.qm = 1.81  # [mol/kg]
            self.T0 = 343.15  # [K]
            self.b0 = 169.62e-5  # [1/Pa]
            self.DH0 = -64020  # [J/mol]
            self.th0 = 0.96
            self.alpha = 0.34
            self.Young_b_fact = 1

        elif data == 'Young': 
            print(f"Selected Toth isotherm data: {data}")
            # Young 2019 (25 - 100 C) - (0 - 101325 Pa)
            self.qm = 4.86  # [mol/kg]
            self.T0 = 298.15  # [K]
            self.b0 = 2.85e-21  # [1/Pa]
            self.DH0 = -117789  # [J/mol]
            self.th0 = 0.209
            self.alpha = 0.523 
            self.Young_b_fact = np.exp(-self.DH0 / R_gas / self.T0)

        elif data == 'Shi':
            print(f"Selected Toth isotherm data: {data}")
            # Shi 2024 (15 - 40 C) - (0 - 1200 ppm)
            self.qm = 4.6416  # [mol/kg]
            self.T0 = 293.15  # [K]
            self.b0 = 39.182e-2  # [1/Pa]
            self.DH0 = -99636  # [J/mol]
            self.th0 = 0.23677
            self.alpha = 0.73530 
            self.Young_b_fact = 1

    def isotherm(self, p, T):
        """Compute the adsorption isotherm using the Toth equation."""
        p = np.clip(p, 0, None)  # Clip pressure to valid range
        b = self.b0 * np.exp(self.DH0 / R_gas / self.T0 * (1 - self.T0 / T)) * self.Young_b_fact
        th = self.th0 + self.alpha * (1 - self.T0 / T)
        q_eq = self.qm * b * p / (1 + (b * p) ** th) ** (1 / th)

        return q_eq, b, th
        
class GABIsotherm(Thermodynamics):
    def __init__(self):
        """Initialize the GAB isotherm model with parameters."""
        print("Selected H2O isotherm equation: GAB")
        self.qm = 3.538  # [mol/kg]
        self.C = 46817.71  # [J/mol]
        self.D = 0.02437  # [1/K]
        self.F = 55845.29  # [J/mol]
        self.G = -41.67  # [J/mol/K]
        self.a_lin = 57220  # [J/mol]
        self.b_lin = -44.38  # [J/mol/K]
        self.A_antoine = 8.07131  # [-]
        self.B_antoine = 1730.63  # [-]
        self.C_antoine = 233.426  # [°C]

    def antoine_pressure(self, T):
        """Compute the water saturation pressure via the Antoine equation."""
        return 133.322 * 10 ** (self.A_antoine - self.B_antoine / (self.C_antoine + (T - 273.15)))

    def isotherm(self, p, T):
        """Compute the adsorption isotherm using the GAB equation."""
        rh = p / self.antoine_pressure(T)
        rh = np.clip(rh, None, 0.9)  # Clip relative humidity to valid range

        E10 = self.a_lin + self.b_lin * T
        E29 = self.F + self.G * T
        E1 = self.C - np.exp(self.D * T)
        k = np.exp((E29 - E10) / (R_gas * T))
        c = np.exp((E1 - E10) / (R_gas * T))

        q_eq = (self.qm * k * c * rh) / ((1 - k * rh) * (1 + (c - 1) * k * rh))
        return q_eq, rh, None
    
class SBIsotherm(Thermodynamics):
    def __init__(self, data='Shi'):
        """Initialize the Toth isotherm model with parameters."""
        print("Selected CO2 isotherm equation: Toth")
        if data == 'Chimani':
            print(f"Selected Toth isotherm data: {data}")
            # Chimani 2024 (15 - 30 C) - (400 - 1300 ppm)
            self.qm = 1.81  # [mol/kg]
            self.T0 = 343.15  # [K]
            self.b0 = 169.62e-5  # [1/Pa]
            self.DH0 = -64020  # [J/mol]
            self.th0 = 0.96
            self.alpha = 0.34
            self.Young_b_fact = 1

        elif data == 'Young': 
            print(f"Selected Toth isotherm data: {data}")
            # Young 2019 (25 - 100 C) - (0 - 101325 Pa)
            self.qm = 4.86  # [mol/kg]
            self.T0 = 298.15  # [K]
            self.b0 = 2.85e-21  # [1/Pa]
            self.DH0 = -117789  # [J/mol]
            self.th0 = 0.209
            self.alpha = 0.523 
            self.Young_b_fact = np.exp(-self.DH0 / R_gas / self.T0)

        elif data == 'Shi':
            print(f"Selected Toth isotherm data: {data}")
            # Shi 2024 (15 - 40 C) - (0 - 1200 ppm)
            self.qm = 4.6416  # [mol/kg]
            self.T0 = 293.15  # [K]
            self.b0 = 39.182e-2  # [1/Pa]
            self.DH0 = -99636  # [J/mol]
            self.th0 = 0.23677
            self.alpha = 0.73530 
            self.Young_b_fact = 1

        # SB from Chimani 2024
        self.gamma = 0.027
        self.beta = 0.061
        
    def isotherm(self, p, q_H2O, T):
        """Compute the adsorption isotherm using the SB Toth equation."""
        b = self.b0 * np.exp(self.DH0 / R_gas / self.T0 * (1 - self.T0 / T)) * self.Young_b_fact
        b *= (1 + self.beta * q_H2O)
        th = self.th0 + self.alpha * (1 - self.T0 / T)
        qm = self.qm / (1 - self.gamma * q_H2O)
        q_eq = qm * b * p / (1 + (b * p) ** th) ** (1 / th)
        return q_eq, qm, b, th
    

