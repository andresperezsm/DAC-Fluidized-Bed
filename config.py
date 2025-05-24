import numpy as np
from isotherm_models import TothIsotherm, SBIsotherm, GABIsotherm
from kinetic_models import LDFRateCO2, TothRate
from column_model import Column
from input_parameters import init_params_PB

class ColumnFactory:
    def __init__(self, ads_config, model_config):
        self.ads_config = ads_config
        self.model_config = model_config
        print(f'\n... INITIALIZING COLUMN MODEL...')

    def get_isotherm_model(self, gas_type):
        #* CO2
        if gas_type == 'CO2':
            if self.ads_config['isotherm_CO2'] == 'SB':
                if not self.ads_config['co_adsorption']:
                    raise ValueError("SB isotherm is only valid when co-adsorption is True.")
                return SBIsotherm(data=self.config['Toth_data'])
            elif self.ads_config['isotherm_CO2'] == 'Toth':
                return TothIsotherm(data=self.ads_config['Toth_data'])
            else:
                raise ValueError("Invalid isotherm for CO2. Choose 'SB' or 'Toth'.")
        #* H2O
        elif gas_type == 'H2O':
            if self.ads_config['isotherm_H2O'] == 'GAB':
                return GABIsotherm()
            else:
                raise ValueError("Invalid isotherm for H2O. Only 'GAB' is available.")
        raise ValueError(f"Unknown isotherm model for {gas_type}")
    
    def get_kinetic_model(self, gas_type, thermo_model, par=None):
        #* CO2
        if gas_type == 'CO2':
            if isinstance(thermo_model, TothIsotherm):
                if self.ads_config['kinetic_CO2'] == 'TothRate':
                    return TothRate(thermo=thermo_model)
                elif self.ads_config['kinetic_CO2'] == 'LDFRateCO2':
                    return LDFRateCO2(thermo=thermo_model)                
                else:
                    raise ValueError("Invalid CO2 kinetics for Toth isotherm. Choose 'Tothrate' or 'LDFRateCO2'.")  

    def get_packed_bed_model(self, kinetics, thermos):

        # print(f'Model Type = {self.model_config['model']}')

        return Column(init_params=init_params_PB, 
                      thermo=thermos, 
                      kin=kinetics, 
                      model_type=self.model_config)

