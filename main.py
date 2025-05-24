from config import ColumnFactory
from plot import PlotClass

if __name__ == "__main__":

    model_config = {
        'model': 'Two-Phase Model',                     # 'Three-Phase Model' or 'Two-Phase Model'
        'constant_phase_fractions': True,              # True or False
        'mass_transf_corr': 'K&L',                      # 'Medrano', 'K&L', 'Grace', 'Xie', 'Hernandez-Jimenez'
    }

    ads_config = {
        'isotherm_CO2': 'Toth',                         # 'Toth', 'SB'
        'Toth_data': 'Young',                           # 'Shi', 'Young, 'Chimani'
        'kinetic_CO2': 'LDFRateCO2',                      # 'TothRate', 'LDF_rate'
        'isotherm_H2O': 'GAB',                          # 'GAB' is the only option
        'kinetic_H2O': 'LDF_rate',                      # 'LDF_rate' is the only option
    }

    plot_config = {
        'update_plot_toggle': True,                     # Update axial/radial profiles in time
        'line_plot_toggle': True,                       # Use 1D plot in 2D simulations: shows axial profiles at r=0 and r=R
        'normalized_concentrations': False,             # normalization with feed conditions (adsorption) or initial conditions (desorption)
        'delta_T_max': 30,                              # factor to scale the temperature y-axis
        'y_max_multiplier': 2,                          # factor to scale the y-axis for clearer rapresentation
        're_ads_fact': 15,                              # factor to scale the axis accounting for re-adsorption effects
        'time_unit': 'hr'                               # factor to convert time of breakthrough plot: 's', 'min', 'hr'
    }


    # Initialize the column data
    config = ColumnFactory(ads_config, model_config)

    # Initialize therodynamics and kinetic models for CO2
    isotherm_CO2 = config.get_isotherm_model('CO2')
    kinetic_CO2 = config.get_kinetic_model('CO2', isotherm_CO2)

    # Initialize therodynamics and kinetic models for CO2
    isotherm_H2O = config.get_isotherm_model('H2O')
    kinetic_H2O = config.get_kinetic_model('H2O', isotherm_H2O)

    # Initialize column model
    PackedBed = config.get_packed_bed_model(kinetics=[kinetic_CO2, kinetic_H2O], thermos=[isotherm_CO2, isotherm_H2O])
    
    # Initialize plotter
    Plotter = PlotClass(plot_config=plot_config)

    # Assign plotter to the column model
    PackedBed.set_plotter(Plotter)

    # Run the packed bed model simulation
    PackedBed.solve_transient_coupled()
    # PackedBed.plot_breakthrough(plot_exp_data=False)




    