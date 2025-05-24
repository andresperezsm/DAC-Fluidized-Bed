import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# Plotting options

class PlotClass: 
    def __init__(self, plot_config=None):
        """
        Initialize the PlotClass with a reference to the ColumnModel instance.

        Parameters:
        - column_model (ColumnModel): The instance of ColumnModel to plot data from.
        """
        self.column_model = None    # To store the ColumnClass instance
        self.fig = None
        self.axs = None

        # Apply the plot configuration if provided, otherwise set defaults
        self.config = plot_config if plot_config is not None else {}

        # Plot configuration options with default values
        self.update_plot_toggle = self.config.get('update_plot_toggle', True)
        self.line_plot_toggle = self.config.get('line_plot_toggle', False)      # Default to 2D plotting unless specified
        self.delta_T_max = self.config.get('delta_T_max', 10)                   # Temperature max delta for plots
        self.y_max_multiplier = self.config.get('y_max_multiplier', 1.3)        # Default y_max multiplier
        self.normalized_concentrations = self.config.get('normalized_concentrations', False)
        self.re_ads_fact = self.config.get('re_ads_fact', 1)
        self.time_unit = self.config.get('time_unit', 'hr')
    
    # ------------------------- Counter-Current Solid plotting ----------------------- # 
    def init_solid_circulation_plot(self):
        plt.ion()  # Enable interactive mode for dynamic plotting
        plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 14,
        'font.serif': ['Times New Roman'],
        'axes.linewidth': 1.5,
        'lines.linewidth': 2.4  
        })
                
        # Axial positions
        z_c = self.column_model.z_c
        c = self.column_model.c

        # Determine maximum y-limits for the plots
        y_max = self.column_model.c_feed_plot.copy() * 1.3

         # Initialize subplots
        self.fig, axs = plt.subplots(1, 2, figsize=(15, 6))
        axs = [axs[0], axs[0].twinx(), axs[1], axs[1].twinx()]
        plt.subplots_adjust(left=0.1, right=0.9, wspace=0.3, hspace=0.4)

        # Configure left subplot for bubble-wake (bw)
        axs[0].set_title("Bubble-Wake Gas and Solid")
        axs[0].set_xlabel("Axial Position [m]")

        axs[0].set_ylabel("Gas Concentration")
        axs[0].set_ylim(0, y_max[..., 0])
        axs[0].plot(z_c, c[:, 0, 0], linestyle='-', color='tab:blue')  # Gas
        axs[0].spines['left'].set_color('tab:blue')
        axs[0].tick_params(axis='y', colors='tab:blue')

        axs[1].set_ylabel("Solid Concentration", color='tab:orange')
        axs[1].set_ylim(0, y_max[..., 1])
        axs[1].plot(z_c, c[:, 0, 1], linestyle='-', color='tab:orange')  # Solid
        axs[1].spines['right'].set_color('tab:orange')
        axs[1].tick_params(axis='y', colors='tab:orange')

        # Configure right subplot for emulsion (e)
        axs[2].set_title("Emulsion Gas and Solid")
        axs[2].set_xlabel("Axial Position [m]")
        axs[2].set_ylabel("Gas Concentration")
        axs[2].set_ylim(0, y_max[..., 2])
        axs[2].plot(z_c, c[:, 0, 2], linestyle='-', color='tab:blue')  # Gas
        axs[2].spines['left'].set_color('tab:blue')
        axs[2].tick_params(axis='y', colors='tab:blue')

        axs[3].set_ylabel("Solid Concentration", color='tab:orange')
        axs[3].set_ylim(0, y_max[..., 3])
        axs[3].plot(z_c, c[:, 0, 3], linestyle='-', color='tab:orange')  # Solid
        axs[3].spines['right'].set_color('tab:orange')
        axs[3].tick_params(axis='y', colors='tab:orange')

        self.axs_CO2 = axs  # Store the axes for later use
                    
    def update_solid_circulation_plot(self, c, t):
        """
        Update the solid circulation plots with new data.
        """

        # c_bw[..., 0] *= self.column_model.eps_bw_gas
        # c_bw[..., 1] *= self.column_model.eps_w_solid
        # c_e[..., 0] *= self.column_model.eps_e_gas
        # c_e[..., 1] *= self.column_model.eps_e_solid

        self.fig.suptitle(f"Column profiles (t = {round(t, 3)} s)", fontsize=18, y=0.98)

        # Update the bubble-wake plots
        self.axs_CO2[0].lines[0].set_ydata(c[:, 0, 0])  # Update line for CO2 bw_g
        self.axs_CO2[1].lines[0].set_ydata(c[:, 0, 1])  # Update line for CO2 bw_s

        # Update the emulsion plots
        self.axs_CO2[2].lines[0].set_ydata(c[:, 0, 2])  # Update line for CO2 e_g
        self.axs_CO2[3].lines[0].set_ydata(c[:, 0, 3])  # Update line for CO2 e_s

        # Redraw the figure to reflect the updates
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
