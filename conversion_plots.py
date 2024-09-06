from geldart_B_WGS import *
from geldart_A_WGS import *
from FluidizationHydrodynamics import *

""" vel_gas = 1.5"""
""" u_mf = 1 in Geldart A"""
""" phase fractions being negative at vel_gas = 1.5"""
""" higher temperatures"""
""" LOWER dt"""

# # Transient:geldart A
# Temperature = 700
# X_CO_in = 0.35
# reactor = geldart_B_WGS(100, 3e-4, 1.0, 0.01, 0.22, X_CO_in, Temperature, 1*101325, c_dependent_diffusion=True, plot=True)
# reactor.solve()

# # Transient: geldart B
# from geldart_B_WGS import *
# Temperature = 750
# X_CO_in = 0 # 0.35
# dt = np.inf
# Nodes = 100
# reactor = geldart_B_WGS(Nodes, 3e-4, 1.0, dt, 1.5, X_CO_in, Temperature, 30*101325, c_dependent_diffusion=True, plot=True)
# reactor.solve()

""" geldart A conversion"""
# Temp = []
# Conversion_CO = []
# temps = [400,450,500,550,600,650,700] # 750
# # temps = [400,500,700]
# X_CO_in = 0.35
# vel_gas = 0.22 # 1.5
# for temp in temps:
#     reactor = geldart_A_WGS(100, 3e-4, 2.0, np.inf, vel_gas, X_CO_in, temp, 30*101325, c_dependent_diffusion=True, plot=False)
#     reactor.solve()
#     Conversion_CO.append(reactor.Conversion_CO)
#     Temp.append(temp)

# print(reactor.umf)
# plt.figure(figsize = (8,6))
# plt.plot(temps,Conversion_CO, label = 'CO2 conversion')
# plt.grid()
# plt.xlabel('Reactor Temperature [K]') 
# plt.ylabel('Conversion [-]') 
# plt.title('Geldart A - WGS Equillibrium Conversion Obtained')
# plt.show()

""" geldart B conversion"""
# from geldart_B_WGS import *
# Temp = []
# Conversion_CO = []

# H2O_CO_ratio = 2.5
# X_CO_in = 1/ (1 + H2O_CO_ratio)
# temps = [273.15 + 180 ,273.15 + 200, 273.15 + 220, 273.15 + 300, 273.15 + 350, 273.15 + 400]
# Lr = 1.2 # Reactor length
# Dr = 35e-3 # 35e-3 # Diameter
# d_p = 300e-6
# Nodes = 100
# rho_s = 1137

# GHSV = np.array([800])
# V_reactor = np.pi*Lr*Dr*Dr/4
# vel = (GHSV*V_reactor/(np.pi*Dr*Dr/4))/3600 # 
# for temp in temps:
#     reactor = geldart_B_WGS(Nodes, d_p, rho_s, Lr, Dr, np.inf, vel, X_CO_in, temp, 30*101325, c_dependent_diffusion=True, plot=False)
#     reactor.solve()
#     Conversion_CO.append(reactor.Conversion_CO)
#     Temp.append(temp)

# plt.figure(figsize = (8,6))
# plt.plot(temps,Conversion_CO, label = 'CO conversion')
# plt.grid()
# plt.xlabel('Reactor Temperature [K]') 
# plt.ylabel('Conversion [-]') 
# plt.title('Geldart B - WGS Equillibrium Conversion Obtained')
# plt.show()

""" EXPERIMENTAL geldart B conversion"""
# # Steam to CO ratio of 2.5
H2O_CO_ratio = 2.5
X_CO_in = 1/ (1 + H2O_CO_ratio)
Temps = [273.15 + 180 ,273.15 + 200, 273.15 + 220, 273.15 + 300, 273.15 + 350, 273.15 + 400]

# Experimental Data
experimental_data = [
    [0.398, 0.214, 0.169, 0.146, 0.114, 0.103],  # experimental_1
    [0.474, 0.269, 0.213, 0.176, 0.114, 0.061], # experimental_2
    [0.623, 0.415, 0.418, 0.321, 0.3014, 0.279], # experimental_3
    [0.868, 0.83, 0.742, 0.6321, 0.5211, 0.447], # experimental_4
    [0.833, 0.731, 0.621, 0.535, 0.516, 0.384], # experimental_5
    [0.755, 0.744, 0.703, 0.669, 0.617, 0.577] # experimental_6
]

Lr = 1.2 # Reactor length
Dr = 35e-3 # 35e-3 # Diameter
d_p = 300e-6
Nodes = 100
rho_s = 1137

GHSV = np.array([800,1600,2400,3200,4000,4800])
V_reactor = np.pi*Lr*Dr*Dr/4
Vels = (GHSV*V_reactor/(np.pi*Dr*Dr/4))/3600 #
print(Vels)
# Vels = [0.2, 0.25, 0.5, 0.75, 1, 1.25] # 1.5

# Initialize the model
model = FluidizationHydrodynamics()
model.init(d_p, Vels[0], Nodes, rho_s)
print(model.u_mf, model.particletype)

fig, ax = plt.subplots(3, 2, figsize=(15, 10))
plot_index = 0

for i, Temp in enumerate(Temps):
    Velocity = []
    Conversion_CO = []
    Fraction_H2 = []
    Fraction_CO = []
    
    row = plot_index // 2
    col = plot_index % 2
    
    for Vel in Vels:
        reactor = geldart_B_WGS(Nodes, d_p, rho_s, Lr, Dr, np.inf, Vel, X_CO_in, Temp, 30*101325, c_dependent_diffusion=True, plot=False)
        reactor.solve()
        Conversion_CO.append(reactor.Conversion_CO)
        Fraction_H2.append(reactor.Fraction_H2)
        Fraction_CO.append(reactor.Fraction_CO)
        Velocity.append(Vel)
    
    # Plot model results
    ax[row, col].plot(Velocity, Conversion_CO, 'o-', label='CO Conversion - Model')
    
    # Plot corresponding experimental data
    experimental_i = experimental_data[i]  # Use the correct experimental data for the current temperature
    ax[row, col].plot(Vels, experimental_i, 'x-', label='CO Conversion - Experimental')
    
    ax[row, col].set_xlabel('Velocity [m/s]') 
    ax[row, col].set_ylabel('Conversion [-]') 
    ax[row, col].set_title(f'Geldart B - WGS Conversion at {Temp:.2f} K')
    ax[row, col].grid()
    ax[row, col].legend()
    ax[row,col].set_ylim((0, 0.9))

    plot_index += 1

plt.tight_layout()
plt.show()







# # plt.figure(figsize = (8,6))
# # plt.plot(Velocity,Conversion_CO, 'o-', label = 'CO conversion')
# # plt.grid()
# # plt.xlabel('Velocity [m/s]') 
# # plt.ylabel('Conversion [-]') 
# # plt.title('Geldart B - WGS Equillibrium Conversion Obtained')
# # plt.show()