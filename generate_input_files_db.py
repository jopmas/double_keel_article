import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.interpolate import interp1d
from scipy.interpolate import interp2d
from matplotlib.pyplot import cycler
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.patches import Patch
import xarray as xr

path = os.getcwd().split('/')
machine_path = '/'+path[1]+'/'+path[2]

###############################################################################################################################################
#Functions
###############################################################################################################################################

def find_nearest(array, value):
    '''Return the index in array nearest to a given value.
    
    Parameters
    ----------
    
    array: array_like
        1D array used to find the index
        
    value: float
        Value to be seached
    '''
    
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def read_params(fpath):
    '''
    Read Nx, Nz, Lx, Lz from param.txt
    '''
    with open(fpath+"param.txt","r") as f:
        line = f.readline()
        line = line.split() #split by space a string to a list of strings
        Nx = int(line[-1])
        
        line = f.readline()
        line = line.split()
        Nz = int(line[-1])

        line = f.readline()
        line = line.split()
        Lx = float(line[-1])

        line = f.readline()
        line = line.split()
        Lz = float(line[-1])

    return Nx, Nz, Lx, Lz

def read_data(prop, step, Nz, Nx, fpath):
    '''
    Read and reshape readed data according to parameters to return a (Nx, Nz) array
    '''
    
    #build filename
    filename = fpath + prop + "_" + str(step) + ".txt"

    data = np.loadtxt(filename, skiprows=2, unpack=True, comments="P")
    data = np.reshape(data, (Nz, Nx))
    
    return data

def calc_mean_temperaure_region(data, Nz, xx, begin, end):
    '''
    This funcition select a region in x direction in a 2D array and calculates the horizontal mean

    Parameters
    ----------

    data: `numpy.ndarray`

    Nz: int
        Number of points in Z direction

    xx: numpy.ndarray
        2D grid with x cordinates

    begin: float
        Start point

    end: float
        End point

    Returns
    -------
    arr: `numpy.ndarray`
        Array containing the horizontal mean of selected region
    '''

    x_region = (xx >= begin) & (xx <= end)
    Nx_aux = len(x_region[0][x_region[0]==True])
    data_sel = data[x_region].reshape(Nz, Nx_aux)
    data_sel_mean = np.mean(data_sel, axis=1)
    
    return data_sel_mean

###############################################################################################################################################
#Customizing matplotlib 
label_size=18
plt.rc('xtick', labelsize=label_size)
plt.rc('ytick', labelsize=label_size)

#Install the following package from (https://www.fabiocrameri.ch/colourmaps/) for inclusive color palletes
#or comment set crameri_colors as False

# crameri_colors=True
crameri_colors=False
if(crameri_colors):
    from cmcrameri import cm as cr
    def get_cycle(cmap, N=None, use_index="auto"):
        if isinstance(cmap, str):
            if use_index == "auto":
                if cmap in ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                            'Dark2', 'Set1', 'Set2', 'Set3',
                            'tab10', 'tab20', 'tab20b', 'tab20c']:
                    use_index=True
                else:
                    use_index=False
            cmap = matplotlib.cm.get_cmap(cmap)
        if not N:
            N = cmap.N
        if use_index=="auto":
            if cmap.N > 100:
                use_index=False
            elif isinstance(cmap, LinearSegmentedColormap):
                use_index=False
            elif isinstance(cmap, ListedColormap):
                use_index=True
        if use_index:
            ind = np.arange(int(N)) % cmap.N
            return cycler("color",cmap(ind))
        else:
            colors = cmap(np.linspace(0,1,N))
            return cycler("color",colors)

    n_colors = 10
    # plt.rcParams["axes.prop_cycle"] = get_cycle(cr.romaO, n_colors)
    # plt.rcParams["axes.prop_cycle"] = get_cycle(cr.oslo, n_colors)
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cr.batlowKS(np.linspace(0, 1, 10)))
    #From Color Universal Design (CUD): https://jfly.uni-koeln.de/color/
    # plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7", "#000000"])
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", ['#88CCEE', '#44AA99', '#117733', '#332288', '#DDCC77', '#999933','#CC6677', '#882255', '#AA4499', '#585858'])
    


###############################################################################################################################################
#Setting the kind of tectonic scenario and number of cores
###############################################################################################################################################

scenario_kind = 'double_keel'

experiemnts = {
               'double_keel': 'Double Cratonic Keel',
               }

# ncores = 20
# ncores = 64
ncores = 128

###############################################################################################################################################
# Domain and interfaces
###############################################################################################################################################

Lx = 1600 * 1.0e3
# total model vertical extent (m)
Lz = 400 * 1.0e3
# number of points in horizontal direction
Nx = 801
# number of points in vertical direction
Nz = 201

# thickness of sticky air layer (m)
thickness_sa = 40 * 1.0e3
#thickness of sediments (m)
thickness_sed = 3 * 1.0e3
# thickness of decolement (m)
thickness_decolement = 1 *1.0e3
# thickness of upper crust (m)
thickness_upper_crust = 21 * 1.0e3
# thickness of lower crust (m)
thickness_lower_crust = 10 * 1.0e3
#Thickness of non cratonic lithosphere
thickness_mlit = 80 * 1.0e3
#thickness of cratonic lihosphere (upper)
thickness_mlit_crat_up = 80 * 1.0e3
#thickness of cratonic lithosphere (bottom)
thickness_mlit_crat_bot = 125 * 1.0e3

# total thickness of lithosphere (m)
thickness_litho = thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickness_mlit #125 km - reference is the non-cratonic lithosphere
thickness_crat_up = thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickness_mlit_crat_up
thickness_crat_bot = thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickness_mlit_crat_up + thickness_mlit_crat_bot
thickness_astnc = Lz - (thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickness_mlit)
thickness_astc = Lz - (thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickness_mlit_crat_up + thickness_mlit_crat_bot)

# seed depth bellow base of lower crust (m)
seed_depth = 3 * 1.0e3 #9 * 1.0e3 #original

x = np.linspace(0, Lx, Nx)
z = np.linspace(Lz, 0, Nz)
X, Z = np.meshgrid(x, z)

interfaces = {
        "litho_crat_bot": np.ones(Nx) * (thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickness_mlit_crat_up + thickness_mlit_crat_bot), #lower portion - this interface starts from the base of cratonic region
        "litho_crat_up": np.ones(Nx) * (thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickness_mlit_crat_up), #upper portion - this interface starts from the base of cratonic region
        "litho_nc": np.ones(Nx) * (thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust), #non cratonic lithosphere - this interface starts from the base of lower crust
        "lower_crust": np.ones(Nx) * (thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust),
        "seed_base": np.ones(Nx) * (thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust - seed_depth),
        "seed_top": np.ones(Nx) * (thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust - seed_depth),
        "upper_crust": np.ones(Nx) * (thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust),
        "decolement": np.ones(Nx) * (thickness_sa + thickness_sed + thickness_decolement),
        "sediments": np.ones(Nx) * (thickness_sa + thickness_sed),
        "air": np.ones(Nx) * (thickness_sa),
        }

#Building non cratonic lithosphere
dx = Lx/(Nx-1)
L_nc = 800.0e3 #m
N_nc = int(L_nc//dx)
n_retreat = 0 #points in x direction to retreat the left protion of cratonic lithosphere
L_retreat = n_retreat * dx #length of retreat

thickening = thickness_mlit #m
interfaces['litho_nc'][Nx//2 - N_nc//2 - n_retreat: Nx//2 + N_nc//2] = thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust + thickening

#Building lower craton
L_thinning = L_nc
N_thinning = N_nc

# thinning = thickness_mlit_crat_up + thickness_mlit_crat_bot #reach the base of the lower crust
thinning = thickness_mlit_crat_bot #reach the base of the lower crust

interfaces['litho_crat_bot'][Nx//2 - N_thinning//2 - n_retreat : Nx//2 + N_thinning//2] = thickness_sa + thickness_crat_bot - thinning

#Building upper craton
thinning = 0.0#thickness_mlit_crat_up
interfaces['litho_crat_up'][Nx//2 - N_thinning//2: Nx//2 + N_thinning//2] = thickness_sa + thickness_crat_up - thinning

#Building seed
# seed thickness (m)
thickness_seed = 6 * 1.0e3
# seed horizontal position (m)
# x_seed = 800 * 1.0e3
x_seed = Lx / 2.0
# x_seed = Lx / 2.0 + 200.0e3
# seed: number of points of horizontal extent
n_seed = 6

interfaces["seed_base"][int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)] = thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust - seed_depth + thickness_seed // 2

interfaces["seed_top"][int(Nx * x_seed // Lx - n_seed // 2) : int(Nx * x_seed // Lx + n_seed // 2)] = thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust - seed_depth - thickness_seed // 2

##############################################################################
#Rheological and Thermal parameters
##############################################################################

#Viscosity scale factor
C_air = 1.0
C_sed = 1.0
C_dec = 0.1
C_upper_crust = 1.0
C_lower_crust = 100.0
C_seed = 0.1
C_mlit = 1.0
C_mlit_uc = 1.0
C_mlit_lc = 1.0
C_ast = 1.0

#density (kg/m3)
rho_air = 1.0
rho_sed = 2700.0
rho_dec = 2350.0
rho_upper_crust = 2700.0
rho_lower_crust = 2800.0
rho_seed = 2800.0
rho_mlit = 3330.0 #3354.0 #phanerozoic
rho_mlit_uc = 3330.0 #3310.0 #archean
rho_mlit_lc = 3330.0 #proterozoic
rho_ast = 3378.0

#radiogenic heat production (W/kg)
H_air = 0.0
H_sed = 1.25e-6 / 2700.0
H_dec = 1.25e-6 / 2700.0
# H_upper_crust = 1.25e-6 / 2700.0 #9.259E-10 #old
# H_lower_crust = 0.2e-6 / 2800.0 #2.85E-10 #old
H_upper_crust = 1.67e-6 / 2700.0 #
H_lower_crust = 0.19e-6 / 2800.0 #
H_seed = 0.2e-6 / 2800.0

# radiogenic_heat_mlit = True
radiogenic_heat_mlit = False
if(radiogenic_heat_mlit):
    H_mlit = 9.0e-12               
    H_mlit_uc = 9.0e-12
    H_mlit_lc = 9.0e-12
else:
    H_mlit = 0.0 #9.0e-12               
    H_mlit_uc = 0.0 #9.0e-12
    H_mlit_lc = 0.0 #9.0e-12

H_ast = 0.0 #Turccote book: 7.38e-12 #Default is 0.0

#Pre exponential constant (Pa**-n s**-1)
A_air = 1.0E-18
A_sed = 8.574e-28
A_dec = 8.574e-28
A_upper_crust = 8.574e-28
A_lower_crust = 8.574e-28
A_seed = 8.574e-28
A_mlit = 2.4168e-15
A_mlit_uc = 2.4168e-15
A_mlit_lc = 2.4168e-15
A_ast = 1.393e-14

#Power law exponent
n_air = 1.0
n_sed = 4.0
n_dec = 4.0
n_upper_crust = 4.0
n_lower_crust = 4.0
n_seed = 4.0
n_mlit = 3.5
n_mlit_uc = 3.5
n_mlit_lc = 3.5
n_ast = 3.0

#Activation energy (J/mol)
Q_air = 0.0
Q_sed = 222.0e3
Q_dec = 222.0e3
Q_upper_crust = 222.0e3
Q_lower_crust = 222.0e3
Q_seed = 222.0e3
Q_mlit = 540.0e3
Q_mlit_uc = 540.0e3
Q_mlit_lc = 540.0e3
Q_ast = 429.0e3

#Activation volume (m3/mol)
V_air = 0.0
V_sed = 0.0
V_dec = 0.0
V_upper_crust = 0.0
V_lower_crust = 0.0
V_seed = 0.0
V_mlit = 25.0e-6
V_mlit_uc = 25.0e-6
V_mlit_lc = 25.0e-6
V_ast = 15.0e-6

with open("interfaces.txt", "w") as f:
    rheology_mlit = 'dry' #rheology of lithospheric mantle: dry olivine or wet olivine
    layer_properties = f"""
        C   {C_ast}   {C_mlit_lc}   {C_mlit_uc}   {C_mlit}   {C_lower_crust}   {C_seed}   {C_lower_crust}   {C_upper_crust}   {C_dec}   {C_sed}   {C_air}
        rho {rho_ast} {rho_mlit_lc} {rho_mlit_uc} {rho_mlit} {rho_lower_crust} {rho_seed} {rho_lower_crust} {rho_upper_crust} {rho_dec} {rho_sed} {rho_air}
        H   {H_ast}   {H_mlit_lc}   {H_mlit_uc}   {H_mlit}   {H_lower_crust}   {H_seed}   {H_lower_crust}   {H_upper_crust}   {H_dec}   {H_sed}   {H_air}
        A   {A_ast}   {A_mlit_lc}   {A_mlit_uc}   {A_mlit}   {A_lower_crust}   {A_seed}   {A_lower_crust}   {A_upper_crust}   {A_dec}   {A_sed}   {A_air}
        n   {n_ast}   {n_mlit_lc}   {n_mlit_uc}   {n_mlit}   {n_lower_crust}   {n_seed}   {n_lower_crust}   {n_upper_crust}   {n_dec}   {n_sed}   {n_air}
        Q   {Q_ast}   {Q_mlit_lc}   {Q_mlit_uc}   {Q_mlit}   {Q_lower_crust}   {Q_seed}   {Q_lower_crust}   {Q_upper_crust}   {Q_dec}   {Q_sed}   {Q_air}
        V   {V_ast}   {V_mlit_lc}   {V_mlit_uc}   {V_mlit}   {V_lower_crust}   {V_seed}   {V_lower_crust}   {V_upper_crust}   {V_dec}   {V_sed}   {V_air}
    """

    for line in layer_properties.split("\n"):
        line = line.strip()
        if len(line):
            f.write(" ".join(line.split()) + "\n")

    # layer interfaces
    data = -1 * np.array(tuple(interfaces.values())).T
    np.savetxt(f, data, fmt="%.1f")

##############################################################################
# Creating parameter file
##############################################################################

# high_kappa_in_asthenosphere = True
high_kappa_in_asthenosphere = False #default

#Convergence criteria
denok                            = 1.0e-14
particles_per_element            = 40

#Surface constrains
sp_surface_tracking              = True
sp_surface_processes             = False


#External inputs: bc velocity, velocity field, precipitation and
#climate change

velocity_from_ascii              = True
# velocity_from_ascii              = False
velocity = 1.0 #cm/yr
if(velocity_from_ascii == True):
    variable_bcv                     = True
else:
    variable_bcv                     = False

#time constrains 
if(variable_bcv == True):
    ti_convergence = 25.0
    ti_quiescence = 75.0
    # dt_quiescence = 0
    # dt_quiescence = 30
    # dt_quiescence = 60
    # dt_quiescence = 100
    dt_quiescence = 300
    tf_quiescence = ti_quiescence + dt_quiescence
    dt_rifting2 = 60.0

    time_max = (tf_quiescence + dt_rifting2)*1.0e6 #210.0e6
else:
    # time_max = 435.0e6
    time_max = 500.0e6

dt_max                           = 10.0e3 #default
step_print                       = 50#25

if(sp_surface_processes == True):
    precipitation_profile_from_ascii = True #False
    climate_change_from_ascii        = True #False
else:
    precipitation_profile_from_ascii = False
    climate_change_from_ascii        = False 

#step files
print_step_files                 = True

checkered = False

# magmatism = 'on'
magmatism = 'off'

rheol = 19

#velocity bc
top_normal_velocity                 = 'fixed'         # ok
top_tangential_velocity             = 'free '         # ok
bot_normal_velocity                 = 'fixed'         # ok
bot_tangential_velocity             = 'free'         # ok
left_normal_velocity                = 'fixed'         # ok
left_tangential_velocity            = 'fixed'         # ok
right_normal_velocity               = 'fixed'         # ok
right_tangential_velocity           = 'fixed'         # ok

# periodic_boundary = True
periodic_boundary = False

if(periodic_boundary == True):
    left_normal_velocity                = 'free'         # ok
    left_tangential_velocity            = 'free '         # ok
    right_normal_velocity               = 'free'         # ok
    right_tangential_velocity           = 'free'         # ok

#temperature bc
top_temperature                     = 'fixed'         # ok
bot_temperature                     = 'fixed'         # ok
# left_temperature                    = 'free'          # ok
left_temperature                    = 'fixed'          # ok
# right_temperature                   = 'free'          # ok
right_temperature                   = 'fixed'          # ok

##############################################################################
# Parameters file
##############################################################################
params = f"""
nx = {Nx}
nz = {Nz}
lx = {Lx}
lz = {Lz}
# Simulation options
multigrid                           = 1             # ok -> soon to be on the command line only
solver                              = direct        # default is direct [direct/iterative]
denok                               = {denok}       # default is 1.0E-4
particles_per_element               = {particles_per_element}          # default is 81
particles_perturb_factor            = 0.7           # default is 0.5 [values are between 0 and 1]
rtol                                = 1.0e-7        # the absolute size of the residual norm (relevant only for iterative methods), default is 1.0E-5
RK4                                 = Euler         # default is Euler [Euler/Runge-Kutta]
Xi_min                              = 1.0e-6       # default is 1.0E-14
random_initial_strain               = 0.0           # default is 0.0
pressure_const                      = -1.0          # default is -1.0 (not used) - useful only in horizontal 2D models
initial_dynamic_range               = True         # default is False [True/False]
periodic_boundary                   = False         # default is False [True/False]
high_kappa_in_asthenosphere         = {high_kappa_in_asthenosphere}         # default is False [True/False]
K_fluvial                           = 2.0e-7        # default is 2.0E-7
m_fluvial                           = 1.0           # default is 1.0
sea_level                           = 0.0           # default is 0.0
basal_heat                          = 0.0          # default is -1.0
# Surface processes
sp_surface_tracking                 = {sp_surface_tracking}         # default is False [True/False]
sp_surface_processes                = {sp_surface_processes}         # default is False [True/False]
sp_dt                               = 1.0e5        # default is 0.0
sp_d_c                              = 1.0          # default is 0.0
plot_sediment                       = False         # default is False [True/False]
a2l                                 = True          # default is True [True/False]
free_surface_stab                   = True          # default is True [True/False]
theta_FSSA                          = 0.5           # default is 0.5 (only relevant when free_surface_stab = True)
# Time constrains
step_max                            = 800000          # Maximum time-step of the simulation
time_max                            = {time_max}  #1.0e9     # Maximum time of the simulation [years]
dt_max                              = {dt_max}      # Maximum time between steps of the simulation [years]
step_print                          = {step_print} #500            # Make file every <step_print>
sub_division_time_step              = 0.5           # default is 1.0
initial_print_step                  = 0             # default is 0
initial_print_max_time              = 1.0e6         # default is 1.0E6 [years]
# Viscosity
viscosity_reference                 = 1.0e26        # Reference viscosity [Pa.s]
viscosity_max                       = 1.0e25        # Maximum viscosity [Pa.s]
viscosity_min                       = 1.0e18        # Minimum viscosity [Pa.s]
viscosity_per_element               = constant      # default is variable [constant/variable]
viscosity_mean_method               = arithmetic      # default is harmonic [harmonic/arithmetic]
viscosity_dependence                = pressure      # default is depth [pressure/depth]
# External ASCII inputs/outputs
interfaces_from_ascii               = True          # default is False [True/False]
n_interfaces                        = {len(interfaces.keys())}           # Number of interfaces int the interfaces.txt file
variable_bcv                        = {variable_bcv} #False         # default is False [True/False]
temperature_from_ascii              = True         # default is False [True/False]
velocity_from_ascii                 = {velocity_from_ascii} #False      # default is False [True/False]
binary_output                       = False         # default is False [True/False]
sticky_blanket_air                  = True         # default is False [True/False]
precipitation_profile_from_ascii    = {precipitation_profile_from_ascii}         # default is False [True/False]
climate_change_from_ascii           = {climate_change_from_ascii}         # default is False [True/False]
print_step_files                    = {print_step_files}          # default is True [True/False]
checkered                           = {checkered}         # Print one element in the print_step_files (default is False [True/False])
sp_mode                             = 5             # default is 1 [0/1/2]
geoq                                = on            # ok
geoq_fac                            = 100.0           # ok
# Physical parameters
temperature_difference              = 1500.         # ok
thermal_expansion_coefficient       = 3.28e-5       # ok
thermal_diffusivity_coefficient     = 1.0e-6 #0.75e-6       #default is 1.0e-6        # ok
gravity_acceleration                = 10.0          # ok
density_mantle                      = 3300.         # ok
external_heat                       = 0.0e-12       # ok
heat_capacity                       = 700.         # ok #default is 1250
non_linear_method                   = on            # ok
adiabatic_component                 = on            # ok
radiogenic_component                = on            # ok
magmatism                           = {magmatism}           # ok
# Velocity boundary conditions
top_normal_velocity                 = fixed         # ok
top_tangential_velocity             = free          # ok
bot_normal_velocity                 = fixed         # ok
bot_tangential_velocity             = free          # ok
left_normal_velocity                = {left_normal_velocity}         # ok
left_tangential_velocity            = {left_tangential_velocity}          # ok
right_normal_velocity               = {right_normal_velocity}         # ok
right_tangential_velocity           = {right_tangential_velocity}         # ok
surface_velocity                    = 0.0e-2        # ok
multi_velocity                      = False         # default is False [True/False]
# Temperature boundary conditions
top_temperature                     = {top_temperature}         # ok
bot_temperature                     = {bot_temperature}         # ok
left_temperature                    = {left_temperature}         # ok
right_temperature                   = {right_temperature}         # ok
rheology_model                      = {rheol}             # ok
T_initial                           = 3             # ok
"""

# Create the parameter file
with open("param.txt", "w") as f:
    for line in params.split("\n"):
        line = line.strip()
        if len(line):
            f.write(" ".join(line.split()) + "\n")


##############################################################################
# Initial temperature field
##############################################################################

# scenario = '/Doutorado/cenarios/mandyoc/stable/lit80km/stable_PT200_rheol19_c1250_C1_HprodAst/'
# scenario_name = scenario.split('/')[-2]
# print(scenario_name)

DeltaT = 0
# DeltaT = 200
# DeltaT = 290
# DeltaT = 350

# preset = True
preset = False

#Force cold cratonic keel
# keel_adjust = True
# keel_adjust = False

if(preset == False):
    T = 1300 * (z - thickness_sa) / (thickness_litho)  # Temperature
    T_cratonic = 1300 * (z - thickness_sa) / (thickness_crat_bot)
    
    # T = 1300 * (z - thickness_sa) / (130*1.0E3)  # Temperature of 1300 isotherm bellow the lithosphere

    ccapacity = 1250*1.0 #937.5=75% #J/kg/K? #DEFAULT
    
    # TP = 1262 #mantle potential temperature
    # TP = 1350
    # TP = 1400
    TP = 1450

    Ta = (TP / np.exp(-10 * 3.28e-5 * (z - thickness_sa) / ccapacity)) + DeltaT #Temperature profile for asthenosphere

    T[T < 0.0] = 0.0 #forcing negative temperatures to 0
    cond1 = Ta<T #VICTOR - selecting where asthenosphere temperature is lower than lithospheric temperature
    T[T > Ta] = Ta[T > Ta] #apply the temperature of asthenosphere (Ta) where temperature (T) is greater than Ta

    T_cratonic[T_cratonic < 0.0] = 0.0 #forcing negative temperatures to 0
    cond1_cratonic = Ta<T_cratonic #VICTOR
    T_cratonic[T_cratonic > Ta] = Ta[T_cratonic > Ta] #apply the temperature of asthenosphere Ta where temperature T is greater than Ta

    # kappa = 0.75*1.0e-6 #thermal diffusivity
    kappa = 1.0e-6 #thermal diffusivity

    H = np.zeros_like(T)

    cond = (z >= thickness_sa) & (z < thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust)  # upper crust
    H[cond] = H_upper_crust

    cond = (z >= thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust) & (
        z < thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust
    )  # lower crust
    H[cond] = H_lower_crust

    #Conductive model
    Taux = np.copy(T)
    Taux_cratonic = np.copy(T_cratonic)

    t = 0
    dt = 5000
    dt_sec = dt * 365 * 24 * 3600
    # cond = (z > thickness_sa + thickness_litho) | (T == 0)  # (T > 1300) | (T == 0) #OLD
    cond = cond1 | (T == 0)  # (T > 1300) | (T == 0) #VICTOR
    cond_cratonic = cond1_cratonic | (T_cratonic == 0)
    dz = Lz / (Nz - 1)
    
    while t < 2000.0e6:
        #non-cratonic region
        T[1:-1] += (
            kappa * dt_sec * ((T[2:] + T[:-2] - 2 * T[1:-1]) / dz ** 2)
            + H[1:-1] * dt_sec / ccapacity
        )
        T[cond] = Taux[cond]

        #cratonic region
        T_cratonic[1:-1] += (
            kappa * dt_sec * ((T_cratonic[2:] + T_cratonic[:-2] - 2 * T_cratonic[1:-1]) / dz ** 2)
            + H[1:-1] * dt_sec / ccapacity
        )
        T_cratonic[cond_cratonic] = Taux_cratonic[cond_cratonic]

        t = t + dt

    T = np.ones_like(X) * T[:, None] #(Nz, Nx)
    T_cratonic = np.ones_like(X) * T_cratonic[:, None] #(Nz, Nx)
    
    xcenter = Lx/2
    xregion_cratonic = (X <= xcenter - L_nc/2 - L_retreat) | (X >= xcenter + L_nc/2)
    xregion_non_cratonic = (X > xcenter - L_nc/2) & (X < xcenter + L_nc/2)

    #Applying the cratonic temperature profile in initial temperature field (T)
    T[xregion_cratonic] = T_cratonic[xregion_cratonic]
    # print('shape T: ', np.shape(T))

    # Save the initial temperature file
    np.savetxt("input_temperature_0.txt", np.reshape(T, (Nx * Nz)), header="T1\nT2\nT3\nT4")
    
else:
    print("Need to implement!")

##############################################################################
# Boundary condition - velocity
##############################################################################
if(velocity_from_ascii == True):
    fac_air = 10.0e3

    # 1 cm/year
    vL = (0.5*velocity/100) / (365 * 24 * 3600)  # m/s

    h_v_const = thickness_crat_bot + 20.0e3  #thickness with constant velocity 
    ha = Lz - thickness_sa - h_v_const  # difference

    vR = 2 * vL * (h_v_const + fac_air + ha) / ha  # this is to ensure integral equals zero

    VX = np.zeros_like(X)
    cond = (Z > h_v_const + thickness_sa) & (X == 0)
    VX[cond] = vR * (Z[cond] - h_v_const - thickness_sa) / ha

    cond = (Z > h_v_const + thickness_sa) & (X == Lx)
    VX[cond] = -vR * (Z[cond] - h_v_const - thickness_sa) / ha

    cond = X == Lx
    VX[cond] += +2 * vL

    cond = Z <= thickness_sa - fac_air
    VX[cond] = 0

    # print(np.sum(VX))

    v0 = VX[(X == 0)]
    vf = VX[(X == Lx)]
    sv0 = np.sum(v0[1:-1]) + (v0[0] + v0[-1]) / 2.0
    svf = np.sum(vf[1:-1]) + (vf[0] + vf[-1]) / 2.0
    # print(sv0, svf, svf - sv0)

    diff = (svf - sv0) * dz

    vv = -diff / Lx
    # print(vv, diff, svf, sv0, dz, Lx)

    VZ = np.zeros_like(X)

    cond = Z == 0
    VZ[cond] = vv
    #save bc to plot arraows in numerical setup
    vels_bc = np.array([v0, vf])
    vz0 = VZ[(z == 0)]

    np.savetxt("vel_bc.txt", vels_bc.T)
    np.savetxt("velz_bc.txt", vz0.T)
    # print(np.sum(v0))

    VVX = np.copy(np.reshape(VX, Nx * Nz))
    VVZ = np.copy(np.reshape(VZ, Nx * Nz))

    v = np.zeros((2, Nx * Nz))

    v[0, :] = VVX
    v[1, :] = VVZ

    v = np.reshape(v.T, (np.size(v)))

    # Create the initial velocity file
    np.savetxt("input_velocity_0.txt", v, header="v1\nv2\nv3\nv4")

    if(variable_bcv == True):
        var_bcv = f""" 4
                    {ti_convergence} -1.0
                    {ti_quiescence} 0.01
                    {tf_quiescence} -100.0
                    {time_max/1.0e6} 0.01
                    """

        # Create the parameter file
        with open("scale_bcv.txt", "w") as f:
            for line in var_bcv.split("\n"):
                line = line.strip()
                if len(line):
                    f.write(" ".join(line.split()) + "\n")

##############################################################################
# Surface processes
##############################################################################

if(sp_surface_processes == True):
    if(climate_change_from_ascii == True):
        #When climate effects will start to act - scaling to 1
        climate = f'''
                2
                0 0.0
                120 0.02
            '''

        with open('climate.txt', 'w') as f:
            for line in climate.split('\n'):
                line = line.strip()
                if len(line):
                    f.write(' '.join(line.split()) + '\n')

    if(precipitation_profile_from_ascii ==True):
        #Creating precipitation profile

        prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/(1))**6) #Lx km
        # prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/8)**6) #original
        # prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/(8*2))**6) #100 km
        # prec = 0.0008*np.exp(-(x-Lx/2)**6/(Lx/(8*4))**6) #50 km

        # plt.figure(figsize=(12, 9), constrained_layout=True)
        # plt.xlim([0, Lx/1.0E3])
        # plt.ylim([0, np.max(prec)])
        # plt.xlabel("km", fontsize=label_size)
        # plt.ylabel("Precipitation", fontsize=label_size)
        # plt.plot(x/1000,prec)
        # plt.grid(':k', alpha=0.7)

        # figname='precipitation_profile.png'
        # plt.savefig(figname, dpi=300)

        np.savetxt("precipitation.txt", prec, fmt="%.8f")


##############################################################################
#
# Creating a single plot with scenario infos
#
##############################################################################

plt.close()
fig, axs = plt.subplots(1, 1, figsize = (14, 6))
ylimplot = [-Lz/1000+thickness_sa/1000, 0+thickness_sa/1000]
#plot scenario layers
#layers colour scheme
cr = 255.
color_air = "xkcd:white"
color_sed = (241./cr,184./cr,68./cr)
color_dec = (137./cr,81./cr,151./cr)
color_uc = (228./cr,156./cr,124./cr)
color_lc = (240./cr,209./cr,188./cr)
color_lit = (155./cr,194./cr,155./cr)
color_mlit_uc = (180. / cr, 194. / cr, 162. / cr)
color_mlit_lc = (180. / cr, 194. / cr, 162. / cr)#(155. / cr, 194. / cr, 155. / cr)
color_ast = (207./cr,226./cr,205./cr)

colors = {'air': color_sed,
    'sediments': color_dec,
    'decolement':color_uc,
    'upper_crust': color_lc,
    'seed_top': color_lc,
    'seed_base': color_lc,
    'lower_crust': color_lit,
    'litho_nc': color_mlit_uc,
    'litho_crat_up': color_mlit_lc,
    'litho_crat_bot': color_ast,
}

for interface in list(interfaces.items())[::-1]:
    label, layer = interface[0], interface[1]
    axs.plot(x/1000, (-layer)/1000+thickness_sa/1000, color='k', lw=1)
    if(label == 'seed_top' or label == 'seed_base'):
        label = 'lower_crust'
        continue
    if(label == 'litho_crat_up'):
        continue

    axs.fill_between(x/1000, -layer/1000+thickness_sa/1000, -Lz/1000+thickness_sa/1000, color=colors[label])#, label=labels[label])

axs.set_xlim(0, Lx/1000)
axs.set_xticks([])
axs.set_yticks([])
axs.set_ylim(ylimplot)
axs.set_xlabel(f"Lx = {Lx/1000:.0f} km; dx = {dx/1000:.0f} km", fontsize=16)
axs.set_ylabel(f"Lz = {Lz/1000:.0f} km; dz = {dz/1000:.0f} km", fontsize=16)

#plotting ghost points to create a legend to the layers
colors_legend = {'air': color_air,
                 'sediments': color_sed,
                 'decolement':color_dec,
                 'upper_crust': color_uc,
                 'seed_top': color_lc,
                 'seed_base': color_lc,
                 'lower_crust': color_lc,
                 'litho_nc': color_lit,
                 'litho_crat_up': color_mlit_uc,
                 'litho_crat_bot': color_mlit_lc,
                 'asthenosphere': color_ast,}

labels_legend = {
    'air': f"Sticky air\n{C_air:.0f} x air\n"+fr"$\rho$ = {rho_air:.0f} kg/m³"+f"\n$h$={thickness_sa/1.0E3:.0f} km",
    'sediments': f"Sediments\n{C_sed:.0f} x wet quartz\n"+fr"$\rho$ = {rho_sed:.0f} kg/m³"+f"\nh={thickness_sed/1.0E3:.0f} km",
    'decolement': f"Decolement\n{C_dec:.1f} x wet quartz\n"+fr"$\rho$ = {rho_dec:.0f} kg/m³"+f"\nh={thickness_decolement/1.0E3:.0f} km",
    'upper_crust': f"Upper crust\n{C_upper_crust:.0f} x wet quartz\n"+fr"$\rho$ = {rho_upper_crust:.0f} kg/m³"+f"\nh={thickness_upper_crust/1.0E3:.0f} km",
    'lower_crust': f"Lower crust\n{C_lower_crust:.0f} x wet quartz\n"+fr"$\rho$ = {rho_lower_crust:.0f} kg/m³"+f"\nh={thickness_lower_crust/1.0E3:.0f} km",
    'litho_nc': f"Non-cratonic\nlith. mantle\n{C_mlit:.0f} x dry olivine\n"+fr"$\rho$ = {rho_mlit:.0f} kg/m³"+f"\nh={thickness_mlit/1.0E3:.0f} km",
    'litho_crat_up': f"Upper cratonic\nlith. mantle\n{C_mlit_uc:.0f} x dry olivine\n"+fr"$\rho$ = {rho_mlit_uc:.0f} kg/m³"+f"\nh={thickness_mlit_crat_up/1.0E3:.0f} km",
    'litho_crat_bot': f"Lower cratonic\nlith. mantle\n{C_mlit_lc:.0f} x dry olivine\n"+fr"$\rho$ = {rho_mlit_lc:.0f} kg/m³"+f"\nh={thickness_mlit_crat_bot/1.0E3:.0f} km",
    'asthenosphere': f"Asthenosphere\n{C_ast:.0f} x wet olivine\n"+fr"$\rho$ = {rho_ast:.0f} kg/m³"+f"\nh={thickness_astnc/1.0E3:.0f}-{thickness_astc/1.0E3:.0f} km"}

legend_elements = []
for key in labels_legend.keys():
    legend_elements.append(Patch(facecolor=colors_legend[key], edgecolor='black', label=labels_legend[key]))

fig.subplots_adjust(bottom=0.25)

leg = axs.legend(handles=legend_elements,
    ncol=len(legend_elements), 
    loc='lower center', 
    bbox_to_anchor=(0.5, -0.35), 
    frameon=False,
    title='Layers properties',
    title_fontsize=15,
    fontsize=8,
    columnspacing=1.0,
)

#Indicating weak seed position
xpos_seed = x_seed/Lx
correction = 0.94
ypos_seed = correction*(1-(thickness_sa + thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust)/Lz)
axs.text(xpos_seed, ypos_seed, f'weak seed\n{thickness_seed/1000:.0f}x{thickness_seed/1000:.0f} km²', color='k', fontsize=12, ha='center', va='center', transform=axs.transAxes)

#Temperature profiles
idx_center = int((Nx-1)/2) 
axt = axs.inset_axes((0.205,
                      0,
                      0.18,
                      1))
# axt.plot(temp_z[:, 0], -(z - t_sa) / 1.0e3, "-r")
axt.plot(T[:, 0], (-z + thickness_sa) / 1.0e3, "-k", label=f"Cratonic")#label=r'T$_{\mathrm{cratonic}}$')
axt.plot(T[:, idx_center], (-z + thickness_sa) / 1.0e3, "-r",label=f"Non-cratonic")# label=r'T$_{\mathrm{non-cratonic}}$')
axt.grid(visible=True, axis='x',which='both',ls='--',color='red',alpha=0.3)
axt.set_ylim(ylimplot)
axt.set_yticks([])
axt.set_xticks(np.linspace(0,1800,7))
axt.patch.set_alpha(0)
axt.xaxis.set_ticks_position('top')
axt.set_yticks([])
axt.tick_params(labelsize=8)
axt.xaxis.label.set_color('red')
axt.tick_params(axis='x', colors='k')
axt.spines['left'].set_visible(False)
axt.spines['right'].set_visible(False)
axt.set_title('Temp °C',color='k')
axt.legend(loc='lower left', fontsize=10, framealpha=0.9)
#axt.spines['bottom'].set_visible(False)

################################
#    Yield Strength Envelope   #
################################

Qnc = np.zeros_like(z)
Anc = np.zeros_like(z)
nnc = np.zeros_like(z)
Vnc = np.zeros_like(z)
Cnc = np.zeros_like(z)
rhonc = np.zeros_like(z)

Qc = np.zeros_like(z)
Ac = np.zeros_like(z)
nc = np.zeros_like(z)
Vc = np.zeros_like(z)
Cc = np.zeros_like(z)
rhoc = np.zeros_like(z)

zaux = z
air = zaux < thickness_sa
sed = (zaux>thickness_sa) & (zaux<thickness_sa+thickness_sed)
dec = (zaux>thickness_sa+thickness_sed) & (zaux<thickness_sa+thickness_sed+thickness_decolement)
uc =  (zaux>=thickness_sa+thickness_sed+thickness_decolement) & (zaux<thickness_sa+thickness_sed+thickness_decolement+thickness_upper_crust)
lc =  (zaux>=thickness_sa+thickness_sed+thickness_decolement+thickness_upper_crust) & (zaux<thickness_sa+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust)
lm =  (zaux>=thickness_sa+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust) & (zaux<=thickness_sa+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust+thickness_mlit)
#upper craton
luc = (zaux>=thickness_sa+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust) & (zaux<=thickness_sa+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust+thickness_mlit_crat_up)
#lower craton
llc = (zaux>=thickness_sa+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust+thickness_mlit_crat_up) & (zaux<=thickness_sa+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust+thickness_mlit_crat_up+thickness_mlit_crat_bot)
#bellow non-cratonic
astnc = zaux>thickness_sa+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust+thickness_mlit
#bellow craton
astc = zaux>thickness_sa+thickness_sed+thickness_decolement+thickness_upper_crust+thickness_lower_crust+thickness_mlit_crat_up+thickness_mlit_crat_bot

#non cratonic rheological properties
Cnc[air] = C_air
Cnc[sed] = C_sed
Cnc[dec] = C_dec
Cnc[uc] = C_upper_crust
Cnc[lc] = C_lower_crust
Cnc[lm] = C_mlit
Cnc[astnc] = C_ast

rhonc[air] = rho_air
rhonc[sed] = rho_sed
rhonc[dec] = rho_dec
rhonc[uc] = rho_upper_crust
rhonc[lc] = rho_lower_crust
rhonc[lm] = rho_mlit
rhonc[astnc] = rho_ast

Anc[air] = A_air
Anc[sed] = A_sed
Anc[dec] = A_dec
Anc[uc] = A_upper_crust
Anc[lc] = A_lower_crust
Anc[lm] = A_mlit
Anc[astnc] = A_ast

nnc[air] = n_air
nnc[sed] = n_sed
nnc[dec] = n_dec
nnc[uc] = n_upper_crust
nnc[lc] = n_lower_crust
nnc[lm] = n_mlit
nnc[astnc] = n_ast

Qnc[air] = Q_air
Qnc[sed] = Q_sed
Qnc[dec] = Q_dec
Qnc[uc] = Q_upper_crust
Qnc[lc] = Q_lower_crust
Qnc[lm] = Q_mlit
Qnc[astnc] = Q_ast

Vnc[air] = V_air
Vnc[sed] = V_sed
Vnc[dec] = V_dec
Vnc[uc] = V_upper_crust
Vnc[lc] = V_lower_crust
Vnc[lm] = V_mlit
Vnc[astnc] = V_ast

#cratonic rheological properties
Cc[air] = C_air
Cc[sed] = C_sed
Cc[dec] = C_dec
Cc[uc] = C_upper_crust
Cc[lc] = C_lower_crust
Cc[luc] = C_mlit_uc
Cc[llc] = C_mlit_lc
Cc[astc] = C_ast

rhoc[air] = rho_air
rhoc[sed] = rho_sed
rhoc[dec] = rho_dec
rhoc[uc] = rho_upper_crust
rhoc[lc] = rho_lower_crust
rhoc[luc] = rho_mlit_uc
rhoc[llc] = rho_mlit_lc
rhoc[astc] = rho_ast

Ac[air] = A_air
Ac[sed] = A_sed
Ac[dec] = A_dec
Ac[uc] = A_upper_crust
Ac[lc] = A_lower_crust
Ac[luc] = A_mlit_uc
Ac[llc] = A_mlit_lc
Ac[astc] = A_ast

nc[air] = n_air
nc[sed] = n_sed
nc[dec] = n_dec
nc[uc] = n_upper_crust
nc[lc] = n_lower_crust
nc[luc] = n_mlit_uc
nc[llc] = n_mlit_lc
nc[astc] = n_ast

Qc[air] = Q_air
Qc[sed] = Q_sed
Qc[dec] = Q_dec
Qc[uc] = Q_upper_crust
Qc[lc] = Q_lower_crust
Qc[luc] = Q_mlit_uc
Qc[llc] = Q_mlit_lc
Qc[astc] = Q_ast

Vc[air] = V_air
Vc[sed] = V_sed
Vc[dec] = V_dec
Vc[uc] = V_upper_crust
Vc[lc] = V_lower_crust
Vc[luc] = V_mlit_uc
Vc[llc] = V_mlit_lc
Vc[astc] = V_ast

sr = 1.0E-15 #strain rate - s-1
# sr = 1.0E-14
R = 8.314 #gas constant - J K−1 mol−1
g = 10.0

Pnc = rhonc[::-1].cumsum()[::-1]*g*dz
Pc = rhoc[::-1].cumsum()[::-1]*g*dz

phi = 2.0*np.pi/180.0
c0 = 4.0E6

sigmanc_min = c0 * np.cos(phi) + Pnc * np.sin(phi)
sigmac_min = c0 * np.cos(phi) + Pc * np.sin(phi)

phi = 15.0*np.pi/180.0
c0 = 20.0E6
sigmanc_max = c0 * np.cos(phi) + Pnc * np.sin(phi)
sigmac_max = c0 * np.cos(phi) + Pc * np.sin(phi)

TKnc = T[:, idx_center] + 273
TKc = T[:, 0] + 273

viscnc = Cnc * Anc**(-1./nnc) * sr**((1.0-nnc)/nnc)*np.exp((Qnc + Vnc*Pnc)/(nnc*R*TKnc))
sigmanc_v = viscnc * sr
condnc = sigmanc_v>sigmanc_max
sigmanc_v[condnc]=sigmanc_max[condnc]

viscc = Cc * Ac**(-1./nc) * sr**((1.0-nc)/nc)*np.exp((Qc + Vc*Pc)/(nc*R*TKc))
sigmac_v = viscc * sr
condc = sigmac_v>sigmac_max
sigmac_v[condc]=sigmac_max[condc]

axsg = axs.inset_axes((0.605,
                       0,
                       0.13,
                       1))

axsg.plot(sigmanc_v/1e9,-(z-thickness_sa)/1e3,'r', label=f'Non-cratonic')
# axsg.plot(sigmanc_min/1e9,-(z-t_sa)/1e3,'k--',lw=0.8)
axsg.plot(sigmac_v/1e9,-(z-thickness_sa)/1e3,'k', label=f'Cratonic')


axsg.grid(visible=True, axis='x',which='both',ls='--',color='gray',alpha=0.8)
axsg.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axsg.set_xlim(-0.1,1.1)
axsg.set_ylim(ylimplot)
axsg.set_yticks([])
axsg.patch.set_alpha(0)
axsg.xaxis.set_ticks_position('top')
axsg.spines['left'].set_visible(False)
axsg.spines['right'].set_visible(False)
axsg.set_title('$\sigma_{YSE}$ (GPa)')
axsg.tick_params(labelsize=8)

#Effective friction angle
axsf = axs.inset_axes((0.800,
                       0,
                       0.10,
                       0.30))
axsf.patch.set_alpha(0)

# xdata = np.array([0, 0.05, 1.05, 1.1])
xdata = np.array([0, 0.25, 0.75, 1.0])
ydata = np.array([15, 15, 2, 2])

fisize = 10
axsf.plot(xdata, ydata, 'k-')
axsf.set_xlim([0.10, 0.9])
axsf.set_ylim([0, 17])
axsf.set_xticks([0.25, 0.5, 0.75])
axsf.set_xticklabels([0.05, ' ', 1.05])
axsf.tick_params('x', top=True, labeltop=False)
axsf.xaxis.set_ticks_position('bottom')
axsf.set_xlabel(r"$\varepsilon$", fontsize=fisize)

axsf.set_yticks([2, 15])
axsf.set_yticklabels(['2°', '15°'])
axsf.set_ylabel('$\Phi_{\mathrm{eff}}$', fontsize=fisize)
axsf.tick_params(labelsize=fisize)

axsfC = axsf.twinx()
axsfC.set_ylim([0, 17])
axsfC.set_yticks([2, 15])
axsfC.set_yticklabels([4, 20])
axsfC.set_ylabel('Cohesion (MPa)', fontsize=fisize, rotation=90)
axsfC.tick_params(labelsize=fisize)

#plot velocity bc
if(velocity_from_ascii == True):
    #Velocity = Right side
    vr_plot = np.round(max(abs(VX[:, -1]* (100.0*365.0 * 24.0 * 3600.0))),0)
    fac = 0.96
    axr = axs.inset_axes((fac,
                         0,
                         (1-fac)*2,
                         1))

    scale_veloc = 100.0*365.0*24.0*3600.0
    crt=0

    #Right side velocity arrows
    axr.fill_betweenx((-z + thickness_sa) / 1.0e3, VX[:, -1]* (100.0*365.0 * 24.0 * 3600.0), 0, color=None, facecolor=None, hatch='---',alpha=0)
    axr.set_ylim(ylimplot)
    axr.set_yticks([])
    axr.patch.set_alpha(0)
    axr.set_title('cm/y')
    axr.xaxis.set_ticks_position('top')
    axr.tick_params(labelsize=8)
    axr.set_xlim([-vr_plot, vr_plot])
    axr.spines['left'].set_visible(False)
    axr.spines['right'].set_visible(False)
    axr.spines['bottom'].set_visible(False)

    # Left side velocity arrows
    vl_plot = np.round(max(abs(VX[:, 0]* (100*365.0 * 24.0 * 3600.0))),0)
    axl = axs.inset_axes((0,
                          0,
                          (1-fac)*2,
                          1))
    
    axl.fill_betweenx((-z + thickness_sa) / 1.0e3, VX[:, 0]* (100*365 * 24 * 3600), 0, color=None, facecolor=None, hatch='---',alpha=0)

    axl.set_ylim(-Lz/1000+thickness_sa/1000, 0+thickness_sa/1000)
    axl.set_yticks([])
    axl.patch.set_alpha(0)
    axl.set_title('cm/y')
    axl.xaxis.set_ticks_position('top')
    axl.set_xlim([0, vl_plot])
    axl.tick_params(labelsize=8)
    axl.spines['left'].set_visible(False)
    axl.spines['right'].set_visible(False)

figname = 'numerical_setup'
fig.savefig(f"{figname}.png", bbox_inches="tight", dpi=300)
plt.close()

##############################################################################
# Scenario infos
##############################################################################

print(f"Scenario kind: {experiemnts[scenario_kind]}")
print(f"N cores: {ncores}")
print('Domain parameters:')
print(f"\tLx: {Lx*1.0e-3} km")
print(f"\tLz: {Lz*1.0e-3} km")
print(f"\tNx: {Nx}")
print(f"\tNz: {Nz}")
print(f"Resolution dx x dz: {1.0e-3*Lx/(Nx-1)} x {1.0e-3*Lz/(Nz-1)} km2")
if(variable_bcv == True):
    print(f"Time of quiescence: {dt_quiescence} Myr")
print('Layers thickness:')
print(f"\tair: {thickness_sa*1.0e-3} km")
print(f"\tsediments: {thickness_sed/1000} km")
print(f"\tdecolement: {thickness_decolement/1000} km")
print(f"\tupper crust: {thickness_upper_crust*1.0e-3} km")
print(f"\tlower crust: {thickness_lower_crust*1.0e-3} km")
print(f"\tnon cratonic mantle lithosphere: {thickness_litho/1000} km")
print(f"\tupper cratonic mantle lithosphere: {thickness_mlit_crat_up/1000} km")
print(f"\tlower cratonic mantle lithosphere: {thickness_mlit_crat_bot/1000} km")
print(f"\tcrust: {(thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust)/1000} km")
print(f"\tnon cratonic lithosphere: {thickness_litho*1.0e-3} km")
print(f"\tcratonic lithosphere: {thickness_crat_bot*1.0e-3} km")
print('Important scale factors (C):')
print(f"\tair: {C_air}")
print(f"\tsediments: {C_seed}")
print(f"\tdecolement: {C_dec}")
print(f"\tupper crust: {C_upper_crust}")
print(f"\tlower crust: {C_lower_crust}")
print(f"\tweak seed: {C_seed}")
print(f"\tnon cratonic mantle lithosphere: {C_mlit}")
print(f"\tupper cratonic mantle lithosphere: {C_mlit_uc}")
print(f"\tlower cratonic mantle lithosphere: {C_mlit_lc}")
print(f"Preset of initial temperature field: {preset}")
print(f"Radiogenic heat in lithospheric mantle: {radiogenic_heat_mlit}")
print(f"Surface process: {sp_surface_processes}")
print(f"Velocity field: {velocity_from_ascii}")
print(f"Variable velocity field: {variable_bcv}")
print(f"Climate change: {climate_change_from_ascii}")
print(f"Periodic Boundary: {periodic_boundary}")
print('Initial temperature field setup:')
print(f"\tPreset of initial temperature field: {preset}")
print(f"\tIncrease in mantle basal temperature (Ta): {DeltaT} oC")
print(f"\tAssumed mantle Potential Temperature for diffusive model: {TP} oC")
print(' ')
print(f"Magmatism: {magmatism}")
print(f'Rheol number in param file: {rheol}')

#Save scenario infos
scenario_infos = ['SCENARIO INFOS:']
scenario_infos.append(' ')
scenario_infos.append('Name: ' + path[-1])
scenario_infos.append(f"Scenario kind: {experiemnts[scenario_kind]}")
scenario_infos.append(f"N cores: {ncores}")
scenario_infos.append(' ')
scenario_infos.append('Domain parameters:')
scenario_infos.append(f"\tLx: {Lx*1.0e-3} km")
scenario_infos.append(f"\tLz: {Lz*1.0e-3} km")
scenario_infos.append(f"\tNx: {Nx}")
scenario_infos.append(f"\tNz: {Nz}")
scenario_infos.append(f"Resolution dx x dz: {1.0e-3*Lx/(Nx-1)} x {1.0e-3*Lz/(Nz-1)} km2")
scenario_infos.append(' ')
if(variable_bcv == True):
    scenario_infos.append(f"Time of quiescence: {dt_quiescence} Myr")
    scenario_infos.append(' ')
scenario_infos.append('Layers thickness:')
scenario_infos.append(f"\tair: {thickness_sa*1.0e-3} km")
scenario_infos.append(f"\tsediments: {thickness_sed/1000} km")
scenario_infos.append(f"\tdecolement: {thickness_decolement/1000} km")
scenario_infos.append(f"\tupper crust: {thickness_upper_crust*1.0e-3} km")
scenario_infos.append(f"\tlower crust: {thickness_lower_crust*1.0e-3} km")
scenario_infos.append(f"\tnon cratonic mantle lithosphere: {thickness_litho/1000} km")
scenario_infos.append(f"\tupper cratonic mantle lithosphere: {thickness_mlit_crat_up/1000} km")
scenario_infos.append(f"\tlower cratonic mantle lithosphere: {thickness_mlit_crat_bot/1000} km")
scenario_infos.append(f"\tcrust: {(thickness_sed + thickness_decolement + thickness_upper_crust + thickness_lower_crust)/1000} km")
scenario_infos.append(f"\tnon cratonic lithosphere: {thickness_litho*1.0e-3} km")
scenario_infos.append(f"\tcratonic lithosphere: {thickness_crat_bot*1.0e-3} km")
scenario_infos.append(' ')
scenario_infos.append(' ')
scenario_infos.append('Important scale factors (C):')
scenario_infos.append(f"\tair: {C_air}")
scenario_infos.append(f"\tsediments: {C_seed}")
scenario_infos.append(f"\tdecolement: {C_dec}")
scenario_infos.append(f"\tupper crust: {C_upper_crust}")
scenario_infos.append(f"\tlower crust: {C_lower_crust}")
scenario_infos.append(f"\tweak seed: {C_seed}")
scenario_infos.append(f"\tnon cratonic mantle lithosphere: {C_mlit}")
scenario_infos.append(f"\tupper cratonic mantle lithosphere: {C_mlit_uc}")
scenario_infos.append(f"\tlower cratonic mantle lithosphere: {C_mlit_lc}")
scenario_infos.append(' ')
scenario_infos.append(f"Preset of initial temperature field: {preset}")
scenario_infos.append(f"Radiogenic heat in lithospheric mantle: {radiogenic_heat_mlit}")
scenario_infos.append(f"Surface process: {sp_surface_processes}")
scenario_infos.append(f"Velocity field: {velocity_from_ascii}")
if(velocity_from_ascii==True):
    scenario_infos.append(f"inital velocity: {vL*(365 * 24 * 3600)*2*100} cm/yr") 
scenario_infos.append(f"Variable velocity field: {variable_bcv}")
scenario_infos.append(f"Climate change: {climate_change_from_ascii}")
scenario_infos.append(f"Periodic Boundary: {periodic_boundary}")
scenario_infos.append('Initial temperature field setup:')
scenario_infos.append(f"\tPreset of initial temperature field: {preset}")
scenario_infos.append(f"\tIncrease in mantle basal temperature (Ta): {DeltaT} oC")
scenario_infos.append(f"\tAssumed mantle Potential Temperature for diffusive model: {TP} oC")
scenario_infos.append(' ')
scenario_infos.append(f"Magmatism: {magmatism}")
scenario_infos.append(f'Rheol number in param file: {rheol}')

np.savetxt('infos_'+path[-1] + '.txt', scenario_infos, fmt="%s")

##############################################################################
#Creating run_scripts
##############################################################################

linux = False
mac = False
aguia = False
hypatia = True

mandyoc_options = '-seed 0,5,8 -strain_seed 0.0,1.0,1.0'


if(linux):
    run_linux = f'''
            #!/bin/bash
            MPI_PATH=$HOME/opt/petsc/arch-label-optimized/bin
            MANDYOC_PATH=$HOME/opt/mandyoc
            NUMBER_OF_CORES=20
            touch FD.out
            $MPI_PATH/mpirun -n $NUMBER_OF_CORES $MANDYOC_PATH/mandyoc {mandyoc_options} | tee FD.out
        '''
    with open('run-linux.sh', 'w') as f:
        for line in run_linux.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')

if(mac):
    claudio=True
    dirname = '${PWD##*/}'

    if(claudio):
        PETSC_DIR = '/Users/claudiomora/Documents/petsc'
        PETSC_ARCH = 'arch-label-optimized/bin/mpiexec'
        MANDYOC = '/Users/claudiomora/Documents/mandyoc/bin/mandyoc'

        run_mac = f'''
                    #!/bin/bash
                    touch FD.out
                    {PETSC_DIR}/{PETSC_ARCH} -n 16 {MANDYOC} {mandyoc_options} | tee FD.log
                    
                    DIRNAME={dirname}

                    zip $DIRNAME.zip interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh
                    zip -u $DIRNAME.zip bc_velocity_*.txt
                    zip -u $DIRNAME.zip density_*.txt
                    zip -u $DIRNAME.zip heat_*.txt
                    zip -u $DIRNAME.zip pressure_*.txt
                    zip -u $DIRNAME.zip sp_surface_global_*.txt
                    zip -u $DIRNAME.zip strain_*.txt
                    zip -u $DIRNAME.zip temperature_*.txt
                    zip -u $DIRNAME.zip time_*.txt
                    zip -u $DIRNAME.zip velocity_*.txt
                    zip -u $DIRNAME.zip viscosity_*.txt
                    zip -u $DIRNAME.zip scale_bcv.txt
                    zip -u $DIRNAME.zip step*.txt
                    zip -u $DIRNAME.zip *.bin*
                    zip -u $DIRNAME.zip *.log
                    

                    #rm *.log
                    rm vel_bc*
                    rm velz*
                    rm bc_velocity*
                    rm velocity*
                    rm step*
                    rm temperature*
                    rm density*
                    rm viscosity*
                    rm heat*
                    rm strain_*
                    rm time*
                    rm pressure_*
                    rm sp_surface_global*
                    rm scale_bcv.txt
                    rm *.bin*

                '''
    else:
        run_mac = f'''
                #!/bin/bash
                MPI_PATH=$HOME/opt/petsc/arch-label-optimized/bin
                MANDYOC_PATH=$HOME/opt/mandyoc
                NUMBER_OF_CORES=6
                touch FD.out
                $MPI_PATH/mpirun -n $NUMBER_OF_CORES $MANDYOC_PATH/mandyoc {mandyoc_options} | tee FD.out
                bash $HOME/Doutorado/cenarios/mandyoc/scripts/zipper_gcloud.sh
                #bash $HOME/Doutorado/cenarios/mandyoc/scripts/clean_gcloud.sh
            '''
    with open('run_mac.sh', 'w') as f:
        for line in run_mac.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')

if(aguia):
    dirname = '${PWD##*/}'

    aguia_machine = 'aguia4'
    # aguia_machine = 'aguia3'

    if(aguia_machine == 'aguia4'):
        partition = 'SP2'
        main_folders = '/temporario2/8672526'

    if(aguia_machine == 'aguia3'):
        partition = 'SP3'
        main_folders =  '/scratch/8672526'

    run_aguia = f'''
            #!/usr/bin/bash

            #SBATCH --partition={partition}
            #SBATCH --ntasks=1
            #SBATCH --nodes=1
            #SBATCH --cpus-per-task={str(int(ncores))}
            #SBATCH --time 192:00:00 #16horas/"2-" para 2 dias com max 8 dias
            #SBATCH --job-name mandyoc-jpms
            #SBATCH --output slurm_%j.log #ou FD.out/ %j pega o id do job
            #SBATCH --mail-type=BEGIN,FAIL,END
            #SBATCH --mail-user=joao.macedo.silva@usp.br

            export PETSC_DIR={main_folders}/opt/petsc
            export PETSC_ARCH=arch-label-optimized
            MANDYOC={main_folders}/opt/mandyoc/bin/mandyoc
            MANDYOC_OPTIONS={mandyoc_options}

            $PETSC_DIR/$PETSC_ARCH/bin/mpiexec -n {str(int(ncores))} $MANDYOC $MANDYOC_OPTIONS

            DIRNAME={dirname}

            zip $DIRNAME.zip interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh
            zip -u $DIRNAME.zip bc_velocity_*.txt
            zip -u $DIRNAME.zip density_*.txt
            zip -u $DIRNAME.zip heat_*.txt
            zip -u $DIRNAME.zip pressure_*.txt
            zip -u $DIRNAME.zip sp_surface_global_*.txt
            zip -u $DIRNAME.zip strain_*.txt
            zip -u $DIRNAME.zip temperature_*.txt
            zip -u $DIRNAME.zip time_*.txt
            zip -u $DIRNAME.zip velocity_*.txt
            zip -u $DIRNAME.zip viscosity_*.txt
            zip -u $DIRNAME.zip scale_bcv.txt
            zip -u $DIRNAME.zip step*.txt
            zip -u $DIRNAME.zip *.log

            #rm *.log
            rm vel_bc*
            rm velz*
            rm bc_velocity*
            rm velocity*
            rm step*
            rm temperature*
            rm density*
            rm viscosity*
            rm heat*
            rm strain_*
            rm time*
            rm pressure_*
            rm sp_surface_global*
            rm scale_bcv.txt
        '''
    with open('run_aguia.sh', 'w') as f:
        for line in run_aguia.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')

if(hypatia):

    dirname = '${PWD##*/}'
    main_folders = '/scratch/joao'
    run_hypatia = f'''
    #!/usr/bin/env bash
    #SBATCH --mail-type=BEGIN,END,FAIL         			# Mail events (NONE, BEGIN, END, FAIL, ALL)
    #SBATCH --mail-user=joao.macedo.silva@usp.br		# Where to send mail
    #SBATCH --ntasks={str(int(ncores))}
    #SBATCH --nodes=1
    #SBATCH --cpus-per-task=1
    #SBATCH --time 72:00:00 # 16 horas; poderia ser “2-” para 2 dias; máximo “8-”
    #SBATCH --job-name mandyoc-jpms
    #SBATCH --output slurm_%j.log

    #Setup of Mandyoc variables:
    PETSC_DIR='{main_folders}/opt/petsc'
    PETSC_ARCH='arch-label-optimized/bin/mpirun'
    MANDYOC='{main_folders}/opt/mandyoc/bin/mandyoc'
    MANDYOC_OPTIONS='{mandyoc_options}'

    #run mandyoc
    $PETSC_DIR/$PETSC_ARCH -n {str(int(ncores))} $MANDYOC $MANDYOC_OPTIONS

    #run of auxiliary scripts to zip and clean the folder
    bash zipper.sh
    bash clean.sh
    '''
    with open('run_hypatia.sh', 'w') as f:
        for line in run_hypatia.split('\n'):
            line = line.strip()
            if len(line):
                f.write(' '.join(line.split()) + '\n')


zipper = f'''
        #!/usr/bin/env bash
        DIRNAME={dirname}

        # Primeiro zipa os arquivos fixos
        zip "$DIRNAME.zip" interfaces.txt param.txt input*_0.txt vel_bc.txt velz_bc.txt run*.sh

        # Lista de padrões
        patterns=(
            "bc_velocity_*.txt"
            "density_*.txt"
            "heat_*.txt"
            "pressure_*.txt"
            "sp_surface_global_*.txt"
            "strain_*.txt"
            "temperature_*.txt"
            "time_*.txt"
            "velocity_*.txt"
            "viscosity_*.txt"
            "scale_bcv.txt"
            "step*.txt"
            "Phi*.txt"
            "dPhi*.txt"
            "X_depletion*.txt"
            "*.bin*.txt"
            "bc*-1.txt"
            )

        # Faz um loop e usa find para evitar o erro "argument list too long"
        for pat in "${{patterns[@]}}"; do
            find . -maxdepth 1 -type f -name "$pat" -exec zip -u "$DIRNAME.zip" {{}} +
        done
    '''
with open('zipper.sh', 'w') as f:
    for line in zipper.split('\n'):
        line = line.strip()
        if len(line):
            f.write(' '.join(line.split()) + '\n')

clean = f'''
        #!/usr/bin/env bash

        # Lista de padrões
        patterns=(
            "bc_velocity_*.txt"
            "density_*.txt"
            "heat_*.txt"
            "pressure_*.txt"
            "sp_surface_global_*.txt"
            "strain_*.txt"
            "temperature_*.txt"
            "time_*.txt"
            "velocity_*.txt"
            "viscosity_*.txt"
            "scale_bcv.txt"
            "step*.txt"
            "Phi*.txt"
            "dPhi*.txt"
            "X_depletion*.txt"
            "*.bin*.txt"
            "bc*-1.txt"
            )

        # Para cada padrão, procurar e remover com segurança
        for pat in "${{patterns[@]}}"; do
            find . -maxdepth 1 -type f -name "$pat" -exec rm -f {{}} +
        done
    '''
with open('clean.sh', 'w') as f:
    for line in clean.split('\n'):
        line = line.strip()
        if len(line):
            f.write(' '.join(line.split()) + '\n')

#zip input files
filename = 'inputs_'+path[-1]+'.zip'
files_list = ' infos*.txt interfaces.txt param.txt input*_0.txt run*.sh vel*.txt scale_bcv.txt *.png precipitation.txt climate.txt zipper.sh clean.sh'
os.system('zip '+filename+files_list)