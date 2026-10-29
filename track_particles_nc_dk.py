########################################################################
# BEFORE RUNNING THIS SCRIPT:                                          #
#  - CREATE THE .nc FILES FROM .txt MANDYOC OUTPUTS                    #
#    (tapioca's meshes2netcdf.jl)                                      #
#  - CREATE THE PARTICLES .nc FILE WITH tapioca's particles2netcdf.jl  #
#    (particles_trajectories.nc) OR WITH StepsNETCDF.jl (_steps.nc)    #
#  - RUN THIS SCRIPT IN THE SAME FOLDER WHERE THE .nc FILES ARE        #
#                                                                      #
# Same as track_particles_parallel_dk.py, but reads the particles from #
# the .nc file instead of the steps/step*.txt files. The output        #
# (_track_xzPT_all_steps.nc) has the same structure.                   #
#                                                                      #
# particles_trajectories.nc only has the particles that exist in its   #
# reference time step: create it with the final step of the tracking   #
# (step_final below) as reference to select all particles there.       #
########################################################################

import numpy as np
import os
import xarray as xr
import subprocess
import multiprocessing #needed to run pymp in mac
multiprocessing.set_start_method('fork') #needed to run pymp in mac
import pymp
import sys
path = os.getcwd().split('/')
machine_path = f'/{path[1]}/{path[2]}' #cat the /home/user/ or /Users/user from system using path
sys.path.append(os.path.abspath(f"{machine_path}/opt/double_keel_article"))
sys.path.append(os.path.dirname(os.path.abspath(__file__))) # the repository folder, if the script is run from it
from functions.mandyoc_nc import open_mandyoc, open_particles # reads tapioca and old .nc files with the same structure

model_path = os.getcwd()

#reading dataset
Tdataset = open_mandyoc(model_path, "temperature")
Pdataset = open_mandyoc(model_path, "pressure")
Sdataset = open_particles(model_path) # particles_trajectories.nc (tapioca) or _steps.nc (StepsNETCDF.jl)

#getting the dimensions
Nx = int(Tdataset.nx)
Nz = int(Tdataset.nz)
Lx = float(Tdataset.lx)
Lz = float(Tdataset.lz)

print(
    "nx:", Nx, "\n",
    "nz:", Nz, "\n",
    "Lx:", Lx, "\n",
    "Lz:", Lz
)

xi = np.linspace(0,Lx,Nx)
zi = np.linspace(-Lz,0,Nz)
xx,zz = np.meshgrid(xi,zi)

dx = Lx/(Nx-1)
dz = Lz/(Nz-1)

#flag to track the particles until the middle of simulation (in time) in case of fast extension rate
# take_specific_time = True
take_specific_time = False

# take_asthenosphere = True
take_asthenosphere = False


if(take_specific_time):
    instant_to_take = 450 #Myr
    idx = (np.abs(Tdataset.time.values - instant_to_take)).argmin()
    step_final = Tdataset.step.values[idx]
else:
    step_final = Tdataset.step.values[-1] #14000

#index of each step of Tdataset inside the particles file
steps_nc = Sdataset.step.values

def steps_nc_index(step):
    return int(np.where(steps_nc == step)[0][0])

if(step_final not in steps_nc):
    raise ValueError(f"Final step {step_final} not found in the particles file")

reference_step = Sdataset.attrs.get("reference_timestep")
if(reference_step is not None and int(reference_step) != step_final):
    print(f"WARNING: the particles file was created with reference step {reference_step}, not with the final step {step_final}.")
    print(f"         Particles that do not exist in step {reference_step} cannot be selected.")

#Reading the data of final step
final = Sdataset.isel(time=steps_nc_index(step_final))
exists = np.isfinite(final.x.values)
x = final.x.values[exists]
z = final.z.values[exists]
id_vec = final.particle.values[exists]
layer_vec = final.layer.values[exists]


###########################################################################
#Selecting the particles to track                                         #
# All particles above search_thickness and between x_begin and x_end [km] #
# and does not belong to air (cc==6)                                      #
###########################################################################

h_air = 40.0e3
# search_thickness = 5.0e3
# search_thickness = 10.0e3
# search_thickness = 12.0e3
# search_thickness = 15.0e3
#for Claudio
search_thickness = 50.0e3
# search_thickness = 35.0e3

# x_begin = 0.0e3
# x_end = Lx
# x_begin = 700.0e3
# x_end = 1400.0e3

x_begin = 500.0e3
x_end = 1600.0e3

# x_begin = 600.0e3
# x_end = 1400.0e3

#claudio
# x_begin = 750.0e3
# x_end = 1150.0e3

# double_keel = False
double_keel = True

if(double_keel):
    asthenosphere_code = 0
    lower_craton_code = 1
    upper_craton_code = 2
    mantle_lithosphere_code = 3
    lower_crust1_code = 4
    seed_code = 5
    lower_crust2_code = 6
    upper_crust_code = 7
    decolement_code = 8
    sediments_code = 9
    air_code = 10
else:
    asthenosphere_code = 0
    mantle_lithosphere_code = 1
    lower_crust1_code = 2
    seed_code = 3
    lower_crust2_code = 4
    upper_crust_code = 5
    decolement_code = 6
    sediments_code = 7
    air_code = 8


if(take_asthenosphere):
    cond = (z>-(h_air + search_thickness)) & (x>=x_begin) & (x<=x_end) & (layer_vec<air_code) & (layer_vec>=asthenosphere_code) #to take particles from asthenosphere and lithosphere.
else:
    cond = (z>-(h_air + search_thickness)) & (x>=x_begin) & (x<=x_end) & (layer_vec<air_code) & (layer_vec>asthenosphere_code) #to take particles from lithosphere

part_selec = id_vec[cond]
layers_selec = layer_vec[cond] #get the layer numbers of the selected particles
n_tracked = np.size(part_selec)

if(n_tracked == 0):
    raise ValueError("No particles selected. Check search_thickness, x_begin, x_end and the reference step of the particles file.")

n_tracked_ast = layers_selec[layers_selec==asthenosphere_code].size
if(double_keel):
    n_tracked_lower_craton = layers_selec[layers_selec==lower_craton_code].size
    n_tracked_upper_craton = layers_selec[layers_selec==upper_craton_code].size
n_tracked_mantle_lithosphere = layers_selec[layers_selec==mantle_lithosphere_code].size
n_tracked_lower_crust1 = layers_selec[layers_selec==lower_crust1_code].size
n_tracked_lower_crust2 = layers_selec[layers_selec==lower_crust2_code].size
n_tracked_upper_crust = layers_selec[layers_selec==upper_crust_code].size
n_tracked_decolement = layers_selec[layers_selec==decolement_code].size
n_tracked_sediments = layers_selec[layers_selec==sediments_code].size

#Reading backwards in time the P and T data
start = int(Tdataset.time.values[0])
end = int(Tdataset.time[:idx].size) if take_specific_time else int(Tdataset.time.size - 1)

step = 1

missing_steps = np.setdiff1d(Tdataset.step.values[start:end+1], steps_nc)
if(missing_steps.size > 0):
    raise ValueError(f"Steps not found in the particles file: {missing_steps}")

#Reading the positions of the selected particles in every step before the parallel part
#(HDF5 files must not be read by processes forked after opening them)
part_idx = np.searchsorted(Sdataset.particle.values, part_selec)
xtrack = np.full((end+1, n_tracked), np.nan)
ztrack = np.full((end+1, n_tracked), np.nan)
for i in range(start, end+1):
    k = steps_nc_index(Tdataset.step.values[i])
    xtrack[i] = Sdataset.x.isel(time=k).values[part_idx]
    ztrack[i] = Sdataset.z.isel(time=k).values[part_idx]

#we need a shared pymp dictionary to store data.
vecx_track_evolution = pymp.shared.dict()
vecz_track_evolution = pymp.shared.dict()
present_pressure_evolution = pymp.shared.dict()
present_temperature_evolution = pymp.shared.dict()
all_time = pymp.shared.list()
all_step = pymp.shared.list()

with pymp.Parallel() as p:
    for i in p.range(end, start-step, -step):
        temperature = Tdataset.temperature[i].values
        pressure = Pdataset.pressure[i].values
        time_current = Tdataset.time.values[i]
        step_current = Tdataset.step.values[i]
        all_time.append(time_current)
        all_step.append(step_current)

        #tracking x, z position (always n_tracked values, in the order of part_selec; NaN where the particle does not exist)
        vecx_track = xtrack[i]
        vecz_track = ztrack[i]
        found = np.isfinite(vecx_track)

        #interpolating T and P only for the particles found
        present_temperature = np.full(n_tracked, np.nan)
        present_pressure = np.full(n_tracked, np.nan)

        xv = vecx_track[found]
        zv = vecz_track[found]
        jm = np.clip((xv // dx).astype(int), 0, Nx-2)
        im = np.clip(((zv + Lz) // dz).astype(int), 0, Nz-2)
        xw = (xv - xx[im, jm]) / dx
        zw = (zv - zz[im, jm]) / dz

        present_temperature[found] = temperature[im,jm]*(1.-xw)*(1.-zw) + \
                                     temperature[im,jm+1]*xw*(1.-zw) + \
                                     temperature[im+1,jm]*(1.-xw)*zw + \
                                     temperature[im+1,jm+1]*xw*zw

        present_pressure[found] = pressure[im,jm]*(1.-xw)*(1.-zw) + \
                                  pressure[im,jm+1]*xw*(1.-zw) + \
                                  pressure[im+1,jm]*(1.-xw)*zw + \
                                  pressure[im+1,jm+1]*xw*zw

        if (~found).sum() > 0:
            print(f"step {step_current}: {(~found).sum()} partículas não encontradas (NaN)")

        vecx_track_evolution[i] = vecx_track
        vecz_track_evolution[i] = vecz_track
        present_pressure_evolution[i] = present_pressure / 1.0e6
        present_temperature_evolution[i] = present_temperature

        np.savetxt(f"track_xzPT_step_{str(Tdataset.step.values[i]).zfill(6)}.txt", np.c_[vecx_track, vecz_track, present_pressure/1.0E6, present_temperature], fmt="%.5f")

vecx_track_dict = dict(vecx_track_evolution)
vecz_track_dict = dict(vecz_track_evolution)
present_pressure_dict = dict(present_pressure_evolution)
present_temperature_dict = dict(present_temperature_evolution)

all_vecx_track = []
all_vecz_track = []
all_present_pressure = []
all_present_temperature = []

all_step = sorted(all_step)
all_time = sorted(all_time)

for i,j,k,l in zip(sorted(vecx_track_dict), sorted(vecz_track_dict), sorted(present_pressure_dict), sorted(present_temperature_dict)):
    all_vecx_track.extend(vecx_track_dict[i])
    all_vecz_track.extend(vecz_track_dict[j])
    all_present_pressure.extend(present_pressure_dict[k])
    all_present_temperature.extend(present_temperature_dict[l])

print(f"len of:\n all_vecx_track {len(all_vecx_track)}\n all_vecz_track {len(all_vecz_track)}\n all_present_pressure {len(all_present_pressure)}\n all_present_temperature {len(all_present_temperature)}\n all_time {len(all_time)}\n all_step {len(all_step)}")
# Creatiing the xarray dataset with the tracked particles (same structure as track_particles_parallel_dk.py)
ds = xr.Dataset(
    {
        "xtrack": (["index"], all_vecx_track[::-1]),
        "ztrack": (["index"], all_vecz_track[::-1]),
        "ptrack": (["index"], all_present_pressure[::-1]),
        "ttrack": (["index"], all_present_temperature[::-1]),
        "step": all_step[::-1],
        "time": all_time[::-1],
        "ntracked": int(n_tracked),
        "particles_layers": layers_selec[::-1]
    },
    coords={
        "index": np.arange(len(all_vecx_track))
    }
)

ds.to_netcdf("_track_xzPT_all_steps.nc")

model_name = model_path.split('/')[-1]
##################################
print(f"Double keel model") if double_keel else print("Homogeneous model")
print(f"Particles file reference step: {reference_step}")
print(f"Selecting particles between {x_begin/1000} and {x_end/1000} km, above {search_thickness/1000} km")
if(take_specific_time):
    print(f"Track particles until {instant_to_take} Myr")
print("Final step: ", step_final)
print(f'Start idx: {start}, End idx: {end}, time of end: {Tdataset.time.values[end]}')

print(f"x size: {np.size(x)}")
print(f"Number of tracked particles (n): {np.size(part_selec)}, where:")
print(f"Asthenosphere: {n_tracked_ast}")
if(double_keel):
    print(f"Lower Craton: {n_tracked_lower_craton}")
    print(f"Upper Craton: {n_tracked_upper_craton}")
print(f"Mantle Lithosphere: {n_tracked_mantle_lithosphere}")
print(f"Lower Crust: {n_tracked_lower_crust1 + n_tracked_lower_crust2}")
print(f"Upper Crust: {n_tracked_upper_crust}")
print(f"Decolement: {n_tracked_decolement}")
print(f"Sediments: {n_tracked_sediments}\n")

print(f"n_tracked x len(all_time) = {n_tracked}*{len(all_time)} = {n_tracked*len(all_time)}")

track_infos = ["Tracking infos:"]
track_infos.append(' ')
track_infos.append(f"Double keel model") if double_keel else track_infos.append("Homogeneous model")
track_infos.append(f"Particles read from the .nc file (reference step: {reference_step})")
track_infos.append(' ')
track_infos.append('Selected Region:')
track_infos.append(f"\tSelecting particles between {x_begin/1000} and {x_end/1000} km, above {search_thickness/1000} km")
track_infos.append(' ')
if(take_specific_time):
    track_infos.append(f"Track particles until {instant_to_take} Myr")
track_infos.append(f"Final step: {step_final}")
track_infos.append(f'Start idx: {start}, End idx: {end}, time of end: {Tdataset.time.values[end]}')
track_infos.append(' ')
track_infos.append(f"x size: {np.size(x)}")
track_infos.append(f"Number of tracked particles (n): {np.size(part_selec)}, where:")
track_infos.append(f"Asthenosphere: {n_tracked_ast}")
if(double_keel):
    track_infos.append(f"Lower Craton: {n_tracked_lower_craton}")
    track_infos.append(f"Upper Craton: {n_tracked_upper_craton}")
track_infos.append(f"Mantle Lithosphere: {n_tracked_mantle_lithosphere}")
track_infos.append(f"Lower Crust: {n_tracked_lower_crust1 + n_tracked_lower_crust2}")
track_infos.append(f"Upper Crust: {n_tracked_upper_crust}")
track_infos.append(f"Decolement: {n_tracked_decolement}")
track_infos.append(f"Sediments: {n_tracked_sediments}\n")
track_infos.append(' ')
track_infos.append(f"len of:\n all_vecx_track {len(all_vecx_track)}\n all_vecz_track {len(all_vecz_track)}\n all_present_pressure {len(all_present_pressure)}\n all_present_temperature {len(all_present_temperature)}\n all_time {len(all_time)}\n all_step {len(all_step)}")
track_infos.append(f"n_tracked x len(all_time) = {n_tracked}*{len(all_time)} = {n_tracked*len(all_time)}")
np.savetxt(f"infos_track_{model_name}.txt", track_infos, fmt="%s")

##################################

# Compressing the files
print("Zipping files")
subprocess.run(f"zip {model_name}.zip track*.txt", shell=True, check=True, capture_output=True, text=True)
print("Files zipped")
