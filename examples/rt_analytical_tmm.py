import sys
import os
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(1,r"C:\Users\arson\Documents\solcore5_fork")

from solcore.structure import Layer
from solcore import material
import numpy as np
import matplotlib.pyplot as plt
import time

# rayflare imports
from rayflare.textures.standard_rt_textures import planar_surface, regular_pyramids
from rayflare.structure import Interface, BulkLayer, Structure
from rayflare.matrix_formalism import process_structure, calculate_RAT
from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure

import seaborn as sns
from cycler import cycler

# Thickness of bottom Ge layer
bulkthick = 10e-6

wavelengths = np.linspace(300, 1850, 60) * 1e-9
pal = sns.color_palette("husl", len(wavelengths))
cols = cycler("color", pal)

params = {"axes.prop_cycle": cols}

plt.rcParams.update(params)

# set options
options = default_options()
options.wavelength = wavelengths
options.project_name = "rt_tmm_comparisons"
options.n_rays = 100000
options.n_theta_bins = 50
options.lookuptable_angles = 200
options.parallel = True
options.I_thresh = 1e-3
options.bulk_profile = False
options.randomize_surface = True
options.periodic = True
options.theta_in = 0
options.n_jobs = -3
options.depth_spacing_bulk = 1e-7

# set up Solcore materials
# ITO = material("ITO")()
# print(ITO.n_path)
# ITO.n_path = "blahblah"
# ITO.load_n_data()
# assert(1==0)
Ge = material("Ge")()
Si = material("Si")()
GaAs = material("GaAs")()
GaInP = material("GaInP")(In=0.5)
Ag = material("Ag")()
SiN = material("Si3N4")()
Air = material("Air")()
Ta2O5 = material("TaOx1")()  # Ta2O5 (SOPRA database)
MgF2 = material("MgF2")()  # MgF2 (SOPRA database)

# front_materials = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5), Layer(464e-9, GaInP), Layer(1682e-9, GaAs)]

front_materials = [Layer(100e-9, SiN)]
back_materials = [Layer(100e-9, SiN)]

# RT/TMM, matrix framework


bulk_Ge = BulkLayer(bulkthick, Ge, name="Ge_bulk")  # bulk thickness in m
bulk_Si = BulkLayer(180e-6, Si, name="Si_bulk")  # bulk thickness in m

## RT with TMM lookup tables

surf_pyr = regular_pyramids(upright=False)  # [texture, flipped texture]
surf_pyr_upright = regular_pyramids(upright=True)
surf_planar = planar_surface()

front_surf = Interface(
    "TMM", layers=front_materials, texture=surf_planar, name="GaInP_GaAs_TMM", coherent=True, prof_layers=[3, 4]
)

front_surf_pyr = Interface(
    "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
)
front_surf_pyr.width_differentials = [front_surf_pyr.widths[0]*0.2] 

back_surf = Interface("RT_TMM", layers=back_materials, texture=surf_pyr, name="SiN_RT_TMM", coherent=True)

back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_pyr, name="SiN_TMM", coherent=True)
back_surf_planar.width_differentials = [back_surf_planar.widths[0]*0.2] 


SC = Structure([front_surf_pyr, bulk_Si, back_surf_planar], incidence=Air, transmission=Air)

times = []
# with saving: 1s, no saving: 0.65s
# on laptop no saving: 0.85s; of which calculate_RAT takes up 0.15s
for iter in range(4):
    t1 = time.time()
    process_structure(SC, options, overwrite=True)
    results_RT = calculate_RAT(SC, options)
    times.append(time.time()-t1)

RAT = results_RT[0]
print(RAT)
plt.plot(RAT['wl'],RAT['R'][0])
plt.plot(RAT['wl'],RAT['T'][0])
plt.show()

print(times[1:])

assert(1==0)

results_per_pass_RT = results_RT[1]
prof_front = results_RT[2][0]

# sum over passes
results_per_layer_front_RT = np.sum(results_per_pass_RT["a"][0], 0)

front_surf_rt = planar_surface(interface_layers=front_materials, prof_layers=[3, 4])  # pyramid size in microns

front_surf_rt_pyr = regular_pyramids(interface_layers=front_materials, prof_layers=[3, 4])  # pyramid size in microns

back_surf_rt = regular_pyramids(upright=False, interface_layers=back_materials)  # pyramid size in microns

back_surf_rt_planar = planar_surface(interface_layers=back_materials)  # pyramid size in microns

rtstr = rt_structure([front_surf_rt, back_surf_rt_planar], [Ge], [bulkthick], Air, Ag, options, use_TMM=True)
# RT + TMM

options.n_rays = 4000

result_RT_only = rtstr.calculate(options)

rt_front = result_RT_only["interface_profiles"][0]

plt.figure()
plt.plot(wavelengths * 1e9, result_RT_only["R"], label="RT")
plt.plot(wavelengths * 1e9, results_RT[0]["R"][0], label="RT + redist")
plt.plot(wavelengths * 1e9, result_RT_only["T"], "--", label="RT")
plt.plot(wavelengths * 1e9, results_RT[0]["T"][0], "--", label="RT + redist")
plt.plot(wavelengths * 1e9, result_RT_only["A_per_interface"][0])
plt.plot(wavelengths * 1e9, results_per_layer_front_RT, "--")
plt.plot(wavelengths * 1e9, result_RT_only["A_per_layer"][:, 0])
plt.plot(wavelengths * 1e9, results_RT[0]["A_bulk"][0], "--")

plt.legend()
plt.show()

plt.figure()
plt.semilogy(rt_front.T, alpha=0.5)
plt.semilogy(prof_front.T, "--")
plt.ylim(1e-13, 0.1)
# plt.legend([str(x) for x in range(10)])
plt.show()

plt.figure()
plt.semilogy(results_RT[3][0][40:50].T / 1e9, alpha=0.5)
plt.semilogy(result_RT_only["profile"][40:50].T, "--")
plt.ylim(1e-17, 0.1)
plt.show()
