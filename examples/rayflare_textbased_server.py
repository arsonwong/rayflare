import time
import numpy as np
import os
import sys
import pandas as pd
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.insert(1,r"D:\Wavelabs\2023-12-24 mockup of PLQE fit\solcore5_20240324")
sys.path.insert(1,r"C:\Users\arson\Documents\solcore5_fork")

from solcore.structure import Layer
from solcore import material
from solcore.light_source import LightSource
from solcore.constants import q

from rayflare.textures import planar_surface, regular_pyramids
from rayflare.structure import Interface, BulkLayer, Structure, Roughness
from rayflare.matrix_formalism import calculate_RAT, process_structure
from rayflare.utilities import get_savepath
from rayflare.options import default_options
from rayflare.angles import theta_summary, make_angle_vector

from sparse import load_npz

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from cycler import cycler



# can define material by loading nk files made by Griddler, e.g. doped silicon
# still need to parameterize to silicon
# still need to treat FCA
# for FCA, what we can do is make rayflare model the overall Si absorption, including FCA
# in which case everything including absorption profile will be correct
# and then simply multiply overall absorption A by alpha(Si_BB)/[alpha(Si_BB)+alpha(Si_FCA)] to get 
# 

# MATERIAL "SiNx_" "SiNx_PECVD [Bak11].csv"
# LAYERSTACK 160e-9 "MgF2" 80e-9 IZO
# PYRAMIDS "surf" "elevation_angle" 55 "upright" True "random_positions" True
# PLANARSURFACE "surfplanar"


# python also has result = eval(expression)
# just literally spell out all the expressions in matlab

def create_new_layer(name, thickness, n_file_path, k_file_path=None):
    mat = material(name)()
    n_file_path = n_file_path.replace("\\", "/")
    mat.n_path = n_file_path    
    if k_file_path is not None:
        k_file_path = k_file_path.replace("\\", "/")
        mat.k_path = k_file_path
        mat.load_n_data()
        mat.load_k_data()
    else:
        mat.load_nk_data()
    layer = Layer(thickness*1e-9, mat)
    return layer

def bulk_profile(results, z_front):
    bulk_absorbed_front = results[0]['bulk_absorbed_front']
    bulk_absorbed_rear = results[0]['bulk_absorbed_rear']
    alphas = results[0]['alphas']
    abscos = results[0]['abscos']

    z_front_widths = 0.5*(z_front[2:]-z_front[:-2])
    z_front_widths = np.insert(z_front_widths, 0, 0.5*(z_front[1]-z_front[0]))
    z_front_widths = np.append(z_front_widths, 0.5*(z_front[-1]-z_front[-2]))
    absorption_profile_front = np.exp(-alphas[:,None,None] * z_front[None,None,:] / abscos[None, :, None])
    absorption_profile_integral = np.sum(absorption_profile_front*z_front_widths[None, None, :], axis=2)
    absorption_profile_front *= bulk_absorbed_front[:,:,None]/absorption_profile_integral[:,:,None]
    absorption_profile_front = np.sum(absorption_profile_front, axis=1)

    z_rear = z_front[-1] - z_front
    z_rear_widths = z_front_widths
    absorption_profile_rear = np.exp(-alphas[:,None,None] * z_rear[None,None,:] / abscos[None, :, None])
    absorption_profile_integral = np.sum(absorption_profile_rear*z_rear_widths[None, None, :], axis=2)
    absorption_profile_rear *= bulk_absorbed_rear[:,:,None]/absorption_profile_integral[:,:,None]
    absorption_profile_rear = np.sum(absorption_profile_rear, axis=1)
    plt.plot(z_front*1e6,absorption_profile_front[140,:]+absorption_profile_rear[140,:], label='WL=1000nm')
    plt.plot(z_front*1e6,absorption_profile_front[130,:]+absorption_profile_rear[130,:], label='WL=950nm')
    plt.xlabel('z (um)')
    plt.ylabel('absorption (arb unit)')
    plt.legend()
    plt.title('Absorption profile in Si')
    plt.show()

    return absorption_profile_front, absorption_profile_rear, z_front_widths

def layer_profile(results, z_front):
    results_per_pass = results[0]['results_per_pass']
    results_pero = np.sum(results_per_pass["a"][0], 0)[:, [5]]
    overall_A = results_pero[:,0] # just flatten

    Aprof = results[0]['Aprof']
    Aprof_front = Aprof[5][0] #layer1,side1
    Aprof_rear = Aprof[5][1] # backside 
    front_local_angles = results[0]['front_local_angles']
    rear_local_angles = results[0]['rear_local_angles']

    part1 = Aprof_front[:,:,0,None]*np.exp(Aprof_front[:,:,4,None]*z_front)
    part2 = Aprof_front[:,:,1,None]*np.exp(-Aprof_front[:,:,4,None]*z_front)
    part3 = (Aprof_front[:,:,2,None] + 1j * Aprof_front[:,:,3,None])*np.exp(1j * Aprof_front[:,:,5,None]*z_front)
    part4 = (Aprof_front[:,:,2,None] - 1j * Aprof_front[:,:,3,None])*np.exp(-1j * Aprof_front[:,:,5,None]*z_front)
    result = np.real(part1 + part2 + 0*part3 + 0*part4)
    absorption_profile_front = front_local_angles[:,:,None]*result
    absorption_profile_front = np.sum(absorption_profile_front,axis=1)

    z_front_widths = 0.5*(z_front[2:]-z_front[:-2])
    z_front_widths = np.insert(z_front_widths, 0, 0.5*(z_front[1]-z_front[0]))
    z_front_widths = np.append(z_front_widths, 0.5*(z_front[-1]-z_front[-2]))

    z_rear = z_front[-1]-z_front
    part1 = Aprof_rear[:,:,0,None]*np.exp(Aprof_rear[:,:,4,None]*z_rear)
    part2 = Aprof_rear[:,:,1,None]*np.exp(-Aprof_rear[:,:,4,None]*z_rear)
    part3 = (Aprof_rear[:,:,2,None] + 1j * Aprof_rear[:,:,3,None])*np.exp(1j * Aprof_rear[:,:,5,None]*z_rear)
    part4 = (Aprof_rear[:,:,2,None] - 1j * Aprof_rear[:,:,3,None])*np.exp(-1j * Aprof_rear[:,:,5,None]*z_rear)
    result = np.real(part1 + part2 + 0*part3 + 0*part4)
    absorption_profile_rear = rear_local_angles[:,:,None]*result
    absorption_profile_rear = np.sum(absorption_profile_rear,axis=1)

    absorption_profile_integral = np.sum((absorption_profile_front+absorption_profile_rear)*z_front_widths[None, :], axis=1)
    absorption_profile_front *= overall_A[:,None]/absorption_profile_integral[:,None]
    absorption_profile_rear *= overall_A[:,None]/absorption_profile_integral[:,None]

    plt.plot(z_front,absorption_profile_front[60]+absorption_profile_rear[60], label='WL=600nm')
    plt.plot(z_front,absorption_profile_front[80]+absorption_profile_rear[80], label='WL=700nm')
    plt.xlabel('z (nm)')
    plt.ylabel('absorption (arb unit)')
    plt.legend()
    plt.title('Absorption profile in perovskite')
    plt.show()

    return absorption_profile_front, absorption_profile_rear, z_front_widths




pal = sns.cubehelix_palette()

cols = cycler("color", pal)

params = {
    "legend.fontsize": "small",
    "axes.labelsize": "small",
    "axes.titlesize": "small",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "axes.prop_cycle": cols,
}

plt.rcParams.update(params)

cur_path = os.path.dirname(os.path.abspath(__file__))
# new materials from data (only need to add once, uncomment following lines to do so:

# from solcore.material_system import create_new_material
# # create_new_material('Perovskite_CsBr_1p6eV', os.path.join(cur_path, 'data/CsBr10p_1to2_n_shifted.txt'), os.path.join(cur_path, 'data/CsBr10p_1to2_k_shifted.txt'))
# create_new_material('Perovskite_CsBr_1p6eV', os.path.join(cur_path, 'data/CsBr10p_1to2_n_shifted.txt'), os.path.join(cur_path, 'data/CsBr10p_1to2_k_shifted_mod.txt'))

# create_new_material('front_ITO', os.path.join(cur_path, 'data/model_med_back_ito_n.txt'), os.path.join(cur_path, 'data/model_med_back_ito_k.txt'))
# create_new_material('ITO_lowdoping', os.path.join(cur_path, 'data/model_heavy_back_ito_n.txt'), os.path.join(cur_path, 'data/model_heavy_back_ito_k.txt'))
# # create_new_material('Ag_Jiang', os.path.join(cur_path, 'data/Ag_UNSW_n.txt'), os.path.join(cur_path, 'data/Ag_UNSW_k.txt'))
# # create_new_material('aSi_i', os.path.join(cur_path, 'data/model_i_a_silicon_n.txt'),os.path.join(cur_path, 'data/model_i_a_silicon_k.txt'))
# # create_new_material('aSi_p', os.path.join(cur_path, 'data/model_p_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_p_a_silicon_k.txt'))
# # create_new_material('aSi_n', os.path.join(cur_path, 'data/model_n_a_silicon_n.txt'), os.path.join(cur_path, 'data/model_n_a_silicon_k.txt'))
# # create_new_material('MgF2_RdeM', os.path.join(cur_path, 'data/MgF2_RdeM_n.txt'), os.path.join(cur_path, 'data/MgF2_RdeM_k.txt'))
# # create_new_material('C60', os.path.join(cur_path, 'data/C60_Ren_n.txt'), os.path.join(cur_path, 'data/C60_Ren_k.txt'))
# # create_new_material('IZO', os.path.join(cur_path, 'data/IZO_Ballif_rO2_10pcnt_n.txt'), os.path.join(cur_path, 'data/IZO_Ballif_rO2_10pcnt_k.txt'))


# matrix multiplication
wavelengths = np.arange(300,1201,5) * 1e-9
# wavelengths = np.linspace(300,1200,50) * 1e-9

options = default_options()
options.wavelength = wavelengths
# options.only_incidence_angle = True
# options.lookuptable_angles = 200
# options.parallel = True
options.project_name = "perovskite_Si_example"
options.n_rays = 2000
options.n_theta_bins = 30
options.nx = 2
options.ny = 2
options.depth_spacing = 1e-9
options.phi_symmetry = np.pi / 2
options.bulk_profile = False
options.detailed = True

Si = material("Si")()
Air = material("Air")()
MgF2 = material("MgF2_RdeM")()
ITO_back = material("ITO_lowdoping")()
ITO_front = material("front_ITO")()
Perovskite = material("Perovskite_CsBr_1p6eV")()
Ag = material("Ag_Jiang")()
aSi_i = material("aSi_i")()
aSi_p = material("aSi_p")()
aSi_n = material("aSi_n")()
LiF = material("LiF")()
IZO = material("IZO")()
C60 = material("C60")()

# materials with constant n, zero k. Layer width is in nm.
Spiro = [12, np.array([0, 1]), np.array([1.65, 1.65]), np.array([0, 0])]
SnO2 = [10, np.array([0, 1]), np.array([2, 2]), np.array([0, 0])]

# stack based on doi:10.1038/s41563-018-0115-4
# alter C60 layer to 1e-9 instead of 15e-9
front_materials = [
    Layer(160e-9, MgF2),
    Layer(80e-9, IZO),
    SnO2,
    Layer(5e-9, C60), 
    Layer(1e-9, LiF),
    Layer(500e-9, Perovskite),
    Layer(250e-9, ITO_front),
    Layer(6.5e-9, aSi_n),
    Layer(6.5e-9, aSi_i),
]

back_materials = [Layer(6.5e-9, aSi_i), Layer(6.5e-9, aSi_p), Layer(250e-9, ITO_back)]

def set_front_materials_thicknesses(thicknesses):
    SnO2 = [thicknesses[2], np.array([0, 1]), np.array([2, 2]), np.array([0, 0])]
    front_materials = [
        Layer(thicknesses[0]*1e-9, MgF2),
        Layer(thicknesses[1]*1e-9, IZO),
        SnO2,
        Layer(thicknesses[3]*1e-9, C60), 
        Layer(thicknesses[4]*1e-9, LiF),
        Layer(thicknesses[5]*1e-9, Perovskite),
        Layer(thicknesses[6]*1e-9, ITO_front),
        Layer(thicknesses[7]*1e-9, aSi_n),
        Layer(thicknesses[8]*1e-9, aSi_i),
    ]
    return front_materials

def set_back_materials_thicknesses(thicknesses):
    back_materials = [Layer(thicknesses[0]*1e-9, aSi_i), Layer(thicknesses[1]*1e-9, aSi_p), Layer(thicknesses[2]*1e-9, ITO_back)]
    return back_materials

def set_bulk_thickness(thickness):
    bulk_Si = BulkLayer(thickness*1e-6, Si, name="Si_bulk")  # bulk thickness in m
    return bulk_Si

def run_simulation(front_materials, front_roughness, back_materials, rear_roughness, surf, surf_back, bulk_Si):
    method = "RT_analytical_TMM"
    if surf[0].N.shape[0]==2: #planar
        method = "TMM"
    front_surf = Interface(
    method,
    texture=surf,
    layers=front_materials,
    name="Perovskite_aSi_widthcorr",
    coherent=True,
    prof_layers=[6] #hopefully with 1-indexed, that is pero
    )
    method = "RT_analytical_TMM"
    if surf_back[0].N.shape[0]==2: #planar
        method = "TMM"
    back_surf = Interface(method, texture=surf_back, layers=back_materials, name="aSi_ITO_2", coherent=True)

    list_ = [front_surf]
    if front_roughness is not None:
        list_.append(front_roughness)
    list_.append(bulk_Si)
    if rear_roughness is not None:
        list_.append(rear_roughness)
    list_.append(back_surf)
    SC = Structure(list_, incidence=Air, transmission=Ag)

    process_structure(SC, options, overwrite=True)
    results = calculate_RAT(SC, options)

    RAT = results[0]['RAT']
    results_per_pass = results[0]['results_per_pass']

    results_per_layer_back = np.sum(results_per_pass["a"][1], 0)

    R_per_pass = np.sum(results_per_pass["r"][0], 2)
    R_0 = R_per_pass[0]
    R_escape = np.sum(R_per_pass[1:, :], 0)

    # only select absorbing layers, sum over passes
    results_per_layer_front = np.sum(results_per_pass["a"][0], 0)[:, [0, 1, 3, 6, 7, 8]]
    results_pero = np.sum(results_per_pass["a"][0], 0)[:, [5]]
    A_pero = results_pero[:,0] # just flatten

    allres = np.flip(
        np.hstack(
            (R_0[:, None], R_escape[:, None], results_per_layer_front, results_per_layer_back, RAT["T"].T, results_pero, RAT["A_bulk"].T)
        ),
        1,
    )

    # calculated photogenerated current (Jsc with 100% EQE)

    spectr_flux = LightSource(
        source_type="standard", version="AM1.5g", x=wavelengths, output_units="photon_flux_per_m", concentration=1
    ).spectrum(wavelengths)[1]

    A_Si = RAT["A_bulk"][0]
    Jph_Si = q * np.trapz(RAT["A_bulk"][0] * spectr_flux, wavelengths) / 10  # mA/cm2
    Jph_Perovskite = q * np.trapz(results_pero[:,0] * spectr_flux, wavelengths) / 10  # mA/cm2



    pal = sns.cubehelix_palette(13, start=0.5, rot=-0.7)

    # plot total R, A, T
    fig = plt.figure(figsize=(5, 4))
    ax = plt.subplot(111)
    ax.stackplot(
        options["wavelength"] * 1e9,
        allres.T,
        labels=[
            "c-Si (bulk)",
            "Perovskite",
            "Ag",
            "rear ITO",
            "aSi-p",
            "aSi-i",
            "aSi-i",
            "aSi-n",
            "front ITO",
            "C$_{60}$",
            "IZO",
            "MgF$_2$",
            "R$_{escape}$",
            "R$_0$",
        ],
        colors=pal,
    )

    min_wl = np.ceil(np.min(wavelengths*1e9))
    max_wl = np.floor(np.max(wavelengths*1e9))
    min_wl = min_wl.astype(int)
    max_wl = max_wl.astype(int)

    lgd = ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("R/A/T")
    ax.set_xlim(300, 1200)
    ax.set_ylim(0, 1.0)
    ax.text(530, 0.5, "Perovskite: \n" + str(round(Jph_Perovskite, 1)) + " mA/cm$^2$", ha="center")
    ax.text(900, 0.5, "Si: \n" + str(round(Jph_Si, 1)) + " mA/cm$^2$", ha="center")

    plt.show()

    return results


input_file_path = 'logfile.txt'
output_file_path = 'output_log.txt'

with open(input_file_path, 'w') as file:
    pass  # Just opening the file is enough to erase its contents

with open(output_file_path, 'w') as file:
    pass  # Just opening the file is enough to erase its contents

with open(input_file_path, 'r') as input_file, open(output_file_path, 'a') as output_file:
    # Move to the end of the file
    input_file.seek(0, 2) 
    
    while True:
        line = input_file.readline()
        if not line:
            time.sleep(0.01)  # Sleep briefly before trying again
            continue
        print(f"New line: {line.strip()}")
        exec(line.strip())
        # try:
        #     exec(line.strip())
        # except Exception as e:
        #     # This block will catch any exception and print the error message
        #     print(f"An error occurred: {e}")
        #     break
        # Write the new line to the output file
        output_file.write(line)
        output_file.flush()  # Ensure the line is written to the file immediately
