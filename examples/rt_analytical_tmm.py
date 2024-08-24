import sys
import os
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.insert(1,r"D:\Wavelabs\2023-12-24 mockup of PLQE fit\solcore5_20240324")
sys.path.insert(1,r"C:\Users\arson\Documents\solcore5_fork")

from solcore.structure import Layer
from solcore import material
import numpy as np
import matplotlib.pyplot as plt
import time

# rayflare imports
from rayflare.textures.standard_rt_textures import planar_surface, regular_pyramids
from rayflare.structure import Interface, BulkLayer, Structure, Roughness
from rayflare.matrix_formalism import process_structure, calculate_RAT
from rayflare.options import default_options
from rayflare.ray_tracing import rt_structure

import seaborn as sns
from cycler import cycler

# Thickness of bottom Ge layer
bulkthick = 10e-6

wavelengths = np.linspace(200, 1400, 100) * 1e-9

pal = sns.color_palette("husl", len(wavelengths))
cols = cycler("color", pal)

params = {"axes.prop_cycle": cols}

plt.rcParams.update(params)

# set options
options = default_options()
options.only_incidence_angle = True
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

# Get the current directory
current_dir = os.getcwd()

# set up Solcore materials
Air = material("Air")()
material_names = ["Si_", "SiNx_", "SiO2_", "air_", "lossy_air_", "glass_", "heavy_ITO_", "ITO_", "aSip_", "aSin_", "aSii_", "Si_5_5e19_", "Si_1_2e10_", "Si_1_9e10_", "Al2O3_"]
dict = {element: index for index, element in enumerate(material_names)}
materials = []
pathnames = ["Si_Crystalline, 300 K [Gre08].csv", "SiNx_PECVD [Bak11].csv", "SiO2_[Rao19].csv", "air.csv", "lossy_air.csv", "glass.csv", "ITO_Sputtered 6.1e20 [Hol13].csv", "ITO_Sputtered 0.78e20 [Hol13].csv", "Si_Amorphous p [Hol12].csv", "Si_Amorphous n [Hol12].csv", "Si_Amorphous i [Hol12].csv", "Si_Crystalline_n_doped_5_5e19.csv", "Si_Crystalline_n_doped_1_2e20.csv", "Si_Crystalline_n_doped_1_9e20.csv", "Al2O3_ALD on Si [Kim97].csv"]
for i, name in enumerate(material_names):
    mat = material(material_names[i])()
    mat.n_path = os.path.join(current_dir, r"PVL_benchmark", pathnames[i])
    mat.k_path = mat.n_path
    if name == "ITO_":
        path_ = [{'parameter':0.17e20,'path':'ITO_Sputtered 0.17e20 [Hol13].csv'},
                 {'parameter':0.30e20,'path':'ITO_Sputtered 0.30e20 [Hol13].csv'},
                {'parameter':0.65e20,'path':'ITO_Sputtered 0.65e20 [Hol13].csv'},
                {'parameter':0.78e20,'path':'ITO_Sputtered 0.78e20 [Hol13].csv'},
                {'parameter':1.0e20,'path':'ITO_Sputtered 1.0e20 [Hol13].csv'},
                {'parameter':2.0e20,'path':'ITO_Sputtered 2.0e20 [Hol13].csv'},
                {'parameter':4.9e20,'path':'ITO_Sputtered 4.9e20 [Hol13].csv'},
                {'parameter':6.1e20,'path':'ITO_Sputtered 6.1e20 [Hol13].csv'}]
        for entry in path_:
            entry['path'] = os.path.join(current_dir, r"PVL_benchmark", entry['path'])
        mat.n_path = path_
        mat.k_path = path_
    mat.load_n_data()
    mat.load_k_data()
    materials.append(mat)

def sel_mat(name):
    return materials[dict[name]]


# front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
# back_materials = [Layer(101e-9, sel_mat("Si_"))]
# options["n_theta_bins"] = 100
# options["only_incidence_angle"] = True
# options["theta_in"] = 0*np.pi/180
# options["phi_in"] = 0*np.pi/180

# surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)

# RAT = results_RT[0]['RAT']
# wl = RAT['wl']*1e9
# R = np.array(RAT['R'][0])
# Tfirst = np.array(RAT['Tfirst'])
# A = 1 - R - Tfirst
# data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct16_results.csv"), skiprows=0, delimiter=',')
# data = np.array(data)
# plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
# plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
# plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
# plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
# plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
# plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
# plt.title('air-heavyITO(200nm)-Si, \nplanar, incident angle = 0')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()


# front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
# back_materials = [Layer(101e-9, sel_mat("Si_"))]
# options["n_theta_bins"] = 100
# options["only_incidence_angle"] = True
# options["theta_in"] = 10*np.pi/180
# options["phi_in"] = 0*np.pi/180

# surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)

# RAT = results_RT[0]['RAT']
# wl = RAT['wl']*1e9
# R = np.array(RAT['R'][0])
# Tfirst = np.array(RAT['Tfirst'])
# A = 1 - R - Tfirst
# data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct17_results.csv"), skiprows=0, delimiter=',')
# data = np.array(data)
# plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
# plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
# plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
# plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
# plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
# plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
# plt.title('air-heavyITO(200nm)-Si, \nplanar, incident angle = 10')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()


# front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
# back_materials = [Layer(101e-9, sel_mat("Si_"))]
# options["n_theta_bins"] = 100
# options["only_incidence_angle"] = True
# options["theta_in"] = 20*np.pi/180
# options["phi_in"] = 0*np.pi/180

# surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)

# RAT = results_RT[0]['RAT']
# wl = RAT['wl']*1e9
# R = np.array(RAT['R'][0])
# Tfirst = np.array(RAT['Tfirst'])
# A = 1 - R - Tfirst
# data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct18_results.csv"), skiprows=0, delimiter=',')
# data = np.array(data)
# plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
# plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
# plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
# plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
# plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
# plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
# plt.title('air-heavyITO(200nm)-Si, \nplanar, incident angle = 20')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()


# front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
# back_materials = [Layer(101e-9, sel_mat("Si_"))]
# options["n_theta_bins"] = 100
# options["only_incidence_angle"] = True
# options["theta_in"] = 30*np.pi/180
# options["phi_in"] = 0*np.pi/180

# surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)

# RAT = results_RT[0]['RAT']
# wl = RAT['wl']*1e9
# R = np.array(RAT['R'][0])
# Tfirst = np.array(RAT['Tfirst'])
# A = 1 - R - Tfirst
# data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct19_results.csv"), skiprows=0, delimiter=',')
# data = np.array(data)
# plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
# plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
# plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
# plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
# plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
# plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
# plt.title('air-heavyITO(200nm)-Si, \nplanar, incident angle = 30')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()



# front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
# back_materials = [Layer(101e-9, sel_mat("Si_"))]
# options["n_theta_bins"] = 100
# options["only_incidence_angle"] = True
# options["theta_in"] = 40*np.pi/180
# options["phi_in"] = 0*np.pi/180

# surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)

# RAT = results_RT[0]['RAT']
# wl = RAT['wl']*1e9
# R = np.array(RAT['R'][0])
# Tfirst = np.array(RAT['Tfirst'])
# A = 1 - R - Tfirst
# data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct20_results.csv"), skiprows=0, delimiter=',')
# data = np.array(data)
# plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
# plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
# plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
# plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
# plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
# plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
# plt.title('air-heavyITO(200nm)-Si, \nplanar, incident angle = 40')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()


# front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
# back_materials = [Layer(101e-9, sel_mat("Si_"))]
# options["n_theta_bins"] = 100
# options["only_incidence_angle"] = True
# options["theta_in"] = 50*np.pi/180
# options["phi_in"] = 0*np.pi/180

# surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)

# RAT = results_RT[0]['RAT']
# wl = RAT['wl']*1e9
# R = np.array(RAT['R'][0])
# Tfirst = np.array(RAT['Tfirst'])
# A = 1 - R - Tfirst
# data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct21_results.csv"), skiprows=0, delimiter=',')
# data = np.array(data)
# plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
# plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
# plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
# plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
# plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
# plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
# plt.title('air-heavyITO(200nm)-Si, \nplanar, incident angle = 50')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()


# front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
# back_materials = [Layer(101e-9, sel_mat("Si_"))]
# options["n_theta_bins"] = 100
# options["only_incidence_angle"] = True
# options["theta_in"] = 60*np.pi/180
# options["phi_in"] = 0*np.pi/180

# surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)

# RAT = results_RT[0]['RAT']
# wl = RAT['wl']*1e9
# R = np.array(RAT['R'][0])
# Tfirst = np.array(RAT['Tfirst'])
# A = 1 - R - Tfirst
# data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct22_results.csv"), skiprows=0, delimiter=',')
# data = np.array(data)
# plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
# plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
# plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
# plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
# plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
# plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
# plt.title('air-heavyITO(200nm)-Si, \nplanar, incident angle = 60')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()


# front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
# back_materials = [Layer(101e-9, sel_mat("Si_"))]
# options["n_theta_bins"] = 100
# options["only_incidence_angle"] = True
# options["theta_in"] = 70*np.pi/180
# options["phi_in"] = 0*np.pi/180

# surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)

# RAT = results_RT[0]['RAT']
# wl = RAT['wl']*1e9
# R = np.array(RAT['R'][0])
# Tfirst = np.array(RAT['Tfirst'])
# A = 1 - R - Tfirst
# data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct23_results.csv"), skiprows=0, delimiter=',')
# data = np.array(data)
# plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
# plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
# plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
# plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
# plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
# plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
# plt.title('air-heavyITO(200nm)-Si, \nplanar, incident angle = 70')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()


# front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
# back_materials = [Layer(101e-9, sel_mat("Si_"))]
# options["n_theta_bins"] = 100
# options["only_incidence_angle"] = True
# options["theta_in"] = 80*np.pi/180
# options["phi_in"] = 0*np.pi/180

# surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)

# RAT = results_RT[0]['RAT']
# wl = RAT['wl']*1e9
# R = np.array(RAT['R'][0])
# Tfirst = np.array(RAT['Tfirst'])
# A = 1 - R - Tfirst
# data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct24_results.csv"), skiprows=0, delimiter=',')
# data = np.array(data)
# plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
# plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
# plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
# plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
# plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
# plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
# plt.title('air-heavyITO(200nm)-Si, \nplanar, incident angle = 80')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()

# assert(1==0)


t1 = time.time()
options["only_incidence_angle"] = True
options["theta_in"] = 0*np.pi/180
options["phi_in"] = 0*np.pi/180
front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
back_materials = [Layer(101e-9, sel_mat("Si_"))]

surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
surf_planar = planar_surface()
front_surf_pyr = Interface(
    "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
)
front_surf_planar = Interface(
    "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
)
back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
SC = Structure([front_surf_pyr, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

process_structure(SC, options, overwrite=True)
results_RT = calculate_RAT(SC, options)
print("time: ", time.time()-t1)

RAT = results_RT[0]['RAT']
wl = RAT['wl']*1e9
R = np.array(RAT['R'][0])
Tfirst = np.array(RAT['Tfirst'])
A = 1 - R - Tfirst
data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct12_results.csv"), skiprows=0, delimiter=',')
data = np.array(data)
plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
plt.title('air-heavyITO(200nm)-Si, \nrandom pyramid texture 50 degrees, incident angle = 0')
plt.xlabel('Wavelength (nm)')
plt.ylabel('RAT')
plt.ylim(0, 1)
plt.legend(loc=0)
plt.show()
t1 = time.time()




t1 = time.time()
front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
back_materials = [Layer(101e-9, sel_mat("Si_"))]
options["n_theta_bins"] = 100
options["only_incidence_angle"] = True
options["theta_in"] = 14*np.pi/180
options["phi_in"] = 0*np.pi/180

surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
surf_planar = planar_surface()
front_surf_pyr = Interface(
    "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
)
front_surf_planar = Interface(
    "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
)
back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
SC = Structure([front_surf_pyr, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

process_structure(SC, options, overwrite=True)
results_RT = calculate_RAT(SC, options)
print("time: ", time.time()-t1)

RAT = results_RT[0]['RAT']
wl = RAT['wl']*1e9
R = np.array(RAT['R'][0])
Tfirst = np.array(RAT['Tfirst'])
A = 1 - R - Tfirst
data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct13_results.csv"), skiprows=0, delimiter=',')
data = np.array(data)
plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
plt.title('air-heavyITO(200nm)-Si, \nrandom pyramid texture 50 degrees, incident angle = 14')
plt.xlabel('Wavelength (nm)')
plt.ylabel('RAT')
plt.ylim(0, 1)
plt.legend(loc=0)
plt.show()
t1 = time.time()



t1 = time.time()
front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
back_materials = [Layer(101e-9, sel_mat("Si_"))]
options["n_theta_bins"] = 100
options["only_incidence_angle"] = True
options["theta_in"] = 28*np.pi/180
options["phi_in"] = 0*np.pi/180

surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
surf_planar = planar_surface()
front_surf_pyr = Interface(
    "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
)
front_surf_planar = Interface(
    "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
)
back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
SC = Structure([front_surf_pyr, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

process_structure(SC, options, overwrite=True)
results_RT = calculate_RAT(SC, options)
print("time: ", time.time()-t1)

RAT = results_RT[0]['RAT']
wl = RAT['wl']*1e9
R = np.array(RAT['R'][0])
Tfirst = np.array(RAT['Tfirst'])
A = 1 - R - Tfirst
data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct14_results.csv"), skiprows=0, delimiter=',')
data = np.array(data)
plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
plt.title('air-heavyITO(200nm)-Si, \nrandom pyramid texture 50 degrees, incident angle = 28')
plt.xlabel('Wavelength (nm)')
plt.ylabel('RAT')
plt.ylim(0, 1)
plt.legend(loc=0)
plt.show()
t1 = time.time()


t1 = time.time()
front_materials = [Layer(200e-9, sel_mat("heavy_ITO_"))]
back_materials = [Layer(101e-9, sel_mat("Si_"))]
options["n_theta_bins"] = 100
options["only_incidence_angle"] = True
options["theta_in"] = 50*np.pi/180
options["phi_in"] = 0*np.pi/180

surf_pyr_upright = regular_pyramids(upright=True, elevation_angle=50)
surf_planar = planar_surface()
front_surf_pyr = Interface(
    "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
)
front_surf_planar = Interface(
    "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
)
back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
SC = Structure([front_surf_pyr, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

process_structure(SC, options, overwrite=True)
results_RT = calculate_RAT(SC, options)
print("time: ", time.time()-t1)

RAT = results_RT[0]['RAT']
wl = RAT['wl']*1e9
R = np.array(RAT['R'][0])
Tfirst = np.array(RAT['Tfirst'])
A = 1 - R - Tfirst
data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct15_results.csv"), skiprows=0, delimiter=',')
data = np.array(data)
plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
plt.title('air-heavyITO(200nm)-Si, \nrandom pyramid texture 50 degrees, incident angle = 50')
plt.xlabel('Wavelength (nm)')
plt.ylabel('RAT')
plt.ylim(0, 1)
plt.legend(loc=0)
plt.show()
t1 = time.time()





front_materials = [Layer(65e-9, sel_mat("SiNx_")), Layer(20e-9, sel_mat("SiO2_"))]
back_materials = [Layer(101e-9, sel_mat("Si_"))]
options["n_theta_bins"] = 100
options["only_incidence_angle"] = True
options["theta_in"] = 0*np.pi/180
options["phi_in"] = 0*np.pi/180

surf_pyr_upright = regular_pyramids(upright=True)
surf_planar = planar_surface()
front_surf_pyr = Interface(
    "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
)
front_surf_planar = Interface(
    "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
)
back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
SC = Structure([front_surf_pyr, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

process_structure(SC, options, overwrite=True)
results_RT = calculate_RAT(SC, options)
print("time: ", time.time()-t1)

RAT = results_RT[0]['RAT']
wl = RAT['wl']*1e9
R = np.array(RAT['R'][0])
Tfirst = np.array(RAT['Tfirst'])
A = 1 - R - Tfirst
data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct1_results.csv"), skiprows=0, delimiter=',')
data = np.array(data)
plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
plt.title('air-SiNx(65nm)-SiO2(20nm)-Si, \nrandom pyramid texture, incident angle = 0')
plt.xlabel('Wavelength (nm)')
plt.ylabel('RAT')
plt.ylim(0, 1)
plt.legend(loc=0)
plt.show()
t1 = time.time()





front_materials = [Layer(65e-9, sel_mat("SiNx_")), Layer(20e-9, sel_mat("SiO2_"))]
back_materials = [Layer(101e-9, sel_mat("Si_"))]

surf_pyr_upright = regular_pyramids(upright=True)
surf_planar = planar_surface()
front_surf_pyr = Interface(
    "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
)
front_surf_planar = Interface(
    "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
)
back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
bulk_Si = BulkLayer(1800e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=Air, transmission=sel_mat("Si_"))

process_structure(SC, options, overwrite=True)
results_RT = calculate_RAT(SC, options)
print("time: ", time.time()-t1)

RAT = results_RT[0]['RAT']
wl = RAT['wl']*1e9
R = np.array(RAT['R'][0])
Tfirst = np.array(RAT['Tfirst'])
A = 1 - R - Tfirst
data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct5_results.csv"), skiprows=0, delimiter=',')
data = np.array(data)
plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
plt.title('air-SiNx(65nm)-SiO2(20nm)-Si, \nplanar front, incident angle = 0')
plt.xlabel('Wavelength (nm)')
plt.ylabel('RAT')
plt.ylim(0, 1)
plt.legend(loc=0)
plt.show()
t1 = time.time()




front_materials = [Layer(100e-9, sel_mat("heavy_ITO_")), Layer(100e-9, sel_mat("SiO2_")), Layer(100e-9, sel_mat("SiNx_"))]
back_materials = [Layer(101e-9, sel_mat("air_"))]

surf_pyr_upright = regular_pyramids(upright=True)
surf_planar = planar_surface()
front_surf_pyr = Interface(
    "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
)
front_surf_planar = Interface(
    "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
)
back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
bulk_Si = BulkLayer(1800e-6, Air, name="Si_bulk")  # bulk thickness in m, make very thick
SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=sel_mat("Si_"), transmission=Air)

process_structure(SC, options, overwrite=True)
results_RT = calculate_RAT(SC, options)
print("time: ", time.time()-t1)

RAT = results_RT[0]['RAT']
wl = RAT['wl']*1e9
R = np.array(RAT['R'][0])
Tfirst = np.array(RAT['Tfirst'])
A = 1 - R - Tfirst
data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct9_results.csv"), skiprows=0, delimiter=',')
data = np.array(data)
plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
plt.title('Si-SiO2(100nm)-SiNx(100nm)-ITO(100nm)-air, \nplanar texture, incident angle = 0')
plt.xlabel('Wavelength (nm)')
plt.ylabel('RAT')
plt.ylim(0, 1)
plt.legend(loc=0)
plt.show()
t1 = time.time()




front_materials = [Layer(100e-9, sel_mat("heavy_ITO_")), Layer(100e-9, sel_mat("SiO2_")), Layer(100e-9, sel_mat("SiNx_"))]
back_materials = [Layer(101e-9, sel_mat("air_"))]

surf_pyr_upright = regular_pyramids(upright=True)
surf_planar = planar_surface()
front_surf_pyr = Interface(
    "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
)
front_surf_planar = Interface(
    "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
)
back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
bulk_Si = BulkLayer(1800e-6, Air, name="Si_bulk")  # bulk thickness in m, make very thick
SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=sel_mat("Si_"), transmission=Air)

options["n_theta_bins"] = 100
options["theta_in"] = 16*np.pi/180
options["phi_in"] = 45*np.pi/180
process_structure(SC, options, overwrite=True)
results_RT = calculate_RAT(SC, options)
print("time: ", time.time()-t1)

RAT = results_RT[0]['RAT']
wl = RAT['wl']*1e9
R = np.array(RAT['R'][0])
Tfirst = np.array(RAT['Tfirst'])
A = 1 - R - Tfirst
data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct10_results.csv"), skiprows=0, delimiter=',')
data = np.array(data)
plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
plt.title('Si-SiO2(100nm)-SiNx(100nm)-ITO(100nm)-air, \nplanar texture, incident angle = 16/45')
plt.xlabel('Wavelength (nm)')
plt.ylabel('RAT')
plt.ylim(0, 1)
plt.legend(loc=0)
plt.show()
t1 = time.time()




front_materials = [Layer(100e-9, sel_mat("heavy_ITO_")), Layer(100e-9, sel_mat("SiO2_")), Layer(100e-9, sel_mat("SiNx_"))]
back_materials = [Layer(101e-9, sel_mat("air_"))]

surf_pyr_upright = regular_pyramids(upright=True)
surf_planar = planar_surface()
front_surf_pyr = Interface(
    "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
)
front_surf_planar = Interface(
    "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
)
back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
bulk_Si = BulkLayer(1800e-6, Air, name="Si_bulk")  # bulk thickness in m, make very thick
SC = Structure([front_surf_planar, bulk_Si, back_surf_planar], incidence=sel_mat("Si_"), transmission=Air)

options["n_theta_bins"] = 100
options["theta_in"] = 8*np.pi/180
options["phi_in"] = 67*np.pi/180
process_structure(SC, options, overwrite=True)
results_RT = calculate_RAT(SC, options)
print("time: ", time.time()-t1)

RAT = results_RT[0]['RAT']
wl = RAT['wl']*1e9
R = np.array(RAT['R'][0])
Tfirst = np.array(RAT['Tfirst'])
A = 1 - R - Tfirst
data = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark", r"teststruct11_results.csv"), skiprows=0, delimiter=',')
data = np.array(data)
plt.plot(wl, R, label='R(Rayflare)',color='red', linestyle='-')
plt.plot(wl, Tfirst, label='T(Rayflare)',color='green', linestyle='-')
plt.plot(wl, A, label='A(Rayflare)',color='purple', linestyle='-')
plt.plot(data[:,0], data[:,2], label='R(PVL)',color='red', linestyle='--')
plt.plot(data[:,0], data[:,4], label='T(PVL)',color='green', linestyle='--')
plt.plot(data[:,0], data[:,3], label='A(PVL)',color='purple', linestyle='--')
plt.title('Si-SiO2(100nm)-SiNx(100nm)-ITO(100nm)-air, \nplanar texture, incident angle = 8/67')
plt.xlabel('Wavelength (nm)')
plt.ylabel('RAT')
plt.ylim(0, 1)
plt.legend(loc=0)
plt.show()

t1 = time.time()

assert(1==0)

wavelengths = np.arange(300,1201,10) * 1e-9
options["wavelength"] = wavelengths

# for iter in range(2):
#     t1 = time.time()

#     options["n_theta_bins"] = 50
#     options["theta_in"] = 0.0
#     options["phi_in"] = 0.0
#     front_materials = [Layer(70e-9, sel_mat("ITO_")), Layer(5e-9, sel_mat("aSip_")), Layer(3e-9, sel_mat("aSii_"))]
#     back_materials = [Layer(3e-9, sel_mat("aSii_")), Layer(5e-9, sel_mat("aSin_")), Layer(70e-9, sel_mat("ITO_"))]

#     surf_pyr_upright = regular_pyramids(upright=True)
#     surf_planar = planar_surface()
#     front_surf_pyr = Interface(
#         "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
#     )
#     front_surf_pyr.width_differentials = [7e-9, 10e-10, 10e-10]
#     front_surf_pyr.nk_parameter_differentials = [10e20, None, None]
#     front_surf_planar = Interface(
#         "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
#     )
#     back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
#     back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
#     bulk_Si = BulkLayer(180e-6, sel_mat("Si_"), name="Si_bulk")  
#     # roughness = Roughness(np.pi/10)
#     roughness = Roughness(0)
#     linestyle = '-'
#     if iter==1:
#         linestyle = '-.'
#         SC = Structure([front_surf_pyr, roughness, bulk_Si, roughness, back_surf_pyr], incidence=Air, transmission=Air, light_trapping_onset_wavelength = 900e-9)
#     else:
#         SC = Structure([front_surf_pyr, bulk_Si, back_surf_pyr], incidence=Air, transmission=Air, light_trapping_onset_wavelength = 900e-9)

#     process_structure(SC, options, overwrite=True)
#     results_RT = calculate_RAT(SC, options)
#     print("time: ", time.time()-t1)

#     for iter in range(len(results_RT)):
#         RAT = results_RT[iter]['RAT']
#         wl = RAT['wl']*1e9
#         R = np.array(RAT['R'][0])
#         T = np.array(RAT['T'][0])
#         A = np.array(RAT['A_bulk'][0])
#         plt.plot(wl, R, color='red', linestyle=linestyle)
#         plt.plot(wl, T, color='green', linestyle=linestyle)
#         plt.plot(wl, A, color='purple', linestyle=linestyle)

#     for iter in range(len(SC.RAT1st['wl'])):
#         wl = SC.RAT1st['wl'][iter]*1e9
#         R1st = SC.RAT1st['R'][iter]
#         plt.plot(wl, R1st, color='red', linestyle=linestyle)
#         A1st = SC.RAT1st['T'][iter]
#         plt.plot(wl, A1st, color='purple', linestyle=linestyle)

# plt.title('HJT Cell')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()

# t1 = time.time()

# assert(1==0)



# # differential
# t1 = time.time()

# options["n_theta_bins"] = 50
# options["theta_in"] = 0.0
# options["phi_in"] = 0.0
# front_materials = [Layer(70e-9, sel_mat("ITO_"), nk_parameter=1e21), Layer(5e-9, sel_mat("aSip_")), Layer(3e-9, sel_mat("aSii_"))]
# back_materials = [Layer(3e-9, sel_mat("aSii_")), Layer(5e-9, sel_mat("aSin_")), Layer(70e-9, sel_mat("ITO_"))]

# surf_pyr_upright = regular_pyramids(upright=True)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_pyr.width_differentials = [30e-9, None, None]
# front_surf_pyr.nk_parameter_differentials = [1e21, None, None]
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# roughness = Roughness(np.pi/10)
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_pyr.width_differentials = [10e-10, 10e-10, 7e-9]
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(180e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_pyr, roughness, bulk_Si, roughness, back_surf_pyr], incidence=Air, transmission=Air, light_trapping_onset_wavelength = 900e-9)

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)
# print("time: ", time.time()-t1)

# for iter in range(len(results_RT)):
#     RAT = results_RT[iter]['RAT']
#     wl = RAT['wl']*1e9
#     R = np.array(RAT['R'][0])
#     T = np.array(RAT['T'][0])
#     A = np.array(RAT['A_bulk'][0])
#     plt.plot(wl, R, color='red', linestyle='-')
#     plt.plot(wl, T, color='green', linestyle='-')
#     plt.plot(wl, A, color='purple', linestyle='-')
#     if iter==1:
#         R_ = np.copy(R)
#         T_ = np.copy(T)
#         A_ = np.copy(A)

# for iter in range(len(SC.RAT1st['wl'])):
#     wl = SC.RAT1st['wl'][iter]*1e9
#     R1st = SC.RAT1st['R'][iter]
#     plt.plot(wl, R1st, color='red', linestyle='-')
#     A1st = SC.RAT1st['T'][iter]
#     plt.plot(wl, A1st, color='purple', linestyle='-')
#     if iter==1:
#         R1st_ = np.copy(R1st)
#         A1st_ = np.copy(A1st)


# front_materials = [Layer(100e-9, sel_mat("ITO_")), Layer(5e-9, sel_mat("aSip_")), Layer(3e-9, sel_mat("aSii_"))]
# back_materials = [Layer(3e-9, sel_mat("aSii_")), Layer(5e-9, sel_mat("aSin_")), Layer(70e-9, sel_mat("ITO_"))]

# surf_pyr_upright = regular_pyramids(upright=True)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# roughness = Roughness(np.pi/10)
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(180e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_pyr, roughness, bulk_Si, roughness, back_surf_pyr], incidence=Air, transmission=Air, light_trapping_onset_wavelength = 900e-9)

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)
# print("time: ", time.time()-t1)

# for iter in range(len(results_RT)):
#     RAT = results_RT[iter]['RAT']
#     wl = RAT['wl']*1e9
#     R = np.array(RAT['R'][0])
#     T = np.array(RAT['T'][0])
#     A = np.array(RAT['A_bulk'][0])
#     plt.plot(wl, R, color='red', linestyle='-.')
#     plt.plot(wl, T, color='green', linestyle='-.')
#     plt.plot(wl, A, color='purple', linestyle='-.')
#     print(np.max(np.abs(R-R_)))
#     print(np.max(np.abs(A-A_)))
#     print(np.max(np.abs(T-T_)))

# for iter in range(len(SC.RAT1st['wl'])):
#     wl = SC.RAT1st['wl'][iter]*1e9
#     R1st = SC.RAT1st['R'][iter]
#     plt.plot(wl, R1st, color='red', linestyle='-.')
#     A1st = SC.RAT1st['T'][iter]
#     plt.plot(wl, A1st, color='purple', linestyle='-.')
#     print(np.max(np.abs(R1st-R1st_)))
#     print(np.max(np.abs(A1st-A1st_)))



# plt.title('HJT Cell')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()

# t1 = time.time()


# assert(1==0)

# # differential
# t1 = time.time()

# options["n_theta_bins"] = 50
# options["theta_in"] = 0.0
# options["phi_in"] = 0.0
# front_materials = [Layer(70e-9, sel_mat("ITO_")), Layer(5e-9, sel_mat("aSip_")), Layer(3e-9, sel_mat("aSii_"))]
# back_materials = [Layer(3e-9, sel_mat("aSii_")), Layer(5e-9, sel_mat("aSin_")), Layer(70e-9, sel_mat("ITO_"))]

# surf_pyr_upright = regular_pyramids(upright=True)
# surf_planar = planar_surface()
# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_pyr.width_differentials = [7e-9, 10e-10, 10e-10]
# front_surf_planar = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="SiN_RT", coherent=True
# )
# back_surf_pyr = Interface("RT_analytical_TMM", layers=back_materials, texture=surf_pyr_upright, name="SiN_TMM", coherent=True)
# back_surf_pyr.width_differentials = [10e-10, 10e-10, 7e-9]
# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# bulk_Si = BulkLayer(180e-6, sel_mat("Si_"), name="Si_bulk")  # bulk thickness in m, make very thick
# SC = Structure([front_surf_pyr, bulk_Si, back_surf_pyr], incidence=Air, transmission=Air, light_trapping_onset_wavelength = None)

# process_structure(SC, options, overwrite=True)
# results_RT = calculate_RAT(SC, options)
# print("time: ", time.time()-t1)

# for iter in range(len(results_RT)):
#     RAT = results_RT[iter]['RAT']
#     wl = RAT['wl']*1e9
#     R = np.array(RAT['R'][0])
#     T = np.array(RAT['T'][0])
#     A = np.array(RAT['A_bulk'][0])
#     plt.plot(wl, R, color='red', linestyle='-')
#     plt.plot(wl, T, color='green', linestyle='-')
#     plt.plot(wl, A, color='purple', linestyle='-')

# plt.title('HJT Cell')
# plt.xlabel('Wavelength (nm)')
# plt.ylabel('RAT')
# plt.ylim(0, 1)
# plt.legend(loc=0)
# plt.show()

# t1 = time.time()


# assert(1==0)








# # print(ITO.n_data)
# # print(ITO.n(500))
# # print(ITO.k(500))
# # assert(1==0)

# # nkdata = np.loadtxt(os.path.join(current_dir, r"PVL_benchmark\ITO_Sputtered 0.78e20 [Hol13].csv"), skiprows=1, encoding='utf-8', delimiter=',')
# # wl = nkdata[:,0]
# # nk = nkdata[:,1] - 1j * nkdata[:,3]
# # print(wl)
# # assert(1==0)

# # ITO.n_path = r"..\PVL_benchmark\ITO_Sputtered 0.78e20 [Hol13].csv"
# # ITO.load_n_data()
# # assert(1==0)

# Ge = material("Ge")()
# # print(Ge.n)
# # print(Ge.k)
# # Ge.load_n_data()
# # print("boo")
# # print(Ge.n_data)
# # Ge.load_k_data()
# # print("boo2")
# # print(Ge.k_data)
# # assert(1==0)
# Si = material("Si")()
# GaAs = material("GaAs")()
# GaInP = material("GaInP")(In=0.5)
# Ag = material("Ag")()
# SiN = material("Si3N4")()
# Air = material("Air")()
# Ta2O5 = material("TaOx1")()  # Ta2O5 (SOPRA database)
# MgF2 = material("MgF2")()  # MgF2 (SOPRA database)

# # front_materials = [Layer(120e-9, MgF2), Layer(74e-9, Ta2O5), Layer(464e-9, GaInP), Layer(1682e-9, GaAs)]

# front_materials = [Layer(100e-9, SiN)]
# back_materials = [Layer(100e-9, SiN)]

# # RT/TMM, matrix framework


# bulk_Ge = BulkLayer(bulkthick, Ge, name="Ge_bulk")  # bulk thickness in m
# bulk_Si = BulkLayer(180e-6, Si, name="Si_bulk")  # bulk thickness in m

# ## RT with TMM lookup tables

# surf_pyr = regular_pyramids(upright=False)  # [texture, flipped texture]
# surf_pyr_upright = regular_pyramids(upright=True)
# surf_planar = planar_surface()

# front_surf = Interface(
#     "TMM", layers=front_materials, texture=surf_planar, name="GaInP_GaAs_TMM", coherent=True, prof_layers=[3, 4]
# )

# front_surf_pyr = Interface(
#     "RT_analytical_TMM", layers=front_materials, texture=surf_pyr_upright, name="SiN_RT", coherent=True
# )
# front_surf_pyr.width_differentials = [front_surf_pyr.widths[0]*0.2] 

# back_surf = Interface("RT_TMM", layers=back_materials, texture=surf_pyr, name="SiN_RT_TMM", coherent=True)

# back_surf_planar = Interface("TMM", layers=back_materials, texture=surf_planar, name="SiN_TMM", coherent=True)
# back_surf_planar.width_differentials = [back_surf_planar.widths[0]*0.2] 



# SC = Structure([front_surf_pyr, bulk_Si, back_surf_planar], incidence=Air, transmission=Air)

# times = []
# # with saving: 1s, no saving: 0.65s
# # on laptop no saving: 0.85s; of which calculate_RAT takes up 0.15s
# # 2024-04-07 tested differential cases (total 3 scenarios including baseline): 1.4s
# # compared to 1 scenario: 0.8898s
# for iter in range(4):
#     t1 = time.time()
#     process_structure(SC, options, overwrite=True)
#     results_RT = calculate_RAT(SC, options)
#     times.append(time.time()-t1)

# for i in range(len(results_RT)):
#     RAT = results_RT[i]['RAT']
#     plt.plot(RAT['wl'],RAT['R'][0])
#     plt.plot(RAT['wl'],RAT['T'][0])
# plt.show()

# print(times[1:])

# assert(1==0)

# results_per_pass_RT = results_RT[1]
# prof_front = results_RT[2][0]

# # sum over passes
# results_per_layer_front_RT = np.sum(results_per_pass_RT["a"][0], 0)

# front_surf_rt = planar_surface(interface_layers=front_materials, prof_layers=[3, 4])  # pyramid size in microns

# front_surf_rt_pyr = regular_pyramids(interface_layers=front_materials, prof_layers=[3, 4])  # pyramid size in microns

# back_surf_rt = regular_pyramids(upright=False, interface_layers=back_materials)  # pyramid size in microns

# back_surf_rt_planar = planar_surface(interface_layers=back_materials)  # pyramid size in microns

# rtstr = rt_structure([front_surf_rt, back_surf_rt_planar], [Ge], [bulkthick], Air, Ag, options, use_TMM=True)
# # RT + TMM

# options.n_rays = 4000

# result_RT_only = rtstr.calculate(options)

# rt_front = result_RT_only["interface_profiles"][0]

# plt.figure()
# plt.plot(wavelengths * 1e9, result_RT_only["R"], label="RT")
# plt.plot(wavelengths * 1e9, results_RT[0]["R"][0], label="RT + redist")
# plt.plot(wavelengths * 1e9, result_RT_only["T"], "--", label="RT")
# plt.plot(wavelengths * 1e9, results_RT[0]["T"][0], "--", label="RT + redist")
# plt.plot(wavelengths * 1e9, result_RT_only["A_per_interface"][0])
# plt.plot(wavelengths * 1e9, results_per_layer_front_RT, "--")
# plt.plot(wavelengths * 1e9, result_RT_only["A_per_layer"][:, 0])
# plt.plot(wavelengths * 1e9, results_RT[0]["A_bulk"][0], "--")

# plt.legend()
# plt.show()

# plt.figure()
# plt.semilogy(rt_front.T, alpha=0.5)
# plt.semilogy(prof_front.T, "--")
# plt.ylim(1e-13, 0.1)
# # plt.legend([str(x) for x in range(10)])
# plt.show()

# plt.figure()
# plt.semilogy(results_RT[3][0][40:50].T / 1e9, alpha=0.5)
# plt.semilogy(result_RT_only["profile"][40:50].T, "--")
# plt.ylim(1e-17, 0.1)
# plt.show()
