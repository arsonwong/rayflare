import time
import numpy as np
import os
import sys
import pandas as pd
from copy import deepcopy
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

SC = None
output_file = None
wavelengths = np.arange(300,1201,5) * 1e-9
silicon_bulk_index = 0
active_interface = []
options = default_options()
options.wavelength = wavelengths
options.only_incidence_angle = False
# options.lookuptable_angles = 200
# options.parallel = True
options.project_name = "perovskite_Si_example"
options.n_rays = 2000
options.n_theta_bins = 30 #90
options.c_azimuth = 0.25 #1.00
options.nx = 2
options.ny = 2
options.depth_spacing = 1e-9
options.phi_symmetry = np.pi / 2
options.bulk_profile = False
options.detailed = True

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

def create_new_material(name, n_file_path, k_file_path=None):
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
    return mat

Glass = create_new_material('Glass',r'C:\Users\arson\Documents\rayflare_fork\temp\glass.txt')

def create_new_layer(name, thickness, n_file_path, k_file_path=None):
    mat = create_new_material(name, n_file_path, k_file_path)
    layer = Layer(thickness*1e-9, mat)
    return layer

def bulk_profile(results, z_front, out_path):
    global silicon_bulk_index, output_file
    output_file.write("0:Rayflare Server: Calculating profile for substrate\n")
    output_file.flush()  # Ensure the line is written to the file immediately

    which_bulk = silicon_bulk_index
    bulk_absorbed_front = results[0]['bulk_absorbed_front'][which_bulk]
    bulk_absorbed_rear = results[0]['bulk_absorbed_rear'][which_bulk]
    alphas = results[0]['alphas'][which_bulk]
    abscos = results[0]['abscos']

    z_front_widths = 0.5*(z_front[2:]-z_front[:-2])
    z_front_widths = np.insert(z_front_widths, 0, 0.5*(z_front[1]-z_front[0]))
    z_front_widths = np.append(z_front_widths, 0.5*(z_front[-1]-z_front[-2]))
    z_front_widths *= 100 #convert to cm
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

    absorption_profile = absorption_profile_front + absorption_profile_rear

    if out_path is not None:
        np.savetxt(out_path, absorption_profile, delimiter=",", fmt="%e")

    return absorption_profile_front, absorption_profile_rear, z_front_widths

def layer_profile(results, z_front, which_interface, which_layer, out_path):
    global active_interface, output_file, SC

    output_file.write("0:Rayflare Server: Calculating profile for layer " + str(which_layer+1) + "\n")
    output_file.flush()  # Ensure the line is written to the file immediately

    which_stack = active_interface[which_interface]
    if which_stack < 0:
        return
    
    interface_count = 0
    for i1, struct in enumerate(SC):        
        if isinstance(struct, Interface):
            if which_stack==interface_count:
                Aprof = SC.TMM_lookup_table[i1]['Aprof']
                Aprof = 0.5*(Aprof.loc[dict(pol='s')]+Aprof.loc[dict(pol='p')]).values
            interface_count += 1

    results_per_pass = results[0]['results_per_pass']
    results_pero = np.sum(results_per_pass["a"][which_stack], 0)[:, [which_layer]]
    overall_A = results_pero[:,0] # just flatten

    Aprof_front = Aprof[which_layer][0] # layer1,side1
    Aprof_rear = Aprof[which_layer][1] # backside 
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
    z_front_widths *= 1e-7 # conver to cm

    z_rear = z_front[-1]-z_front
    part1 = Aprof_rear[:,:,0,None]*np.exp(Aprof_rear[:,:,4,None]*z_rear)
    part2 = Aprof_rear[:,:,1,None]*np.exp(-Aprof_rear[:,:,4,None]*z_rear)
    # part3 = (Aprof_rear[:,:,2,None] + 1j * Aprof_rear[:,:,3,None])*np.exp(1j * Aprof_rear[:,:,5,None]*z_rear)
    # part4 = (Aprof_rear[:,:,2,None] - 1j * Aprof_rear[:,:,3,None])*np.exp(-1j * Aprof_rear[:,:,5,None]*z_rear)
    result = np.real(part1 + part2 + 0*part3 + 0*part4)
    absorption_profile_rear = rear_local_angles[:,:,None]*result
    absorption_profile_rear = np.sum(absorption_profile_rear,axis=1)

    absorption_profile_integral = np.sum((absorption_profile_front+absorption_profile_rear)*z_front_widths[None, :], axis=1)
    absorption_profile_front *= overall_A[:,None]/absorption_profile_integral[:,None]
    absorption_profile_rear *= overall_A[:,None]/absorption_profile_integral[:,None]
    absorption_profile_front = np.nan_to_num(absorption_profile_front, nan=0)
    absorption_profile_rear = np.nan_to_num(absorption_profile_rear, nan=0)
    absorption_profile = absorption_profile_front + absorption_profile_rear

    if out_path is not None:
        np.savetxt(out_path, absorption_profile, delimiter=",", fmt="%e")

    return absorption_profile_front, absorption_profile_rear, z_front_widths

def run_simulation(top_medium, bottom_medium, front_materials, front_roughness, back_materials, rear_roughness, surf, surf_back, 
                   cell_bulk, active_layer_indices1, active_layer_indices2, active_layer_indices3, active_layer_indices4, active_layer_indices5, active_layer_indices6, 
                   top_cover_bulk, top_cover_front_materials, top_cover_rear_materials, 
                   bottom_cover_bulk, bottom_cover_front_materials, bottom_cover_rear_materials, 
                   bottom_cover_front_last_layer, bottom_cover_front_last_layer_R, bottom_cover_rear_last_layer, bottom_cover_rear_last_layer_R, 
                   enable_front_incidence, front_angular_distribution, enable_rear_incidence, rear_angular_distribution,
                   front_out_path=None, rear_out_path=None, reconstruct_SC=True):
    t1 = time.time()
    global output_file, options, Glass, active_interface, silicon_bulk_index, SC
    options['output_file'] = output_file

    active_layer_indices = [active_layer_indices1,active_layer_indices2,active_layer_indices3,active_layer_indices4,active_layer_indices5,active_layer_indices6]

    if reconstruct_SC:
        output_file.write("0:Rayflare Server: Setting up the layers\n")
        output_file.flush()  # Ensure the line is written to the file immediately

        top_cover_front_surf = Interface(
            "TMM",
            texture=planar_surface(),
            layers=top_cover_front_materials,
            name="glass",
            coherent=True,
            prof_layers=active_layer_indices[0]
        )

        if len(top_cover_rear_materials)>0:
            top_cover_rear_surf = Interface(
                "TMM",
                texture=planar_surface(),
                layers=top_cover_rear_materials,
                name="glass",
                coherent=True,
                prof_layers=active_layer_indices[1]
            )
            top_cover_spacer = BulkLayer(0.0, top_cover_rear_materials[-1].material, name="spacer")

        if len(bottom_cover_front_materials)>0:
            bottom_cover_front_surf = Interface(
                "TMM",
                texture=planar_surface(),
                layers=bottom_cover_front_materials,
                name="glass",
                coherent=True,
                prof_layers=active_layer_indices[4]
            )
            bottom_cover_spacer = BulkLayer(0.0, bottom_cover_front_materials[0].material, name="spacer")

        bottom_cover_rear_surf = Interface(
            "TMM",
            texture=planar_surface(),
            layers=bottom_cover_rear_materials,
            name="glass",
            coherent=True,
            prof_layers=active_layer_indices[5]
        )

        method = "RT_analytical_TMM"
        if surf[0].N.shape[0]==2: #planar
            method = "TMM"
        front_surf = Interface(method,texture=surf,layers=front_materials,name="Perovskite_aSi_widthcorr",coherent=True,prof_layers=active_layer_indices[2]) 
        method = "RT_analytical_TMM"
        if surf_back[0].N.shape[0]==2: #planar
            method = "TMM"
        back_surf = Interface(method, texture=surf_back, layers=back_materials, name="aSi_ITO_2", coherent=True,prof_layers=active_layer_indices[3])

        silicon_bulk_index = 0
        active_interface = [-1,-1,-1,-1,-1,-1]
        list_ = []
        interface_index = 0
        if top_cover_bulk is not None:
            silicon_bulk_index = 1
            list_.append(top_cover_front_surf)
            active_interface[0] = interface_index
            interface_index += 1
            list_.append(top_cover_bulk)
            if len(top_cover_rear_materials)>0:
                silicon_bulk_index += 1
                list_.append(top_cover_rear_surf)
                active_interface[1] = interface_index
                interface_index += 1
                list_.append(top_cover_spacer)
            
        list_.append(front_surf)
        active_interface[2] = interface_index
        interface_index += 1
        if front_roughness is not None:
            list_.append(front_roughness)
        list_.append(cell_bulk)
        if rear_roughness is not None:
            list_.append(rear_roughness)
        # if bottom_cover_front_last_layer > 0 or bottom_cover_rear_last_layer > 0:
        #     if bottom_cover_front_last_layer==1 or (bottom_cover_front_last_layer==0 and bottom_cover_rear_last_layer==1):
        #         reflector = Interface("Mirror", texture = planar_surface(), layers=[], name="mirror", coherent=True)
        #     else:
        #         reflector = Interface("Lambertian", texture = planar_surface(), layers=[], name="mirror", coherent=True)
        if False: #len(back_materials)==0 and len(bottom_cover_front_materials)==0 and bottom_cover_front_last_layer > 0:
            pass
        else:
            list_.append(back_surf)
            active_interface[3] = interface_index
            interface_index += 1

            if bottom_cover_bulk is not None:
                if len(bottom_cover_front_materials)>0:
                    list_.append(bottom_cover_spacer)
                    list_.append(bottom_cover_front_surf)
                    active_interface[4] = interface_index
                    interface_index += 1
                list_.append(bottom_cover_bulk)
                list_.append(bottom_cover_rear_surf)
                active_interface[5] = interface_index
                interface_index += 1

        SC = Structure(list_, incidence=top_medium, transmission=bottom_medium)

        output_file.write("0:Rayflare Server: Processing the structure\n")
        output_file.flush()  # Ensure the line is written to the file immediately

        process_structure(SC, options, overwrite=True)

    enable_ = [enable_front_incidence, enable_rear_incidence]
    side_ = [1, -1]
    front_results = []
    rear_results = []
    for i12 in range(2):
        if enable_[i12]==1:
            options["incident_side"] = side_[i12]
            if i12==0:
                options["incidence_angular_distribution"] = front_angular_distribution
                options["message"] = "0:Rayflare Server: Simulating front incidence"
                output_file.write(options["message"] + "\n")
            else:
                options["incidence_angular_distribution"] = rear_angular_distribution
                options["message"] = "0:Rayflare Server: Simulating rear incidence"
                output_file.write(options["message"] + "\n")
            
            output_file.flush()  # Ensure the line is written to the file immediately

            results = calculate_RAT(SC, options)
            if i12==0:
                front_results = deepcopy(results)
            else:
                rear_results = deepcopy(results)

            output_file.write("0:Rayflare Server: Post-processing\n")
            output_file.flush()  # Ensure the line is written to the file immediately

            RAT = results[0]['RAT']
            results_per_pass = results[0]['results_per_pass']

            output = [wavelengths*1e9]
            columns = ['Wavelength(nm)','Cover transmittance']
            if i12==0:
                t = results_per_pass["t"][0][0,:,:]
            else:
                t = results_per_pass["r"][-1][0,:,:]
            t = np.sum(t,axis=1)
            output.append(t)

            for i, indices in enumerate(active_layer_indices):
                for index in indices:
                    results_A_ = np.sum(results_per_pass["a"][active_interface[i]], 0)[:, [index-1]]
                    A_ = results_A_[:,0] # just flatten
                    output.append(A_)
                    columns.append('A'+str(i)+'-'+str(index))

            A_Si = RAT["A_bulk"][silicon_bulk_index]
            output.append(A_Si)
            columns.append('A_substrate')

            if i12==0:
                out_path = front_out_path
            else:
                out_path = rear_out_path

            if out_path is not None:
                output = np.array(output).T
                df = pd.DataFrame(output, columns=columns)
                df.to_csv(out_path, index=False)        

    return front_results, rear_results


input_file_path = 'logfile.txt'
output_file_path = 'output_log.txt'

with open(input_file_path, 'w') as file:
    pass  # Just opening the file is enough to erase its contents

with open(output_file_path, 'w') as file:
    pass  # Just opening the file is enough to erase its contents

with open(input_file_path, 'r') as input_file:
    output_file = open(output_file_path, 'a')

    output_file.write("0:Rayflare Server: Setting up the layers\n")
    output_file.flush()  # Ensure the line is written to the file immediately

    # Move to the end of the file
    input_file.seek(0, 2) 
    
    while True:
        line = input_file.readline()
        if not line:
            time.sleep(0.01)  # Sleep briefly before trying again
            continue
        print(line)
        line_split = line.split(":",1)
        line_before_colon = line_split[0]
        line_after_colon = line_split[1]
        print(f"New line: {line.strip()}")
        output_file.write(line_before_colon + ": received\n")
        try:
            exec(line_after_colon)
        except Exception as e:
            # This block will catch any exception and print the error message
            print(f"An error occurred: {e}")
            output_file.write(f"-1: Error: {e}\n")
            break
        output_file.write(line_before_colon + ": executed\n")
        output_file.flush()  # Ensure the line is written to the file immediately
    output_file.close()
