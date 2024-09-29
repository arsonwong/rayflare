import time
import numpy as np
import sys
import os
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# sys.path.insert(1,r"D:\Wavelabs\2023-12-24 mockup of PLQE fit\solcore5_20240324")
sys.path.insert(1,r"C:\Users\arson\Documents\solcore5_fork")

from solcore.structure import Layer
from solcore import material

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
    mat.n_path = n_file_path
    if k_file_path is not None:
        mat.k_path = k_file_path
        mat.load_n_data()
        mat.load_k_data()
    else:
        mat.load_nk_data()
    return mat

def bulk_profile(bulk_absorbed_front, bulk_absorbed_rear, z_front):
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
    return absorption_profile_front, absorption_profile_rear, z_front_widths

def layer_profile(Aprof_front, Aprof_rear, front_local_angles, rear_local_angles, overall_A, z_front):
    part1 = Aprof_front[:,:,0,None]*np.exp(Aprof_front[:,:,4,None]*z_front)
    part2 = Aprof_front[:,:,1,None]*np.exp(-Aprof_front[:,:,4,None]*z_front)
    part3 = (Aprof_front[:,:,2,None] + 1j * Aprof_front[:,:,3,None])*np.exp(1j * Aprof_front[:,:,5,None]*z_front)
    part4 = (Aprof_front[:,:,2,None] - 1j * Aprof_front[:,:,3,None])*np.exp(-1j * Aprof_front[:,:,5,None]*z_front)
    result = np.real(part1 + part2 + part3 + part4)
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
    result = np.real(part1 + part2 + part3 + part4)
    absorption_profile_rear = rear_local_angles[:,:,None]*result
    absorption_profile_rear = np.sum(absorption_profile_rear,axis=1)

    absorption_profile_integral = np.sum((absorption_profile_front+absorption_profile_rear)*z_front_widths[None, :], axis=1)
    absorption_profile_front *= overall_A[:,None]/absorption_profile_integral[:,None]
    absorption_profile_rear *= overall_A[:,None]/absorption_profile_integral[:,None]

    return absorption_profile_front, absorption_profile_rear, z_front_widths

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
            time.sleep(5)  # Sleep briefly before trying again
            continue
        print(f"New line: {line.strip()}")
        try:
            exec(line.strip())
        except Exception as e:
            # This block will catch any exception and print the error message
            print(f"An error occurred: {e}")
            break
        # Write the new line to the output file
        output_file.write(line)
        output_file.flush()  # Ensure the line is written to the file immediately
