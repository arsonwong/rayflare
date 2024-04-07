# Copyright (C) 2021-2024 Phoebe Pearce
#
# This file is part of RayFlare and is released under the GNU Lesser General Public License (LGPL), version 3.
# Please see the LICENSE.txt file included as part of this package.
#
# Contact: p.pearce@unsw.edu.au

import numpy as np
from solcore.state import State

from rayflare.transfer_matrix_method.lookup_table import make_TMM_lookuptable
from rayflare.structure import Interface, RTgroup, BulkLayer
from rayflare.ray_tracing import RT
from rayflare.rigorous_coupled_wave_analysis import RCWA
from rayflare.transfer_matrix_method import TMM
from rayflare.angles import make_angle_vector
from rayflare.matrix_formalism.ideal_cases import lambertian_matrix, mirror_matrix
from rayflare.utilities import get_savepath, get_wavelength
from rayflare import logger
from sparse import COO, stack

def make_D(alphas, thick, thetas):
    """
    Makes the bulk absorption vector for the bulk material.

    :param alphas: absorption coefficient (m^{-1})
    :param thick: thickness of the slab in m
    :param thetas: incident thetas in angle_vector (second column)

    :return:
    """
    diag = np.exp(-alphas[:, None] * thick / abs(np.cos(thetas[None, :])))
    D_1 = stack([COO.from_numpy(np.diag(x)) for x in diag])
    return D_1

def process_structure(SC, options, save_location="default", overwrite=False):
    """
    Function which takes a list of Interface and BulkLayer objects, and user options, and carries out the
    necessary calculations to populate the redistribution matrices.

    :param SC: list of Interface and BulkLayer objects. Order is [Interface, BulkLayer, Interface]
    :param options: a dictionary or State object listing the user options
    :param save_location: string - location where the calculated redistribution matrices should be stored.
          Currently recognized are:

              - 'default', which stores the results in folder in your home directory called 'RayFlare_results'
              - 'current', which stores the results in the current working directory
              - or you can specify the full path location for wherever you want the results to be stored.

              In each case, the results will be stored in a subfolder with the name of the project (options.project_name)
    :param overwrite: boolean - if True, will overwrite any existing results in the save_location. If False, will re-use
            any existing results (based on the project name, save_location and names of the surfaces) if they are available.
    """

    if isinstance(options, dict):
        options = State(options)

    def determine_only_incidence(sd, j1, oia):
        if sd == "front" and j1 == 0 and oia:
            only_inc = True
        else:
            only_inc = False

        return only_inc

    def determine_coherency(strt):

        coh = strt.coherent

        if not strt.coherent:
            c_list = strt.coherency_list
        else:
            c_list = None

        return coh, c_list

    layer_widths = []

    structpath = get_savepath(save_location, options["project_name"])

    for i1, struct in enumerate(SC):
        if isinstance(struct, BulkLayer):
            layer_widths.append(struct.width * 1e9)  # convert m to nm
        if isinstance(struct, Interface):
            layer_widths.append(
                (np.array(struct.widths) * 1e9).tolist()
            )  # convert m to nm

    SC.TMM_lookup_table = []
    for i1, struct in enumerate(SC):
        if isinstance(struct, Interface):
            # Check: is this an interface type which requires a lookup table?

            if struct.method == "RT_TMM" or struct.method == "RT_analytical_TMM":
                logger.info(f"Making RT/TMM lookuptable for element {i1} in structure")
                if i1 == 0:  # top interface
                    incidence = SC.incidence
                else:  # not top interface
                    incidence = SC[i1 - 1].material  # bulk material above

                if i1 == (len(SC) - 1):  # bottom interface
                    substrate = SC.transmission
                else:  # not bottom interface
                    substrate = SC[i1 + 1].material  # bulk material below

                coherent, coherency_list = determine_coherency(struct)

                prof_layers = struct.prof_layers

                # takes 0.098619s
                # takes 0.04754 without including unpol and not saving results
                SC.TMM_lookup_table.append(make_TMM_lookuptable(
                    struct.layers,
                    incidence,
                    substrate,
                    struct.name,
                    options,
                    structpath,
                    coherent,
                    coherency_list,
                    prof_layers,
                    [1, -1],
                    overwrite,
                    include_unpol=False,
                    save = False,
                    width_differentials = struct.width_differentials
                ))

        if len(SC.TMM_lookup_table) < i1+1:
            SC.TMM_lookup_table.append(None)

    stored_front_redistribution_matrices = []
    stored_rear_redistribution_matrices = []

    theta_spacing = options.theta_spacing if "theta_spacing" in options else "sin"

    theta_intv, phi_intv, angle_vector = make_angle_vector(
        options["n_theta_bins"],
        options["phi_symmetry"],
        options["c_azimuth"],
        theta_spacing,
    )
    
    for i1, struct in enumerate(SC):
        if isinstance(struct, BulkLayer):
            SC.bulkIndices.append(i1)
            get_wavelength(options)
            n_a_in = int(len(angle_vector) / 2)
            thetas = angle_vector[:n_a_in, 1]
            stored_front_redistribution_matrices.append(make_D(struct.material.alpha(options["wavelength"]), struct.width, thetas))

        if isinstance(struct, Interface):
            SC.interfaceIndices.append(i1)

            if i1 == 0:
                incidence = SC.incidence
            else:
                incidence = SC[i1 - 1].material  # bulk material above

            if i1 == (len(SC) - 1):
                substrate = SC.transmission
                which_sides = ["front"]
            else:
                substrate = SC[i1 + 1].material  # bulk material below
                which_sides = ["front", "rear"]

            if struct.method == "Mirror":
                mirror_matrix(
                    angle_vector,
                    theta_intv,
                    phi_intv,
                    struct.name,
                    options,
                    structpath,
                    front_or_rear="front",
                    save=True,
                    overwrite=overwrite,
                )

            if struct.method == "Lambertian":

                # assuming this is a Lambertian reflector right now
                lambertian_matrix(
                    angle_vector,
                    theta_intv,
                    struct.name,
                    structpath,
                    "front",
                    save=True,
                    overwrite=overwrite,
                )

            if struct.method == "TMM":
                logger.info(f"Making matrix for planar surface using TMM for element {i1} in structure")

                coherent, coherency_list = determine_coherency(struct)

                prof_layers = struct.prof_layers

                for side in which_sides:
                    # only_incidence_angle = determine_only_incidence(side, i1, options['only_incidence_angle'])
                    allArrays, absArrays = TMM(
                        struct.layers,
                        incidence,
                        substrate,
                        struct.name,
                        options,
                        structpath,
                        coherent=coherent,
                        coherency_list=coherency_list,
                        prof_layers=prof_layers,
                        front_or_rear=side,
                        save=False,
                        overwrite=overwrite,
                        width_differentials = struct.width_differentials
                    )
                    if side=="front":
                        stored_front_redistribution_matrices.append([allArrays,absArrays])
                    else:
                        stored_rear_redistribution_matrices.append([allArrays,absArrays])

            if struct.method == "RT_TMM" or struct.method == "RT_analytical_TMM":
                logger.info(f"Ray tracing with TMM lookup table for element {i1} in structure")

                analytical_approx = False
                if struct.method == "RT_analytical_TMM":
                    analytical_approx = True

                prof = struct.prof_layers
                n_abs_layers = len(struct.layers)

                group = RTgroup(textures=[struct.texture])
                for side in which_sides:
                    only_incidence_angle = determine_only_incidence(
                        side, i1, options["only_incidence_angle"]
                    )

                    # takes 0.374021s with save, 0.18945 without save
                    allArrays, absArrays = RT(
                        group,
                        incidence,
                        substrate,
                        struct.name,
                        options,
                        structpath,
                        1,
                        side,
                        n_abs_layers,
                        prof,
                        only_incidence_angle,
                        layer_widths[i1],
                        save=False,
                        overwrite=overwrite,
                        analytical_approx = analytical_approx,
                        lookuptable=SC.TMM_lookup_table[i1],
                        width_differentials = struct.width_differentials
                    )
                    if side=="front":
                        stored_front_redistribution_matrices.append([allArrays,absArrays])
                    else:
                        stored_rear_redistribution_matrices.append([allArrays,absArrays])

            if struct.method == "RT_Fresnel" or struct.method == "RT_analytical_Fresnel":
                logger.info(f"Ray tracing with Fresnel equations for element {i1} in structure")

                analytical_approx = False
                if struct.method == "RT_analytical_TMM":
                    analytical_approx = True
                group = RTgroup(textures=[struct.texture])
                for side in which_sides:
                    only_incidence_angle = determine_only_incidence(
                        side, i1, options["only_incidence_angle"]
                    )

                    RT(
                        group,
                        incidence,
                        substrate,
                        struct.name,
                        options,
                        structpath,
                        0,
                        side,
                        0,
                        None,
                        only_incidence_angle=only_incidence_angle,
                        save=True,
                        overwrite=overwrite,
                        analytical_approx = analytical_approx
                    )

            if struct.method == "RCWA":
                logger.info(f"RCWA calculation for element {i1} in structure")

                prof = struct.prof_layers

                for side in which_sides:
                    # only_incidence_angle = determine_only_incidence(side, i1, options['only_incidence_angle'])

                    RCWA(
                        struct.layers,
                        struct.d_vectors,
                        struct.rcwa_orders,
                        options,
                        structpath,
                        incidence,
                        substrate,
                        only_incidence_angle=False,
                        prof_layers=prof,
                        front_or_rear=side,
                        surf_name=struct.name,
                        save=True,
                        overwrite=overwrite,
                    )
        if len(stored_front_redistribution_matrices) < i1+1:
            stored_front_redistribution_matrices.append(None)
        if len(stored_rear_redistribution_matrices) < i1+1:
            stored_rear_redistribution_matrices.append(None)

    SC.stored_redistribution_matrices = [stored_front_redistribution_matrices, stored_rear_redistribution_matrices]
