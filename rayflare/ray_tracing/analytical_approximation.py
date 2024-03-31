import numpy as np
import xarray as xr
import os
from rayflare.utilities import get_savepath
from rayflare.angles import fold_phi, make_angle_vector, overall_bin
from sparse import COO, save_npz, stack

# to do
# analytical_front_surface - input is a bunch of rays, in optics it's  called initial_ray
# output is another list, just like in optics
# should change to analytical_scatter
# but now what if ray points up and scatters off underside?
# analytical_scatter must specify whether ray is coming down, or going up
# for an interlayer, there is both reflected rays and transmitted that need to be kept track
# 

# this class describes a ray's / rays' polarization as well as intensity
# Each ray is composed of two perpendicular polarization components, represented by the s_vector and p_vector
# They are called s,p for convenience: if ray is created from an interaction with an interface, then the s_vector 
# is perpendicular to the plane of incidence, and p_vector is parallel to the plane of incidence
# but in general, s_vector and p_vector can be arbitrary directions as long as they are perpendicular to each other and 
# perpendicular to the ray's direction of travel.  
# The lengths of the s_vector, p_vector represent the amplitude of the E field of their respective light
# Therefore (s_vector length)^2 + (p_vector length)^2 = ray intensity
# If s_vector length = p_vector length, the ray is unpolarized
# If one of s_vector, p_vector has zero length, then ray is linearly polarized
# If s_vector, p_vector have nonzero but different length, then ray is neither completely polarized or completely unpolarized
# Elliptical or circular polarization is not represented
# If s_vector, p_vector have shape (3,), then the instance is one ray
# if s_vector, p_vector have shape (N,3), then the instance represents N rays
class Polarization:
    def __init__(self, s_vector, p_vector):
        self.s_vector = s_vector
        self.p_vector = p_vector
    def getIntensity(self):
        if self.s_vector.ndim==1:
            return self.s_vector[0]**2+self.s_vector[1]**2+self.s_vector[2]**2+self.p_vector[0]**2+self.p_vector[1]**2+self.p_vector[2]**2
        return self.s_vector[:,0]**2+self.s_vector[:,1]**2+self.s_vector[:,2]**2+self.p_vector[:,0]**2+self.p_vector[:,1]**2+self.p_vector[:,2]**2

def make_arbitrary_perpendicular_direction(direction):
    index_of_max_abs = np.argmax(np.abs(direction))
    other_index = 0 if index_of_max_abs > 0 else 1
    perpendicular_direction = np.copy(direction)
    perpendicular_direction[other_index] = -direction[index_of_max_abs]
    perpendicular_direction[index_of_max_abs] = direction[other_index]
    return perpendicular_direction

# this class additionally stores a ray's direction, as well as a lot more "metadata"
# like probability (the probability that this ray exists), parent, etc
# It also has propagate_R_or_T which calculates its own polarization s_vector, p_vector
# given its parent ray, its own direction, the plane that its parent scattered off of, 
# and the resultant Rs, Rp or Ts, Tp
class Ray:
    def __init__(self, direction, probability=1.0, parent=None, scatter_angle=None, scatter_plane_normal=None):
        self.direction = direction
        self.probability = probability
        self.scatter_angle = scatter_angle
        self.scatter_plane_normal = scatter_plane_normal
        self.parent = parent
        self.transmitted_child = None
        self.reflected_child = None
        self.polarization = None
        self.propagated = False
        self.memorized_items = []
        if parent is None: # first initial ray, then just make circular polarized
            self.propagated = True
            s_vector = make_arbitrary_perpendicular_direction(direction)
            p_vector = np.cross(direction, s_vector)
            length_ = np.linalg.norm(s_vector)
            s_vector /= (length_*np.sqrt(2.0))
            length_ = np.linalg.norm(p_vector)
            p_vector /= (length_*np.sqrt(2.0))
            self.polarization = Polarization(s_vector, p_vector)
        if parent:
            self.probability = self.probability*parent.probability

    def turn_into_initial_ray(self):
        self.scatter_angle = None
        self.parent = None
        self.scatter_plane_normal=None
        self.propagated = True
        self.num_of_texture_scatter = 0

    def getIntensity(self):
        return self.probability*self.polarization.getIntensity()

    def point_down(self):
        if self.direction[2] > 0:
            self.direction[2] *= -1.0
            self.polarization.s_vector[2] *= -1.0
            self.polarization.p_vector[2] *= -1.0
    
    def propagate_R_or_T(self, R_or_T_s, R_or_T_p):
        assert(self.parent is not None and self.parent.propagated == True)
        self.propagated = True
        parent_s_vector = self.parent.polarization.s_vector
        parent_p_vector = self.parent.polarization.p_vector
        if len(self.memorized_items)==0:
            parent_direction = self.parent.direction
            new_s_direction = np.cross(parent_direction, self.scatter_plane_normal)
            length_ = np.linalg.norm(new_s_direction)
            if length_ == 0: # which could happen if parent_direction, self.reflection_plane_normal are parallel
                # then just make an arbitrary direction which is perpendicular to parent_direction
                new_s_direction = make_arbitrary_perpendicular_direction(parent_direction)
            else:
                new_s_direction /= length_
            new_p_direction = np.cross(self.direction, new_s_direction)
            new_p_direction /= np.linalg.norm(new_p_direction)
            old_p_direction = np.cross(parent_direction, new_s_direction)
            old_p_direction /= np.linalg.norm(old_p_direction)
            self.memorized_items.append(new_s_direction)
            self.memorized_items.append(old_p_direction)
            self.memorized_items.append(new_p_direction)
        else:
            new_s_direction = self.memorized_items[0]
            old_p_direction = self.memorized_items[1]
            new_p_direction = self.memorized_items[2]

        s_component = np.array([np.dot(parent_s_vector, new_s_direction), np.dot(parent_p_vector, new_s_direction)])
        if np.ndim(s_component) == 1:
            s_component = s_component.reshape(1,-1)
        else:
            s_component = s_component.T
        s_component = np.sqrt(s_component[:,0]**2+s_component[:,1]**2) 
        s_component_after_scatter = s_component*np.sqrt(R_or_T_s) # R_s can be a vector of many wavelengths
        p_component = np.array([np.dot(parent_s_vector, old_p_direction), np.dot(parent_p_vector, old_p_direction)])
        if np.ndim(p_component) == 1:
            p_component = p_component.reshape(1,-1)
        else:
            p_component = p_component.T
        p_component = np.sqrt(p_component[:,0]**2+p_component[:,1]**2) 
        p_component_after_scatter = p_component*np.sqrt(R_or_T_p) # R_p can be a vector of many wavelengths
        s_vector = np.outer(s_component_after_scatter, new_s_direction) 
        p_vector = np.outer(p_component_after_scatter, new_p_direction)
        self.polarization = Polarization(s_vector, p_vector)
        return s_component, p_component  # this is for reference, aid in other calculations if needed

def get_ray_directions(ray_queue):
    directions = []
    for i, ray in enumerate(ray_queue):
        if i == 0:
            directions = ray.direction.reshape(1, -1)
        else:
            directions = np.vstack((directions, ray.direction.reshape(1, -1)))
    return directions


theta_lamb = np.linspace(0, 0.999 * np.pi / 2, 100)
def traverse_vectorised(width, theta, alpha, I_i, positions, I_thresh, direction):

    ratio = alpha / np.real(np.abs(np.cos(theta)))
    DA_u = I_i[:, None] * ratio[:, None] * np.exp((-ratio[:, None] * positions[None, :]))
    I_back = I_i * np.exp(-ratio * width)

    stop = np.where(I_back < I_thresh)[0]

    if direction == -1:
        DA_u = np.flip(DA_u)

    intgr = np.trapz(DA_u, positions, axis=1)

    DA = np.divide(
        ((I_i[:, None] - I_back[:, None]) * DA_u).T, intgr,
    ).T

    DA[intgr == 0] = 0

    return DA, stop, I_back

def calc_RAT_Fresnel(theta, pol, *args):
    n1 = args[0]
    n2 = args[1]
    theta_t = np.arcsin((n1 / n2) * np.sin(theta))
    if pol == "s":
        Rs = (
                np.abs(
                    (n1 * np.cos(theta) - n2 * np.cos(theta_t))
                    / (n1 * np.cos(theta) + n2 * np.cos(theta_t))
                )
                ** 2
        )
        return Rs, [0]

    if pol == "p":
        Rp = (
                np.abs(
                    (n1 * np.cos(theta_t) - n2 * np.cos(theta))
                    / (n1 * np.cos(theta_t) + n2 * np.cos(theta))
                )
                ** 2
        )
        return Rp, [0]

    else:
        Rs = (
                np.abs(
                    (n1 * np.cos(theta) - n2 * np.cos(theta_t))
                    / (n1 * np.cos(theta) + n2 * np.cos(theta_t))
                )
                ** 2
        )
        Rp = (
                np.abs(
                    (n1 * np.cos(theta_t) - n2 * np.cos(theta))
                    / (n1 * np.cos(theta_t) + n2 * np.cos(theta))
                )
                ** 2
        )
        return (Rs + Rp) / 2, np.array([0])

def calc_RAT_Fresnel_vec(theta, pol, *args):

    n1 = args[0]
    n2 = args[1]
    ratio = np.clip((n1[None, :] / n2[None, :]) * np.sin(theta[:, None]), -1, 1)
    theta_t = np.arcsin(ratio)

    if pol == "s":
        Rs = (
                np.abs(
                    (n1[None, :] * np.cos(theta[:,None]) - n2[None, :] * np.cos(theta_t))
                    / (n1[None, :] * np.cos(theta[:, None]) + n2[None, :] * np.cos(theta_t))
                )
                ** 2
        )

        # Rs[np.isnan(Rs)] = 1

        return Rs, np.array([0]), 1-Rs

    if pol == "p":
        Rp = (
                np.abs(
                    (n1[None, :] * np.cos(theta_t) - n2[None, :] * np.cos(theta[:,None]))
                    / (n1[None, :] * np.cos(theta_t) + n2[None, :] * np.cos(theta[:,None]))
                )
                ** 2
        )

        # Rp[np.isnan(Rp)] = 1

        return Rp, np.array([0]), 1-Rp

    else:
        Rs = (
                np.abs(
                    (n1[None, :] * np.cos(theta[:,None]) - n2[None, :] * np.cos(theta_t))
                    / (n1[None, :] * np.cos(theta[:,None]) + n2[None, :] * np.cos(theta_t))
                )
                ** 2
        )
        Rp = (
                np.abs(
                    (n1[None, :] * np.cos(theta_t) - n2[None, :] * np.cos(theta[:,None]))
                    / (n1[None, :] * np.cos(theta_t) + n2[None, :] * np.cos(theta[:,None]))
                )
                ** 2
        )
        # Rs[np.isnan(Rs)] = 1
        # Rp[np.isnan(Rp)] = 1

        return (Rs + Rp) / 2, np.array([0])

def calc_RAT_TMM(theta, pol, *args):
    lookuptable = args[0]
    side = args[1]

    data = lookuptable.loc[dict(side=side, pol=pol)].sel(
        angle=abs(theta), method="nearest"
    )

    R = np.real(data["R"].data)
    A_per_layer = np.real(data["Alayer"].data)
    T = 1 - R - np.sum(A_per_layer, axis=1)

    return R, A_per_layer, T



def RT_analytical(
    wl,
    theta_in,
    phi_in,
    n0, 
    n1, 
    max_interactions, 
    surface,
    phi_sym,
    theta_intv,
    phi_intv,
    angle_vector,
    Fr_or_TMM,
    n_abs_layers,
    lookuptable,
    side
):

    how_many_faces = len(surface.N)
    normals = surface.N

    opposite_faces = np.where(np.dot(normals, normals.T) < 0)[1]

    if len(opposite_faces) == 0:
        max_interactions =  1

    if Fr_or_TMM == 0:
        calc_RAT = calc_RAT_Fresnel
        R_args = [n0, n1]
    else:
        calc_RAT = calc_RAT_TMM
        R_args = [lookuptable, 1]

    area = np.sqrt(
    np.sum(np.cross(surface.P_0s - surface.P_1s, surface.P_2s - surface.P_1s, axis=1) ** 2, 1)
    ) / 2

    relevant_face = np.arange(how_many_faces)

    # define the incident ray
    ray_queue = [Ray(direction = np.array([np.sin(theta_in)*np.cos(phi_in), np.sin(theta_in)*np.sin(phi_in), -np.cos(theta_in)*side]))]

    # do the analytical ray tracing here
    scattered_rays = []
    A_mat = np.zeros((len(wl), n_abs_layers))
    for _ in range(max_interactions):
        num_of_rays = len(ray_queue)
        if num_of_rays==0:
            break
        ray_directions = get_ray_directions(ray_queue)
        cos_inc = -np.dot(ray_directions, normals[relevant_face].T) # dot product, resulting in shape (num of rays, num of faces)
        hit_prob = cos_inc * area[relevant_face] # scale by area of each triangle, still shape (num of rays, num of faces)
        hit_prob[cos_inc < 0] = 0  # if negative, then the ray is shaded from that pyramid face and will never hit it
        hit_prob = hit_prob / np.sum(hit_prob, axis=1)[:, None]
        scatter_angles = np.arccos(cos_inc)

        for i in range(how_many_faces):
            normal_component = np.dot(cos_inc[:,i][:,None],normals[i][None,:])
            reflected_directions = ray_directions - 2 * normal_component
            reflected_directions = reflected_directions / np.linalg.norm(reflected_directions, axis=1)[:, None]

            tr_par = (n0 / n1) * (ray_directions - normal_component)
            tr_par_length = np.linalg.norm(tr_par,axis=1)
            tr_perp = -np.sqrt(1 - tr_par_length ** 2)[:, None] * normals[relevant_face]

            refracted_directions = np.real(tr_par + tr_perp)
            refracted_directions  = refracted_directions / np.linalg.norm(refracted_directions, axis=1)[:,None]

            for j in range(num_of_rays):
                if hit_prob[j,i] > 0:
                    # reflected and transmitted rays
                    Rs, As_per_layer, Ts = calc_RAT(np.arccos(cos_inc), 's', *R_args)
                    Rp, Ap_per_layer, Tp = calc_RAT(np.arccos(cos_inc), 'p', *R_args)
                    reflected_ray = Ray(direction = reflected_directions[j], probability = hit_prob[j][i], parent=ray_queue[j], 
                                         scatter_angle=scatter_angles[j,i], scatter_plane_normal=normals[i])
                    s_component, p_component = reflected_ray.propagate_R_or_T(Rs, Rp)
                    A_mat += ray_queue[j].getIntensity()*(s_component[:,None]*As_per_layer + p_component[:,None]*Ap_per_layer)
                    ray_queue[j].reflected_child = reflected_ray
                    if np.sign(reflected_directions[j][2])*side==1:
                        scattered_rays.append(reflected_ray)
                    else:
                        ray_queue.append(reflected_ray)
                    if tr_par_length[j] < 1: # not total internally reflected
                        transmitted_ray = Ray(direction = refracted_directions[j], probability = hit_prob[j][i], parent=ray_queue[j], 
                                         scatter_angle=scatter_angles[j,i], scatter_plane_normal=normals[i])
                        transmitted_ray.propagate_R_or_T(Ts, Tp)
                        ray_queue[j].transmitted_child = transmitted_ray
                        scattered_rays.append(transmitted_ray)
            
        ray_queue = ray_queue[num_of_rays:]

    # now, compile the results
    scattered_ray_directions = get_ray_directions(scattered_rays)
    horizontal_comp = np.sqrt(scattered_ray_directions[:,0]**2+scattered_ray_directions[:,1]**2)
    horizontal_comp[horizontal_comp==0] = 1.0 # just to avoid division by zero later
    thetas_out = np.arccos(scattered_ray_directions[:,2])
    phis_out = np.arccos(scattered_ray_directions[:,1]/horizontal_comp)

    phis_out = fold_phi(phis_out, phi_sym)
    phi_in = fold_phi(phi_in, phi_sym)

    # theta_local_incidence = np.zeros((n_angles, nx * ny))

    if side == -1:
        thetas_out = np.pi - thetas_out

    theta_local_incidence = np.abs(theta_local_incidence)
    n_thetas = len(theta_intv) - 1

    if Fr_or_TMM > 0:
        # now we need to make bins for the absorption
        theta_intv = np.append(theta_intv, 11)
        phi_intv = phi_intv + [np.array([0])]

    binned_theta_out = np.digitize(thetas_out, theta_intv, right=True) - 1
    phis_out = xr.DataArray(
        phis_out,
        coords={"theta_bin": (["angle_in", "position"], binned_theta_out)},
        dims=["angle_in", "position"],
    )
    bin_out = (
        phis_out.groupby("theta_bin")
        .map(overall_bin, args=(phi_intv, angle_vector[:, 0]))
        .data
    )

    out_mat = np.zeros((len(wl), len(angle_vector))) 
    for l1 in range(len(thetas_out)):
        out_mat[:,bin_out[l1]] += scattered_rays[l1].getIntensity()

    out_mat = COO.from_numpy(out_mat)  # sparse matrix
    A_mat = COO.from_numpy(A_mat)

    return out_mat, A_mat
    

# 2024-03-26: Johnson's implementation that uses the Ray class
# direction = upwards (-1) or downwards (1)
# if upwards, then make the surface normals all inverted
def analytical_scatter(surface, surf_index, is_texture_scatter, rays_in, n0, n1, max_interactions, Fr_or_TMM=0, lookuptable=None, direction=-1):
    ray_queue = rays_in
    reflected_rays = []
    transmitted_rays = []

    how_many_faces = len(surface.N)
    normals = surface.N
    # if rays going upwards towards bottom side of surface, then make the surface normals all inverted 
    if direction==-1:
        normals = -1*normals

    # todo: do something about this
    opposite_faces = np.where(np.dot(normals, normals.T) < 0)[1]

    area = np.sqrt(
        np.sum(np.cross(surface.P_0s - surface.P_1s, surface.P_2s - surface.P_1s, axis=1) ** 2, 1)
        ) / 2

    relevant_face = np.arange(how_many_faces)

    # first do ray tracing, without calculating the intensity of reflectance/transmittance
    for _ in range(max_interactions):
        num_of_rays = len(ray_queue)
        if num_of_rays==0:
            break
        ray_directions = get_ray_directions(ray_queue)
        cos_inc = -np.dot(ray_directions, normals[relevant_face].T) # dot product, resulting in shape (num of rays, num of faces)
        hit_prob = cos_inc * area[relevant_face] # scale by area of each triangle, still shape (num of rays, num of faces)
        hit_prob[cos_inc < 0] = 0  # if negative, then the ray is shaded from that pyramid face and will never hit it
        hit_prob = hit_prob / np.sum(hit_prob, axis=1)[:, None]
        scatter_angles = np.arccos(cos_inc)

        for i in range(how_many_faces):
            normal_component = np.dot(cos_inc[:,i][:,None],normals[i][None,:])
            reflected_directions = ray_directions - 2 * normal_component
            reflected_directions = reflected_directions / np.linalg.norm(reflected_directions, axis=1)[:, None]

            tr_par = (n0 / n1) * (ray_directions - normal_component)
            tr_par_length = np.linalg.norm(tr_par,axis=1)
            tr_perp = -np.sqrt(1 - tr_par_length ** 2)[:, None] * normals[relevant_face]

            refracted_directions = np.real(tr_par + tr_perp)
            refracted_directions  = refracted_directions / np.linalg.norm(refracted_directions, axis=1)[:,None]

            for j in range(num_of_rays):
                if hit_prob[j,i] > 0:
                    # reflected and transmitted rays
                    reflected_ray = Ray(direction = reflected_directions[j], probability = hit_prob[j][i], parent=ray_queue[j], 
                                         scatter_angle=scatter_angles[j,i], scatter_plane_normal=normals[i], is_texture_scatter=is_texture_scatter, spawned_surf_index=surf_index, up_or_down=direction*-1)
                    ray_queue[j].reflected_child = reflected_ray
                    if np.sign(reflected_directions[j][2])*direction==1:
                        reflected_rays.append(reflected_ray)
                    else:
                        ray_queue.append(reflected_ray)
                    if tr_par_length[j] < 1: # not total internally reflected
                        transmitted_ray = Ray(direction = refracted_directions[j], probability = hit_prob[j][i], parent=ray_queue[j], 
                                         scatter_angle=scatter_angles[j,i], scatter_plane_normal=normals[i], is_texture_scatter=is_texture_scatter, spawned_surf_index=surf_index, up_or_down=direction)
                        ray_queue[j].transmitted_child = transmitted_ray
                        transmitted_rays.append(transmitted_ray)
            
        ray_queue = ray_queue[num_of_rays:]
    return reflected_rays, transmitted_rays


def analytical_front_surface(front, r_in, n0, n1, pol, max_interactions, n_layers, direction,
                             n_reps,
                             positions,
                             bulk_width,
                             alpha_bulk,
                             I_thresh,
                             Fr_or_TMM=0,
                             lookuptable=None,
                             ):

    # n0 should be real
    # n1 can be complex

    # TODO:
    # reflectance (not intensity but directions) will always be the same, regardles of wavelength,
    # so this could be calculated once and then used for all wavelengths. Currently, because the
    # analytical calculation divides the rays into categories at the end, the accuracy of the R/A/T value
    # will be limited to 1/n_rays. This will be addressed in a future release.

    how_many_faces = len(front.N)
    normals = front.N
    opposite_faces = np.where(np.dot(normals, normals.T) < 0)[1]

    if len(opposite_faces) == 0:
        max_interactions =  1

    if Fr_or_TMM == 0:
        calc_RAT = calc_RAT_Fresnel
        R_args = [n0, n1]

    else:
        calc_RAT = calc_RAT_TMM
        R_args = [lookuptable, 1]

    r_inc = np.tile(r_in, (how_many_faces, 1))  # (4, 3) array

    area = np.sqrt(
        np.sum(np.cross(front.P_0s - front.P_1s, front.P_2s - front.P_1s, axis=1) ** 2, 1)
        ) / 2

    relevant_face = np.arange(how_many_faces)

    R_per_it = np.zeros((how_many_faces, max_interactions))
    T_per_it = np.zeros((how_many_faces, max_interactions))
    T_dir_per_it = np.zeros((how_many_faces, max_interactions))
    A_per_it = np.zeros((how_many_faces, n_layers, max_interactions))

    stop_it = np.ones(how_many_faces, dtype=int) * max_interactions

    cos_inc = -np.sum(normals[relevant_face] * r_inc, 1)  # dot product

    hit_prob = area[relevant_face] * cos_inc  # scale by area of each triangle
    hit_prob[
        cos_inc < 0] = 0  # if negative, then the ray is shaded from that pyramid face and will never hit it
    hit_prob = hit_prob / np.sum(hit_prob)  # initial probability of hitting each face

    reflected_ray_directions = np.zeros((how_many_faces, 3, max_interactions))
    transmitted_ray_directions = np.zeros((how_many_faces, 3, max_interactions))

    N_interaction = 0

    while N_interaction < max_interactions:

        cos_inc = -np.sum(normals[relevant_face] * r_inc, 1)  # dot product

        reflected_direction = r_inc - 2 * np.sum(r_inc*normals[relevant_face], axis=1)[:,None] * normals[relevant_face]
        reflected_direction = reflected_direction / np.linalg.norm(reflected_direction, axis=1)[:, None]

        reflected_ray_directions[:, :, N_interaction] = reflected_direction

        cos_inc[cos_inc < 0] = 0
        # if negative, then the ray is shaded from that pyramid face and will never hit it

        tr_par = (n0 / n1) * (r_inc - np.sum(r_inc*normals[relevant_face], axis=1)[:,None] * normals[relevant_face])
        tr_perp = -np.sqrt(1 - np.linalg.norm(tr_par,axis=1) ** 2)[:, None] * normals[relevant_face]

        refracted_rays = np.real(tr_par + tr_perp)
        refracted_rays  = refracted_rays / np.linalg.norm(refracted_rays, axis=1)[:,None]
        transmitted_ray_directions[:, :,  N_interaction] = refracted_rays

        R_prob, A_prob = calc_RAT(np.arccos(cos_inc), pol, *R_args)

        if np.sum(A_prob) > 0:
            A_prob_sum = np.sum(A_prob, axis=1)

        else:
            A_prob_sum = 0

        T_per_it[:, N_interaction] = 1 - R_prob - A_prob_sum

        A_per_it[:, :, N_interaction] = A_prob

        T_dir_per_it[:, N_interaction] = np.abs(
            refracted_rays[:, 2] / np.linalg.norm(refracted_rays,
                                                  axis=1))  # cos (global) of refracted ray

        cos_inc[reflected_direction[:, 2] > 0] = 0
        stop_it[
            np.all((reflected_direction[:, 2] > 0, stop_it > N_interaction),
                   axis=0)] = N_interaction
         # want to end for this surface, since rays are travelling upwards -> no intersection

        R_per_it[:,
        N_interaction] = R_prob  # intensity reflected from each face, relative to incident total intensity 1

        # once ray travels upwards once, want to end calculation for that plane; don't want to
        # double count

        if len(opposite_faces) > 0:
            relevant_face = opposite_faces[relevant_face]

        r_inc = reflected_direction

        if np.sum(cos_inc) == 0:
            # no more interactions with any of the faces
            break

        N_interaction += 1

    remaining_intensity = np.insert(np.cumprod(R_per_it, axis=1), 0, np.ones(how_many_faces),
                                    axis=1)[:, :-1]

    R_total = np.array([hit_prob[j1] * np.prod(R_per_it[j1, :stop_it[j1] + 1]) for j1 in
               range(how_many_faces)])
    final_R_directions = np.array([reflected_ray_directions[j1, :, stop_it[j1]] for j1 in
                          range(how_many_faces)])
    # the weight of each of these directions is R_total

    # loop through faces and interactions:
    final_T_directions = []
    final_T_weights = []
    final_T_n_interactions = []

    for j1 in range(how_many_faces):
        for j2 in range(stop_it[j1] + 1):
            final_T_directions.append(transmitted_ray_directions[j1, :, j2])
            final_T_weights.append(hit_prob[j1]*remaining_intensity[j1, j2]*T_per_it[j1, j2])
            final_T_n_interactions.append(j2 + 1)

    final_T_weights = np.array(final_T_weights)
    final_T_weights[final_T_weights < 0] = 0
    final_T_directions = np.array(final_T_directions)

    A_total = hit_prob[:, None] * np.sum(remaining_intensity[:, None, :] * A_per_it, axis=2)

    theta_out_R = np.arccos(final_R_directions[:, 2] / np.linalg.norm(final_R_directions, axis=1))
    phi_out_R = np.arctan2(final_R_directions[:, 1], final_R_directions[:, 0])
    # number of reps of each theta value for the angular distribution:
    n_reps_R = n_reps * R_total

    theta_out_T = np.arccos(final_T_directions[:, 2] / np.linalg.norm(final_T_directions, axis=1))
    phi_out_T = np.arctan2(final_T_directions[:, 1], final_T_directions[:, 0])

    n_reps_T = n_reps * final_T_weights

    n_reps_A_surf = np.sum(A_total) * n_reps

    # now make sure n_reps_R, n_reps_T and n_reps_A_surf add to n_reps, remained is divided fairly:
    n_reps_R_int = np.floor(n_reps_R).astype(int)
    n_reps_T_int = np.floor(n_reps_T).astype(int)
    n_reps_A_surf_int = np.floor(n_reps_A_surf).astype(int)

    n_reps_R_remainder = np.sum(n_reps_R - n_reps_R_int)
    n_reps_T_remainder = np.sum(n_reps_T - n_reps_T_int)
    n_reps_A_surf_remainder = n_reps_A_surf - n_reps_A_surf_int

    rays_to_divide = n_reps - np.sum(n_reps_R_int) - np.sum(n_reps_T_int) - n_reps_A_surf_int

    # add these rays to the ones with the highest remainders:
    extra_rays_R = np.round(n_reps_R_remainder / (
                n_reps_R_remainder + n_reps_T_remainder + n_reps_A_surf_remainder) * rays_to_divide).astype(
        int)
    extra_rays_T = np.round(n_reps_T_remainder / (n_reps_T_remainder + n_reps_A_surf_remainder) * (
                rays_to_divide - extra_rays_R)).astype(int)

    extra_rays_A = rays_to_divide - extra_rays_R - extra_rays_T

    n_reps_R_int[np.argmax(n_reps_R_remainder)] += extra_rays_R
    n_reps_T_int[np.argmax(n_reps_T_remainder)] += extra_rays_T
    n_reps_A_surf_int += extra_rays_A

    # see which of the transmitted rays reach the back of the Si before falling below
    # I_thresh

    DA, stop, I = traverse_vectorised(
        bulk_width,
        theta_out_T,
        alpha_bulk,
        np.ones_like(theta_out_T),
        positions,
        I_thresh,
        direction,
    )

    I_out_actual = final_T_weights*I
    A_bulk_actual = np.sum(final_T_weights - I_out_actual)

    theta_out_T[stop] = np.nan
    phi_out_T[stop] = np.nan

    # make the list of theta_out values

    theta_R_reps = np.concatenate(
        [np.tile(theta_out_R[j], n_reps_R_int[j]) for j in range(how_many_faces)])
    phi_R_reps = np.concatenate(
        [np.tile(phi_out_R[j], n_reps_R_int[j]) for j in range(how_many_faces)])
    n_interactions_R_reps = np.concatenate(
        [np.tile(stop_it[j] + 1, n_reps_R_int[j]) for j in range(how_many_faces)])
    I_R_reps = np.ones_like(theta_R_reps)
    n_passes_R_reps = np.zeros_like(theta_R_reps)

    theta_A_surf_reps = np.ones(n_reps_A_surf_int) * np.nan
    phi_A_surf_reps = np.ones(n_reps_A_surf_int) * np.nan
    n_interactions_A_surf_reps = np.ones(n_reps_A_surf_int)
    I_A_surf_reps = np.zeros_like(theta_A_surf_reps)
    n_passes_A_surf_reps = np.zeros_like(theta_A_surf_reps)

    theta_T_reps = np.concatenate(
        [np.tile(theta_out_T[j], n_reps_T_int[j]) for j in range(len(theta_out_T))])
    phi_T_reps = np.concatenate(
        [np.tile(phi_out_T[j], n_reps_T_int[j]) for j in range(len(phi_out_T))])
    n_interactions_T_reps = np.concatenate(
        [np.tile(final_T_n_interactions[j], n_reps_T_int[j]) for j in
         range(len(final_T_n_interactions))])
    I_T_reps = np.concatenate([np.tile(I[j], n_reps_T_int[j]) for j in range(len(I))])

    n_passes_T_reps = np.ones_like(theta_T_reps)

    theta_out = np.concatenate([theta_R_reps, theta_A_surf_reps, theta_T_reps])
    phi_out = np.concatenate([phi_R_reps, phi_A_surf_reps, phi_T_reps])
    n_interactions = np.concatenate(
        [n_interactions_R_reps, n_interactions_A_surf_reps, n_interactions_T_reps])
    I_out = np.concatenate([I_R_reps, I_A_surf_reps, I_T_reps])

    n_passes = np.concatenate(
        [n_passes_R_reps, n_passes_A_surf_reps, n_passes_T_reps])

    profile = np.sum(final_T_weights[:, None] * DA, axis=0)

    return theta_out, phi_out, I_out, n_interactions, n_passes, A_bulk_actual, profile, np.sum(A_total, axis=0)


def lambertian_scattering(strt, save_location, options):

    structpath = get_savepath(save_location, options.project_name)

    I_theta = np.cos(theta_lamb)
    I_theta = I_theta/np.sum(I_theta)

    phi = np.linspace(0, options.phi_symmetry, 40)

    # make a grid of rays with these thetas and phis

    theta_grid, phi_grid = np.meshgrid(theta_lamb, phi)
    theta_grid = theta_grid.flatten()
    phi_grid = phi_grid.flatten()

    r_a_0 = np.real(
        np.array(
            [np.sin(theta_grid) * np.cos(phi_grid), np.sin(theta_grid) * np.sin(phi_grid),
             np.cos(theta_grid)]
        )
    )

    r_a_0_rear = np.copy(r_a_0)
    r_a_0_rear[2, :] = -r_a_0_rear[2, :]

    result_list = []

    for mat_index in range(1, len(strt.widths) + 1):

        front_inside = strt.textures[mat_index - 1][1]
        rear_inside = strt.textures[mat_index][0]

        n_triangles_front = len(front_inside.P_0s)
        n_triangles_rear = len(rear_inside.P_0s)

        hit_prob_front = np.matmul(front_inside.N, r_a_0)

        theta_local_front = np.arccos(hit_prob_front)

        theta_local_front[theta_local_front > np.pi / 2] = 0

        hit_prob_rear = -np.matmul(rear_inside.N, r_a_0_rear)
        theta_local_rear = np.arccos(hit_prob_rear)

        theta_local_rear[theta_local_rear > np.pi / 2] = 0

        n_front_layers = len(strt.textures[mat_index - 1][0].interface_layers) if hasattr(strt.textures[mat_index - 1][0], 'interface_layers') else 0
        n_rear_layers = len(strt.textures[mat_index][0].interface_layers) if hasattr(strt.textures[mat_index][0], 'interface_layers') else 0

        unique_angles_front, inverse_indices_front = np.unique(theta_local_front, return_inverse=True)
        unique_angles_rear, inverse_indices_rear = np.unique(theta_local_rear, return_inverse=True)

        if n_front_layers > 0:
            lookuptable_front = xr.open_dataset(os.path.join(structpath, front_inside.name + f"int_{mat_index - 1}.nc"))

            data_front = lookuptable_front.loc[dict(side=-1, pol=options.pol)].sel(
                angle=abs(unique_angles_front), wl=options.wavelength * 1e9, method="nearest"
            ).load()
            R_front = np.real(data_front["R"].data).T
            A_per_layer_front = np.real(data_front["Alayer"].data).T
            A_all_front = A_per_layer_front[:, inverse_indices_front].reshape(
                (n_front_layers,) + theta_local_front.shape + (len(options.wavelength),))

            A_reshape_front = A_all_front.reshape(
                (n_front_layers, n_triangles_front, len(phi), len(theta_lamb), len(options.wavelength)))


        else:
            R_front = \
                calc_RAT_Fresnel_vec(unique_angles_front, options.pol,
                                     strt.mats[mat_index].n(options.wavelength),
                                     strt.mats[mat_index - 1].n(options.wavelength))[0]
            A_reshape_front = 0

        R_all_front = R_front[inverse_indices_front].reshape(
            theta_local_front.shape + (len(options.wavelength),))

        if n_rear_layers > 0:
            lookuptable_rear = xr.open_dataset(os.path.join(structpath, rear_inside.name + f"int_{mat_index}.nc"))
            data_rear = lookuptable_rear.loc[dict(side=1, pol=options.pol)].sel(
                angle=abs(unique_angles_rear), wl=options.wavelength * 1e9, method="nearest"
            )
            R_rear = np.real(data_rear["R"].data).T
            A_per_layer_rear = np.real(data_rear["Alayer"].data).T
            A_all_rear = A_per_layer_rear[:, inverse_indices_rear].reshape(
                (n_rear_layers,) + theta_local_rear.shape + (len(options.wavelength),))

            A_reshape_rear = A_all_rear.reshape(
                (n_rear_layers, n_triangles_rear, len(phi), len(theta_lamb), len(options.wavelength)))


        else:
            R_rear = \
            calc_RAT_Fresnel_vec(unique_angles_rear, options.pol, strt.mats[mat_index].n(options.wavelength),
                                 strt.mats[mat_index + 1].n(options.wavelength))[0]
            A_reshape_rear = 0

        R_all_rear = R_rear[inverse_indices_rear].reshape(
            theta_local_rear.shape + (len(options.wavelength),))

        # now populate matrix of local angles based on these probabilities

        # identify allowed angles:

        # surface normals:

        hit_prob_front[hit_prob_front < 0] = 0
        hit_prob_rear[hit_prob_rear < 0] = 0

        # calculate area of each triangle
        area_front = np.sqrt(
            np.sum(np.cross(front_inside.P_0s - front_inside.P_1s, front_inside.P_2s - front_inside.P_1s, axis=1) ** 2, 1)
            ) / 2

        area_front = area_front / max(area_front)

        hit_prob_front = area_front[:, None] * hit_prob_front / np.sum(hit_prob_front, axis=0)

        hit_prob_reshape_front = hit_prob_front.reshape((n_triangles_front, len(phi), len(theta_lamb)))
        # now take the average over all the faces and azimuthal angles
        R_reshape_front = R_all_front.reshape((n_triangles_front, len(phi), len(theta_lamb), len(options.wavelength)))

        R_weighted_front = R_reshape_front * hit_prob_reshape_front[:, :, :, None]
        R_polar_front = np.sum(np.mean(R_weighted_front, 1), 0)

        A_surf_weighted_front = A_reshape_front * hit_prob_reshape_front[None, :, :, :, None]
        A_polar_front = np.sum(np.mean(A_surf_weighted_front, 2), 1)

        area_rear = np.sqrt(
            np.sum(np.cross(rear_inside.P_0s - rear_inside.P_1s, rear_inside.P_2s - rear_inside.P_1s, axis=1) ** 2, 1)
            ) / 2

        area_rear = area_rear / max(area_rear)

        hit_prob_rear = area_rear[:, None] * hit_prob_rear / np.sum(hit_prob_rear, axis=0)

        hit_prob_reshape_rear = hit_prob_rear.reshape((n_triangles_rear, len(phi), len(theta_lamb)))
        # now take the average over all the faces and azimuthal angles
        R_reshape_rear = R_all_rear.reshape((n_triangles_rear, len(phi), len(theta_lamb), len(options.wavelength)))

        R_weighted_rear = R_reshape_rear * hit_prob_reshape_rear[:, :, :, None]
        R_polar_rear = np.sum(np.mean(R_weighted_rear, 1), 0)

        A_surf_weighted_rear = A_reshape_rear * hit_prob_reshape_rear[None, :, :, :, None]
        A_polar_rear = np.sum(np.mean(A_surf_weighted_rear, 2), 1)

        # calculate travel distance for each ray
        I_rear = I_theta[:, None] * np.exp(-strt.widths[0] * strt.mats[1].alpha(options.wavelength[None, :]) / np.cos(theta_lamb)[:, None])

        R_1 = np.sum(I_theta[:, None]*R_polar_front, axis=0)
        R_2 = np.sum(I_theta[:, None]*R_polar_rear, axis=0)

        A_1 = np.sum(I_theta[:, None]*A_polar_front, axis=1)
        A_2 = np.sum(I_theta[:, None]*A_polar_rear, axis=1)
        # total probability of absorption in bulk:

        # infinite series:

        A_bulk = 1 - np.sum(I_rear, axis=0)

        T_1 = 1 - R_1 - np.sum(A_1, axis=0)
        T_2 = 1 - R_2 - np.sum(A_2, axis=0)

        r = (1 - R_1 * R_2 * (1 - A_bulk) ** 2)
        # if starting after reflection from front:
        # P_escape_front_down = (1 - A_bulk) ** 2 * T_1 * R_2 / r
        # P_escape_back_down = (1 - A_bulk) * T_2 / r
        # P_absorb_down = (A_bulk + (1 - A_bulk) * R_2 * A_bulk) / r
        # P_front_surf_down = (1 - A_bulk) ** 2 * R_2 * A_1 / r
        # P_rear_surf_down = (1 - A_bulk) * A_2 / r
        P_escape_front_down = (1 - A_bulk) * T_1 * R_2 / r
        P_escape_back_down = T_2 / r
        P_absorb_down = R_2 * A_bulk * (1 - A_bulk * R_1 + R_1)/ r
        P_front_surf_down = (1 - A_bulk) * R_2 * A_1 / r
        P_rear_surf_down = A_2 / r

        # if starting after reflection from rear:
        P_escape_front_up = T_1 / r
        P_escape_back_up = (1 - A_bulk) * T_2 * R_1 / r
        P_absorb_up = R_1 * A_bulk * (1 - A_bulk * R_2 + R_2)/ r
        P_front_surf_up = A_1 / r
        P_rear_surf_up = (1 - A_bulk) * R_1 * A_2 / r

        initial_down = xr.DataArray(np.stack((P_escape_front_down, P_absorb_down, P_escape_back_down)),
                     dims=['event', 'wavelength'],
                     coords={'event': ['R', 'A_bulk', 'T'], 'wavelength': options.wavelength})

        initial_up = xr.DataArray(np.stack((P_escape_front_up, P_absorb_up, P_escape_back_up)),
                     dims=['event', 'wavelength'],
                     coords={'event': ['R', 'A_bulk', 'T'], 'wavelength': options.wavelength})

        # does layer order need tp be flipped?
        front_surf_P = xr.DataArray(np.stack((P_front_surf_down, P_front_surf_up)),
                        dims=['direction', 'layer', 'wavelength'],
                        coords={'direction': [1, -1], 'wavelength': options.wavelength})

        rear_surf_P = xr.DataArray(np.stack((P_rear_surf_down, P_rear_surf_up)),
                        dims=['direction', 'layer', 'wavelength'],
                        coords={'direction': [1, -1], 'wavelength': options.wavelength})


        # Add a new dimension for the initial direction
        initial_down = initial_down.expand_dims({"direction": [1]})
        initial_up = initial_up.expand_dims({"direction": [-1]})

        # Concatenate the two xarrays along the new dimension
        merged = xr.concat([initial_down, initial_up], dim="direction")

    return merged, front_surf_P, rear_surf_P, [R_1, R_2]


def calculate_lambertian_profile(strt, I_wl, options, initial_direction,
                                 lambertian_results, alphas, position):

    I_theta = np.cos(theta_lamb)
    I_theta = I_theta / np.sum(I_theta)

    profile_wl = np.zeros((len(I_wl), len(position)))

    [R_top, R_bot] = lambertian_results

    if initial_direction == 1:
        R1 = R_bot # CHECK
        R2 = R_top

    else:
        R1 = R_top
        R2 = R_bot

    for i1, I0 in enumerate(I_wl):

        I = I0
        I_angular = I * I_theta
        direction = -initial_direction # first thing that happens is reflection!
        DA = np.zeros((len(theta_lamb), len(position)))

        while I > options.I_thresh:

            # 1st surf interaction
            I_angular = I_angular * R1[i1]

            # absorption

            DA_pass, _, I_angular = traverse_vectorised(
                strt.widths[0]*1e6,
                theta_lamb,
                alphas[i1],
                I_angular,
                position,
                options.I_thresh,
                direction,
            )

            DA += DA_pass

            I_angular = I_angular * R2[i1]

            direction = -direction

            DA_pass, _, I_angular = traverse_vectorised(
                strt.widths[0],
                theta_lamb,
                alphas[i1],
                I_angular,
                position,
                options.I_thresh,
                direction,
            )
            DA += DA_pass

            direction = -direction

            I = np.sum(I_angular)

        profile_wl[i1] = np.sum(DA, axis=0)

    return profile_wl
