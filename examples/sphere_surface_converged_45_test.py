from joblib import Parallel, delayed
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as mtri
from scipy.spatial import ConvexHull
from rayflare.textures import xyz_texture
from time import time
import os
import seaborn as sns
from solcore import material
from rayflare.ray_tracing import rt_structure
from rayflare.options import default_options


d_bulk = 0
# # going to get converted to um (multiplied by 1e6)
#
# def points_on_sphere(N, radius, h):
#     """ Generate N evenly distributed points on the unit sphere centered at
#     the origin. Uses the 'Golden Spiral'.
#     Code by Chris Colbert from the numpy-discussion list.
#     """
#     phi = (1 + np.sqrt(5)) / 2 # the golden ratio
#     long_incr = 2*np.pi / phi # how much to increment the longitude
#
#     dz = 2.0 / float(N) # a unit sphere has diameter 2
#     bands = np.arange(N) # each band will have one point placed on it
#     z = bands * dz - 1 + (dz/2) # the height z of each band/point
#     r = np.sqrt(1 - z*z) # project onto xy-plane
#     az = bands * long_incr # azimuthal angle of point modulo 2 pi
#     x = r * np.cos(az)
#     y = r * np.sin(az)
#
#     x = radius * x
#     y = radius * y
#     z = radius * z + h
#     return x, y, z
#
# def average_g(triples):
#     return np.mean([triple[2] for triple in triples])
#
# # fig = plt.figure()
# # ax = fig.add_subplot(111, projection='3d')
# X, Y, Z = points_on_sphere(2**15, radius=0.8, h=0.242)
#
# X = X[Z >= 0]
# Y = Y[Z >= 0]
# Z = Z[Z >= 0]
#
# # add a line of points at Z = 0 to make sure there won't be a gap between sphere and planar surface
#
r_cross = np.sqrt(0.8**2 - 0.242**2)

# n_points = np.int(np.sum(np.all([np.sqrt(X**2 + Y**2) < r_cross + 0.01, np.sqrt(X**2 + Y**2) > r_cross - 0.01], axis=0)/2))
n_points = 10001
# new points
Z_new = np.zeros(n_points)
phis = np.linspace(0, 2*np.pi, n_points)
X_new = r_cross*np.cos(phis)
Y_new = r_cross*np.sin(phis)
#
# X = np.hstack([X_new, X])
# Y = np.hstack([Y_new, Y])
# Z = np.hstack([Z_new, Z])
#
#
# Triples = np.array(list(zip(X, Y, Z)))
#
# Triples_back = np.array(list(zip(X, Y, -Z - d_bulk)))
#
#
#
# hull = ConvexHull(Triples)
# hull_back = ConvexHull(Triples_back)
# triangles = hull.simplices
# triangles_back = hull_back.simplices
#
# colors = np.array([average_g([Triples[idx] for idx in triangle]) for
#                    triangle in triangles_back])
#
# # collec = ax.plot_trisurf(mtri.Triangulation(X, Y, triangles),
# # #         Z, shade=False, cmap=plt.get_cmap('Blues'), array=colors,
# # #         edgecolors='none')
# # # collec.autoscale()
# #
# # plt.show()
#
# [front, back] = xyz_texture(X, Y, Z)
#
# front.simplices = triangles
# front.P_0s = front.Points[triangles[:,0]]
# front.P_1s = front.Points[triangles[:,1]]
# front.P_2s = front.Points[triangles[:,2]]
# front.crossP = np.cross(front.P_1s - front.P_0s, front.P_2s - front.P_0s)
# front.size = front.P_0s.shape[0]
# front.zcov = 0
#
# back.simplices = triangles_back
# back.P_0s = back.Points[triangles_back[:,0]]
# back.P_1s = back.Points[triangles_back[:,1]]
# back.P_2s = back.Points[triangles_back[:,2]]
# back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)
# back.size = back.P_0s.shape[0]
# back.zcov = 0
#
# cross_normalized = back.crossP/np.sqrt(np.sum(back.crossP**2,1))[:,None]
#
# flat = np.abs(cross_normalized[:,2]) > 0.9
# bottom_surface = np.all((back.P_0s[:,2] > -0.1, back.P_1s[:,2] > -0.1, back.P_2s[:,2] > -0.1), axis=0)
# bottom_planar = np.all((flat, bottom_surface), axis=0)
#
#
# back.simplices = triangles_back[~bottom_planar]
# back.P_0s = back.P_0s[~bottom_planar]
# back.P_1s = back.P_1s[~bottom_planar]
# back.P_2s = back.P_2s[~bottom_planar]
# back.crossP = back.crossP[~bottom_planar]
# back.size = back.P_0s.shape[0]
#
#
# hyperhemi = [back, front]
#
#
GaAs = material('GaAs')()
Air = material('Air')()

#
# flat_surf = planar_surface(size=0.77) # size in microns
#
flat_surf_2 = xyz_texture(np.hstack([0,X_new]), np.hstack([0,Y_new]), np.hstack([0,Z_new]))
#
# reg_pyr = regular_pyramids()
#
#
# X_norm = np.mean([back.P_0s[:,0], back.P_1s[:,0], back.P_2s[:,0]], 0)
# Y_norm = np.mean([back.P_0s[:,1], back.P_1s[:,1], back.P_2s[:,1]], 0)
# Z_norm = np.mean([back.P_0s[:,2], back.P_1s[:,2], back.P_2s[:,2]], 0)
#
#
# # make all the normals points inwards/"upwards":
# from copy import deepcopy
#
# for index in np.arange(len(X_norm)):
#     current_N = back.crossP[index, :]
#     X_sign = -np.sign(X_norm[index])
#     # N[2] should be < 0
#     if np.sign(current_N[0]) != X_sign:
#         P1_current = deepcopy(back.P_1s[index, :])
#         P2_current = deepcopy(back.P_2s[index, :])
#         back.P_1s[index, :] = P2_current
#         back.P_2s[index, :] = P1_current
#
#         s1_current = back.simplices[index, 1]
#         s2_current = back.simplices[index, 2]
#         back.simplices[index, 1] = s2_current
#         back.simplices[index, 2] = s1_current
#
#
# back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)
#
# for index in np.arange(len(X_norm)):
#     current_N = back.crossP[index, :]
#     Y_sign = -np.sign(Y_norm[index])
#     # N[2] should be < 0
#     if np.sign(current_N[1]) != Y_sign:
#         P1_current = deepcopy(back.P_1s[index, :])
#         P2_current = deepcopy(back.P_2s[index, :])
#         back.P_1s[index, :] = P2_current
#         back.P_2s[index, :] = P1_current
#
#         s1_current = back.simplices[index, 1]
#         s2_current = back.simplices[index, 2]
#         back.simplices[index, 1] = s2_current
#         back.simplices[index, 2] = s1_current
#
#
# back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)
#
# above_middle = np.where(Z_norm > -0.222)[0]
# below_middle = np.where(Z_norm < -0.262)[0]
#
#
# for index in above_middle:
#     current_N = back.crossP[index, :]
#
#     # N[2] should be < 0
#     if current_N[2] > 0:
#         P1_current = deepcopy(back.P_1s[index, :])
#         P2_current = deepcopy(back.P_2s[index, :])
#         back.P_1s[index, :] = P2_current
#         back.P_2s[index, :] = P1_current
#
#         s1_current = back.simplices[index, 1]
#         s2_current = back.simplices[index, 2]
#         back.simplices[index, 1] = s2_current
#         back.simplices[index, 2] = s1_current
#
#
# for index in below_middle:
#     current_N = back.crossP[index, :]
#
#     # N[2] should be > 0
#     if current_N[2] < 0:
#         P1_current = deepcopy(back.P_1s[index, :])
#         P2_current = deepcopy(back.P_2s[index, :])
#         back.P_1s[index, :] = P2_current
#         back.P_2s[index, :] = P1_current
#
#         s1_current = back.simplices[index, 1]
#         s2_current = back.simplices[index, 2]
#         back.simplices[index, 1] = s2_current
#         back.simplices[index, 2] = s1_current
#
#
# back.crossP = np.cross(back.P_1s - back.P_0s, back.P_2s - back.P_0s)
#
#
# X_norm = np.mean([back.P_0s[:,0], back.P_1s[:,0], back.P_2s[:,0]], 0)
# Y_norm = np.mean([back.P_0s[:,1], back.P_1s[:,1], back.P_2s[:,1]], 0)
# Z_norm = np.mean([back.P_0s[:,2], back.P_1s[:,2], back.P_2s[:,2]], 0)

from rayflare.textures.standard_rt_textures import hyperhemisphere

[front, back] = hyperhemisphere(2**13, 0.8, 0.242)

hyperhemi = [back, front]

X_norm = np.mean([front.P_0s[:, 0], front.P_1s[:, 0], front.P_2s[:, 0]], 0)
Y_norm = np.mean([front.P_0s[:, 1], front.P_1s[:, 1], front.P_2s[:, 1]], 0)
Z_norm = np.mean([front.P_0s[:, 2], front.P_1s[:, 2], front.P_2s[:, 2]], 0)

fig = plt.figure()


ax = plt.subplot(111, projection='3d')
ax.view_init(elev=30., azim=60)
ax.plot_trisurf(front.Points[:,0], front.Points[:,1], front.Points[:,2],
                triangles=front.simplices,  shade=False, cmap=plt.get_cmap('Blues'),# array=colors,
        edgecolors='none')
# ax.plot_trisurf(flat_surf_2[0].Points[:,0], flat_surf_2[0].Points[:,1], flat_surf_2[0].Points[:,2],
#                 triangles=flat_surf_2[0].simplices)

#
ax.quiver(X_norm, Y_norm, Z_norm, front.crossP[:,0], front.crossP[:,1], front.crossP[:,2], length=0.1, normalize=True,
          color='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

X_norm = np.mean([back.P_0s[:, 0], back.P_1s[:, 0], back.P_2s[:, 0]], 0)
Y_norm = np.mean([back.P_0s[:, 1], back.P_1s[:, 1], back.P_2s[:, 1]], 0)
Z_norm = np.mean([back.P_0s[:, 2], back.P_1s[:, 2], back.P_2s[:, 2]], 0)

fig = plt.figure()


ax = plt.subplot(111, projection='3d')
ax.view_init(elev=30., azim=60)
ax.plot_trisurf(back.Points[:,0], back.Points[:,1], back.Points[:,2],
                triangles=back.simplices,  shade=False, cmap=plt.get_cmap('Blues'),# array=colors,
        edgecolors='none')
# ax.plot_trisurf(flat_surf_2[0].Points[:,0], flat_surf_2[0].Points[:,1], flat_surf_2[0].Points[:,2],
#                 triangles=flat_surf_2[0].simplices)

#
ax.quiver(X_norm, Y_norm, Z_norm, back.crossP[:,0], back.crossP[:,1], back.crossP[:,2], length=0.1, normalize=True,
          color='k')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()

rtstr = rt_structure(textures=[flat_surf_2, hyperhemi],
                    materials = [GaAs],
                    widths=[d_bulk], incidence=Air, transmission=Air)

options = default_options()

options.x_limits = [-0.05, 0.05]
options.y_limits = [-0.05, 0.05]

options.initial_material = 1
options.initial_direction = 1

options.xlim = 0.05
options.ylim = 0.05


options.periodic = 0

nxs = [20]

thetas = np.linspace(0, np.pi / 2 - 0.05, 10)

pal = sns.color_palette("rocket", 4)

plt.figure()

for i1, nx in enumerate(nxs):

    options.wavelengths = np.array([6e-6])
    options.parallel = False
    options.n_rays = nx**2

    options.theta = 0.1
    options.nx = nx
    options.ny = nx
    options.pol = 'u'

    print(options.n_rays)

    minimum_angle = np.pi - np.pi*45/180

    T_values = np.zeros(len(thetas))
    T_total = np.zeros(len(thetas))
    n_interactions = np.zeros(len(thetas))
    theta_distribution = np.zeros((len(thetas), options.n_rays))


    start = time()

    for j1, th in enumerate(thetas):
        print(j1, th)

        options.theta_in = th
        result = rtstr.calculate(options)
        T_values[j1] = np.sum(result['thetas'] > minimum_angle)/options.n_rays
        T_total[j1] = result['T']
        n_interactions[j1] = np.mean(result['n_interactions'])
        theta_distribution[j1] = result['thetas']

    print(time() - start)

    min_angle_old = np.pi - 17.5*np.pi/180

    T_175 = np.array([np.sum(x > min_angle_old) / options.n_rays for x in theta_distribution])
    T_45 = np.array([np.sum(x > minimum_angle) / options.n_rays for x in theta_distribution])


# T_v_ref = np.loadtxt('results/ref_T_values.txt')
T_tot_ref = np.loadtxt('results/ref_T_total.txt')
T_175_ref = np.loadtxt('results/ref_T_175.txt')
T_45_ref = np.loadtxt('results/ref_T_45.txt')
n_int_ref = np.loadtxt('results/ref_n_interactions.txt')

plt.figure()
plt.plot(thetas*180/np.pi, T_total, '-', label=str(options.n_rays), color=pal[0])
plt.scatter(thetas*180/np.pi, T_total, edgecolors=pal[0], facecolors='none')
plt.plot(thetas*180/np.pi, T_tot_ref, '--', label=str(options.n_rays), color=pal[0])

# plt.plot(thetas*180/np.pi, T_values, '-', label=str(options.n_rays), color=pal[1])
# plt.scatter(thetas*180/np.pi, T_values, edgecolors=pal[1], facecolors='none')
# plt.plot(thetas*180/np.pi, T_v_ref, '--', label=str(options.n_rays), color=pal[1])

plt.plot(thetas*180/np.pi, T_45, '-', color=pal[2])
plt.scatter(thetas*180/np.pi, T_45, edgecolors=pal[2], facecolors='none')
plt.plot(thetas*180/np.pi, T_45_ref, '--', color=pal[2])

plt.plot(thetas*180/np.pi, T_175, '-', color=pal[3])
plt.scatter(thetas*180/np.pi, T_175, edgecolors=pal[3], facecolors='none')
plt.plot(thetas*180/np.pi, T_175_ref, '--', color=pal[3])

plt.xlim(0, 90)
plt.xlabel(r'$\beta$ (rads)')
plt.ylabel('Transmission')
plt.show()
#
# for ln in [17.5, 2*17.5, 3*17.5, 4*17.5]:
#     plt.axvline(x=ln)


plt.show()

plt.figure()
plt.plot(thetas*180/np.pi, n_interactions)
plt.show()

# np.savetxt('results/ref_T_values.txt',  T_values)
# np.savetxt('results/ref_T_total.txt',  T_total)
# np.savetxt('results/ref_T_175.txt',  T_175)
# np.savetxt('results/ref_T_45.txt',  T_45)
# np.savetxt('results/ref_n_interactions.txt', n_interactions)

