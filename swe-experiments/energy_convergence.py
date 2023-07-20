from matplotlib import pyplot as plt
from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
import numpy as np
from scipy.stats import linregress
import scipy
import os

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

mode = 'run'
dev = 'cpu'

nx = ny = 5

g = 9.80616
f = 7.292e-5
radius = 6.37122e6
poly_order = 3

u_0 = 80
h_0 = 10_000

def initial_condition(face):

    def zonal_flow(lat):
        lat_0 = np.pi / 7
        lat_1 = 0.5 * np.pi - lat_0

        e_n = np.exp(-4 / (lat_1 - lat_0) ** 2)

        out = np.zeros_like(lat)
        mask = (lat_0 < lat) & (lat < lat_1)
        out[mask] = (u_0 / e_n) * np.exp(1 / ((lat[mask] - lat_0) * (lat[mask] - lat_1)))
        return out


    def func(lat):
        u_ = zonal_flow(lat)
        out = -radius * u_ * (2 * np.sin(lat) * f + np.tan(lat) * u_ / radius)

        return out / g

    lats = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 100_000)
    dlat = np.diff(lats).mean()
    vals = func(lats)
    h_reg = h_0 + (np.cumsum(vals) - 0.5 * (vals[0] + vals[-1])) * dlat
    h_interp = scipy.interpolate.interp1d(lats, h_reg)


    lat, long = face.geometry.lat_long(face.xs, face.ys, face.zs)
    lat_vec_x, lat_vec_y, lat_vec_z, long_vec_x, long_vec_y, long_vec_z = face.geometry.lat_long_vecs(face.xs, face.ys, face.zs)
    h = h_interp(lat)

    alpha = 1 / 3
    beta = 1 / 15
    lat_2 = np.pi / 4
    h_pert = 120 * np.cos(lat) * np.exp(-(long / alpha) ** 2) * np.exp(-((lat_2 - lat) / beta) ** 2)
    h += h_pert

    u_ = zonal_flow(lat)
    u = long_vec_x * u_
    v = long_vec_y * u_
    w = long_vec_z * u_

    return u, v, w, h

dts = np.array([50, 40, 30, 20, 10]).astype(np.float64)
energy_errors = []
tend = 10 * 24 * 3600 # 10 days

for dt in dts:
    print('Running', dt)
    solver = DGCubedSphereSWE(
        poly_order, nx, ny, g, f,
        0.8, device=dev, solution=None, a=0.0, radius=radius,
        dtype=np.float64
    )

    for face in solver.faces.values():
        face.set_initial_condition(*initial_condition(face))

    while solver.time < tend:
        dt = min(dt, tend - solver.time)
        solver.time_step(dt=dt)

    energy_error = abs(solver.energy_list[-1] - solver.energy_list[0]) / solver.energy_list[0]
    energy_errors.append(energy_error)
    print(energy_error)


energy_errors = np.array(energy_errors)

r, *_ = linregress(np.log(dts), np.log(np.array(energy_errors)))
print('Energy convergence:', r)

plt.figure(1)
plt.ylabel("Relative L2 error")
plt.xlabel("$\Delta t$ (s)")
plt.loglog(dts, energy_errors, '--o', label="Energy error")
#
plt.plot(dts, 0.8 * energy_errors[0] * (dts[0] ** -3) * (dts ** 3), linestyle='--', label='3rd order')
plt.grid()
plt.legend()
plt.savefig('./plots/energy_error.png')
plt.show()
