from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE

plt.rcParams['font.size'] = '12'

mode = 'plot'
dev = 'cpu'

eps = 0.8
g = 9.80616
f = 7.292e-5
radius = 6.37122e6
poly_order = 3
#
# angle = 30 * (np.pi / 180)


def initial_condition(face):
    lat, long = face.geometry.lat_long(face.xs, face.ys, face.zs)
    lat_vec_x, lat_vec_y, lat_vec_z, long_vec_x, long_vec_y, long_vec_z = face.geometry.lat_long_vecs(face.xs, face.ys, face.zs)

    # lam = long
    # theta = lat
    u_0 = 2 * np.pi * 6.37122e6 / (12 * 24 * 3600)
    h_0 = 2.94e4 / g
    u_ = u_0 * np.cos(lat)
    h = h_0 - (1 / g) * (face.geometry.radius * f * u_0 + 0.5 * u_0 ** 2) * np.sin(lat) ** 2
    # h = h_0 - (1 / g) * (0.5 * u_0 ** 2) * np.cos(lat) ** 2
    # h -= (1 / g) *

    u = long_vec_x * u_
    v = long_vec_y * u_
    w = long_vec_z * u_

    return u, v, w, h


#(F1p + F2p) * (F1 + F2p) = F1p

exps = ['Cons.', 'Diss.']
coeffs = [0.5, 0.0]

ns = np.array([3, 5, 10, 15, 30]) #, 25, 30])
grid_spacing = 360 / (ns * 4 * 3)
tend = 5.0 * 24 * 3600

for exp, a in zip(exps, coeffs):
    h_errors = []
    vel_errors = []
    for n in ns:
        print('Running', n)
        solver = DGCubedSphereSWE(
            poly_order, n, n, g, f,
            eps, device=dev, solution=None, a=a, radius=radius,
            dtype=np.float64, damping=None,
        )

        solver0 = DGCubedSphereSWE(
            poly_order, n, n, g, f,
            eps, device=dev, solution=None, a=0.5, radius=radius,
            dtype=np.float64, damping=None
        )

        for face in solver.faces.values():
            face.set_initial_condition(*initial_condition(face))

        for face in solver0.faces.values():
            face.set_initial_condition(*initial_condition(face))

        h_norm = np.sqrt(sum(face.integrate(face.h ** 2) for face in solver.faces.values()))
        u_norm = np.sqrt(sum(face.integrate(face.u ** 2 + face.v ** 2 + face.w ** 2) for face in solver.faces.values()))

        while solver.time < tend:
            # if solver.time > 3600 * 24:
            #     solver.damping = 'adaptive'
            dt = solver.get_dt()
            dt = min(dt, tend - solver.time)
            solver.time_step(dt=dt)

        h_error = sum(f1.integrate((f1.h - f2.h) ** 2) for f1, f2 in zip(solver.faces.values(), solver0.faces.values()))
        h_error = np.sqrt(h_error) / h_norm

        u_error = sum(f1.integrate((f1.u - f2.u) ** 2 + (f1.v - f2.v) ** 2 + (f1.w - f2.w) ** 2) for f1, f2 in zip(solver.faces.values(), solver0.faces.values()))
        u_error = np.sqrt(u_error) / u_norm

        h_errors.append(h_error)
        vel_errors.append(u_error)


    h_errors = np.array(h_errors)
    vel_errors = np.array(vel_errors)

    r, *_ = linregress(np.log(grid_spacing), np.log(np.array(h_errors)))
    print('h convergence:', r)
    r, *_ = linregress(np.log(grid_spacing), np.log(np.array(vel_errors)))
    print('vel convergence:', r)

    plt.figure(1)
    plt.ylabel("Relative L2 error")
    plt.xlabel(r"Resolution (degrees)")
    plt.loglog(grid_spacing, h_errors, '--o', label=f"D {exp}")
    plt.loglog(grid_spacing, vel_errors, '--o', label=f"u {exp}")



plt.plot(grid_spacing, (1 / 3) * h_errors[0] * (grid_spacing[0] ** (-4)) * (grid_spacing.astype(np.float64) ** (4)), linestyle='--', label='4th order')
plt.plot(grid_spacing, 3 * h_errors[0] * (grid_spacing[0] ** (-3)) * (grid_spacing.astype(np.float64) ** (3)), linestyle='--', label='3rd order')
plt.grid()
plt.legend()
plt.savefig('./plots/williamson_2_error.png')
plt.show()
