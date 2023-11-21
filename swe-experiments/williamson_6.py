from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
import os

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

mode = 'plot'
dev = 'cpu'

eps = 0.8
g = 9.80616
f = 7.292e-5
radius = 6.37122e6
poly_order = 3

def initial_condition(face):
    lat, long = face.geometry.lat_long(face.xs, face.ys, face.zs)
    lat_vec_x, lat_vec_y, lat_vec_z, long_vec_x, long_vec_y, long_vec_z = face.geometry.lat_long_vecs(face.xs, face.ys, face.zs)

    # theta = lat
    o = K = 7.848e-6
    h0 = 8e3
    a = radius

    ct = np.cos(lat)
    st = np.sin(lat)

    R = 4
    A = 0.5 * o * (2 * f + o) * ct**2
    A += 0.25 * K**2 * ct**(2 * R) * (
        (R + 1) * ct**2 + (2 * R**2 - R - 2) - 2 * R**2 * ct**(-2)
    )

    B = 2 * (f + o) * K / ((R + 1) * (R + 2))
    B *= ct**R * (
        (R**2 + 2 * R + 2) - (R + 1)**2 * ct**2
    )

    C = 0.25 * K**2 * ct**(2 * R) * (
        (R + 1) * ct**2 - (R + 2)
    )

    gh = g * h0
    gh += (a**2) * (A + B * np.cos(R * long) + C * np.cos(2 * R * long))
    h = gh / g

    u_ = a * o * ct
    u_ += a * K * ct**(R - 1) * (R * st**2 - ct**2) * np.cos(R * long)
    v_ = -a * K * R * ct**(R - 1) * st * np.sin(R * long)

    u = long_vec_x * u_ + lat_vec_x * v_
    v = long_vec_y * u_ + lat_vec_y * v_
    w = long_vec_z * u_ + lat_vec_z * v_

    angular_group_velocity = (R * (3 + R) * o - 2 * f)
    angular_group_velocity /= (1 + R) * (2 + R)
    angular_group_velocity *= (180 / np.pi) * 24 * 3600
    return u, v, w, h


def plot_height(idx, label):
    fig = plt.figure(idx, figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitiude (degrees)")

    vmin = 8000
    vmax = 10600
    n = int((vmax - vmin) / 200)
    levels = vmin + 200 * np.arange(n + 1)
    im = solver.latlong_triangular_plot(ax, vmin=vmin, vmax=vmax, plot_func=lambda s: s.h, n=n, levels=levels)
    plt.colorbar(im[0])

    im = solver.latlong_triangular_plot(ax, vmin=vmin, vmax=vmax, plot_func=lambda s: s.h, n=n, lines=True, levels=levels)

    print('h min max:', min(f.h.min() for f in solver.faces.values()), max(f.h.max() for f in solver.faces.values()))

    plt.savefig(f'./plots/williamson_6_{label}.png')


tend = 14.0 * 24 * 3600

solver = DGCubedSphereSWE(
    poly_order, 16, 16, g, f,
    eps, device=dev, solution=None, a=0.5, radius=radius,
    dtype=np.float64, damping=None
)

for face in solver.faces.values():
    face.set_initial_condition(*initial_condition(face))

plot_height(1, 'ic')

mode = 'plot'

if mode == 'run':
    for i in range(14):
        print('Running day', i)
        tend = solver.time + 3600 * 24
        print('h min max:', min(f.h.min() for f in solver.faces.values()), max(f.h.max() for f in solver.faces.values()), solver.get_dt())
        while solver.time < tend:
            dt = solver.get_dt()
            dt = min(dt, tend - solver.time)
            solver.time_step(dt=dt)

        fn_template = f"williamson_6_day_{i + 1}.npy"
        solver.save_restart(fn_template, 'data')


fn_template = "williamson_6_day_14.npy"
solver.load_restart(fn_template, 'data')
plot_height(2, 'final')

plt.show()