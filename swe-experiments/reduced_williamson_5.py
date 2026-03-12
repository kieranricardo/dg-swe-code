from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
import os

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

mode = 'run'
dev = 'cpu'

eps = 0.8
g = 9.80616 / 250
f = 7.292e-5
radius = 6.37122e6
u_0 = 0.5
h_0 = 5960.0

# print('Froude number:', u_0 / np.sqrt(g * h_0))

poly_order = 3

def initial_condition(face):
    lat, long = face.geometry.lat_long(face.xs, face.ys, face.zs)
    lat_vec_x, lat_vec_y, lat_vec_z, long_vec_x, long_vec_y, long_vec_z = face.geometry.lat_long_vecs(face.xs, face.ys, face.zs)

    # lam = long
    # theta = lat

    u_ = u_0 * np.cos(lat)
    h = h_0 - (1 / g) * (face.geometry.radius * f * u_0 + 0.5 * u_0 ** 2) * np.sin(lat) ** 2
    # h = h_0 - (1 / g) * (0.5 * u_0 ** 2) * np.cos(lat) ** 2
    # h -= (1 / g) *

    R = np.pi / 9
    r = np.sqrt((long)**2 + (lat - np.pi / 6)**2)
    b = 2_000.0 * (1 - r / R)
    b[b < 0.0] = 0.0
    # print('b min max:', b.min(), b.max())
    # print()

    u = long_vec_x * u_
    v = long_vec_y * u_
    w = long_vec_z * u_

    return u, v, w, h - b, b





def plot_data(idx, label, plot_func):
    fig = plt.figure(idx, figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitiude (degrees)")
    #
    im = solver.latlong_triangular_plot(ax, plot_func=plot_func, levels=50)
    plt.colorbar(im[0])
    im = solver.latlong_triangular_plot(ax, plot_func=plot_func, levels=50)

    # print('\nb min max:', min(f.b.min() for f in solver.faces.values()), max(f.b.max() for f in solver.faces.values()))
    # print('h min max:', min(f.h.min() for f in solver.faces.values()), max(f.h.max() for f in solver.faces.values()))
    # print('h+b min max:', min((f.h + f.b).min() for f in solver.faces.values()), max((f.h + f.b).max() for f in solver.faces.values()))

    plt.savefig(f'./plots/reduced_williamson_5_{label}.png')


def plot_orography(idx):
    fig = plt.figure(idx, figsize=(8, 4))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitiude (degrees)")

    vmin = min(f.b.min() for f in solver.faces.values())
    vmax = max(f.b.max() for f in solver.faces.values())
    #n = int((vmax - vmin) / 200)
    n = 100
    im = solver.latlong_triangular_plot(ax, vmin=vmin, vmax=vmax, plot_func=lambda s: s.b, n=n)
    plt.colorbar(im[0])

    print('b min max:', min(f.b.min() for f in solver.faces.values()), max(f.b.max() for f in solver.faces.values()))

    plt.savefig(f'./plots/williamson_5_orography.png')

solver = DGCubedSphereSWE(
    poly_order, 32, 32, g, f,
    eps, device=dev, solution=None, a=0.5, radius=radius,
    dtype=np.float64, damping=None
)

for face in solver.faces.values():
    face.set_initial_condition(*initial_condition(face))

print('Initial dt:', solver.get_dt())

# exit(0)

# plot_orography(1)
mode = 'restart'
i_start = 40
if mode == 'run':

    for i in range(20):
        print('\nRunning day', i)
        tend = solver.time + 3600 * 24
        print('h min max:', min(f.h.min() for f in solver.faces.values()), max(f.h.max() for f in solver.faces.values()), solver.get_dt())
        while solver.time < tend:
            dt = solver.get_dt()
            dt = min(dt, tend - solver.time)
            solver.time_step(dt=dt)

        fn_template = f"reduced_williamson_5_day_{i + 1}.npy"
        solver.save_restart(fn_template, 'data')

if mode == 'restart':

    fn_template = f"reduced_williamson_5_day_{i_start}.npy"
    solver.load_restart(fn_template, 'data')

    for i in range(i_start, i_start + 20):
        print('\nRunning day', i)
        tend = solver.time + 3600 * 24
        print('h min max:', min(f.h.min() for f in solver.faces.values()), max(f.h.max() for f in solver.faces.values()), solver.get_dt())
        while solver.time < tend:
            dt = solver.get_dt()
            dt = min(dt, tend - solver.time)
            solver.time_step(dt=dt)

        fn_template = f"reduced_williamson_5_day_{i + 1}.npy"
        solver.save_restart(fn_template, 'data')


day = 20
# plot_func=lambda s: s.h + s.b
plot_func = lambda s: s.vorticity() - s.f

for day in [20, 40, 60]:
    fn_template = f"reduced_williamson_5_day_{day}.npy"
    solver.load_restart(fn_template, 'data')
    plot_data(3, f'vorticity_day_{day}', plot_func)


plt.show()