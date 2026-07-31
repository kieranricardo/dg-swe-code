from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from dg_swe.dg_cubed_sphere_swe_numpy import DGCubedSphereSWENumpy
import os
import time

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

mode = 'restart'
i_start = 720
dev = 'cpu'

eps = 0.8 * 2
tangent_diss = True
h_diss = True

if h_diss:
    ah = 0.5
else:
    ah = 0.0
nx = ny = 64

g = 9.80616 / 250
f = 7.292e-5
radius = 6.37122e6
u_0 = 0.5
h_0 = 5960.0
s_0 = 4000

def get_fn_template(day, tangent_diss):
    suffix = ''
    if tangent_diss:
        suffix = suffix + 'tangent_diss'

    if h_diss:
        suffix = suffix + '_h_diss'

    if s_0 == 4000:
        suffix = suffix + '_big_mountain'
    elif s_0 != 3000:
        raise ValueError(f'suffix: expedcted one of 3000, 4000. Found {s_0}.')

    return f"reduced_williamson_5_day_{day}_nx{nx}_p{poly_order}_{suffix}.npy"
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
    b = s_0 * (1 - r / R)
    b[b < 0.0] = 0.0
    # print('b min max:', b.min(), b.max())
    # print()

    u = long_vec_x * u_
    v = long_vec_y * u_
    w = long_vec_z * u_

    return u, v, w, h - b, b


def plot_data(idx, label, plot_func, vmin=None, vmax=None):
    fig = plt.figure(idx, figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.set_xlabel("Longitude (degrees)")
    ax.set_ylabel("Latitiude (degrees)")
    #
    im = solver.latlong_triangular_plot(ax, plot_func=plot_func, levels=100, vmin=vmin, vmax=vmax)
    plt.colorbar(im[0])

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


solver = DGCubedSphereSWENumpy(
    poly_order, nx, ny, g, f,
    eps, device=dev, solution=None, a=0.5, ah=ah, radius=radius,
    dtype=np.float64, damping=None, tangent_diss=tangent_diss
)

for face in solver.faces.values():
    face.set_initial_condition(*initial_condition(face))

print('Initial dt:', solver.get_dt())

if mode == 'run':
    t0 = time.time()
    for i in range(100):
        print('\nRunning day', i + 1)
        tend = solver.time + 3600 * 24
        print('h min max:', min(f.h.min() for f in solver.faces.values()), max(f.h.max() for f in solver.faces.values()), solver.get_dt())
        is_nan = any(np.isnan(f.h).any() for f in solver.faces.values())
        if is_nan:
            print(f'NaN at day {i + 1}.')
            break
        while solver.time < tend:
            dt = solver.get_dt()
            dt = min(dt, tend - solver.time)
            solver.time_step(dt=dt, order=34)

        if ((i + 1) % 20 == 0):
            fn_template = get_fn_template(i + 1, tangent_diss)
            print('Saving:', fn_template)
            solver.save_restart(fn_template, 'data')
    t1 = time.time()
    print('Wall time for 1 day:', (t1 - t0) / 20)

if mode == 'restart':

    fn_template = get_fn_template(i_start, tangent_diss)
    print('Loading:', fn_template)
    solver.load_restart(fn_template, 'data')

    for i in range(i_start, i_start+360):
        print('\nRunning day', i + 1)
        tend = solver.time + 3600 * 24
        print('h min max:', min(f.h.min() for f in solver.faces.values()), max(f.h.max() for f in solver.faces.values()), solver.get_dt())
        is_nan = any(np.isnan(f.h).any() for f in solver.faces.values())
        if is_nan:
            print(f'NaN at day {i+1}.')
            break
        while solver.time < tend:
            dt = solver.get_dt()
            dt = min(dt, tend - solver.time)
            solver.time_step(dt=dt, order=34)

        if ((i + 1) % 1 == 0):
            fn_template = get_fn_template(i + 1, tangent_diss)
            print('Saving:', fn_template)
            solver.save_restart('np_' + fn_template, 'data')
        # fn_template = f"reduced_williamson_5_day_{i + 1}.npy"
        # solver.save_restart(fn_template, 'data')

if mode == 'process-data':

    day = 0
    fn_template = get_fn_template(day, tangent_diss=True)
    solver.save_restart(fn_template, 'data')

    vort = solver.vorticity()

    for name in solver.face_names:
        fp = solver.make_fp('vorticity', name, fn_template, 'data')
        print(fp)
        np.save(fp, vort[name] - solver.faces[name].f)

        face = solver.faces[name]
        lat_long = face.geometry.lat_long(face.xs, face.ys, face.zs)

        fp = solver.make_fp('lat', name, fn_template, 'data')
        np.save(fp, lat_long[0])

        fp = solver.make_fp('long', name, fn_template, 'data')
        np.save(fp, lat_long[1])


    for day in range(20, 721, 20):
        fn_template = get_fn_template(day, tangent_diss=True)
        solver.load_restart(fn_template, 'data')

        vort = solver.vorticity()

        for name in solver.face_names:
            fp = solver.make_fp('vorticity', name, fn_template, 'data')
            print(fp)
            np.save(fp, vort[name] - solver.faces[name].f)


h_plot_func=lambda s: s.h

vort_plot_func = lambda s: s.vorticity() - s.f

pv_plot_func = lambda s: (s.vorticity() - s.f) / (s.h)

# exit(0)
#days = np.array([1, 2, 3, 4]) * 360
# days = list(days) + [1800,]
days = [720,]
for i, day in enumerate(days):
# for i, day in enumerate([360, 400, 700]):
    fn_template = get_fn_template(day, tangent_diss=True)
    solver.load_restart(fn_template, 'data')
    print(min(f.h.min() for f in solver.faces.values()))
    # plot_data(2 * i + 1, f'vort_day_{day}_nx{nx}_p{poly_order}', vort_plot_func, vmin=-3e-5, vmax=3e-5)
    plot_data(2 * i + 1, f'vort_day_{day}_nx{nx}_p{poly_order}', pv_plot_func)
    # plot_data(2 * i + 1, f'pv_day_{day}_nx{nx}_p{poly_order}', pv_plot_func)
    plot_data(2 * i + 2, f'height_day_{day}_nx{nx}_p{poly_order}', h_plot_func)

    # fn_template = fn_template = get_fn_template(day, tangent_diss=False)
    # solver.load_restart(fn_template, 'data')
    # plot_data(2 * i + 3, f'vort_day_{day}_nx{nx}_p{poly_order}_tangent_diss', vort_plot_func)
    # # plot_data(2 * i + 1, f'pv_day_{day}_nx{nx}_p{poly_order}', pv_plot_func)
    # plot_data(2 * i + 4, f'height_day_{day}_nx{nx}_p{poly_order}_tangent_diss', h_plot_func)



plt.show()