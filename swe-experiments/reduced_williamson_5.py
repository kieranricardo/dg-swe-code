from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
import os
import time
from mpi4py import MPI
import cmocean
import argparse

plt.rcParams['font.size'] = '12'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size == 1:
    nprocx = nprocy = 1
else:
    nprocx = nprocy = int(np.sqrt(size // 6))

cfl = 1.3
h_diss = True

if h_diss:
    ah = 0.5
else:
    ah = 0.0

parser = argparse.ArgumentParser()
parser.add_argument('--order', type=int, help='Polynomial order')
parser.add_argument('--nx', type=int, help='Number of cells in horizontal')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--restart', action='store_true')
parser.add_argument('--day', type=int, help='Polynomial order')
parser.add_argument('--flux_type', type=str, help='Flux type')
args = parser.parse_args()

if args.plot:
    mode = 'plot'
    day = args.day
elif args.restart:
    mode = 'restart'    
    i_start = args.day
else:
    mode = 'run'    
    day = args.day

nx = ny = args.nx
poly_order = args.order
g = 9.80616 / 250
f = 7.292e-5
radius = 6.37122e6
u_0 = 0.5
h_0 = 5960.0
s_0 = 3000
flux_type = args.flux_type

# max wave speed approx 20 towards later end of simulation
# can be a bit higher though
wave_speed = 20 
# 4 * nx * 3 nodal points around equator
dx_average = 2 * np.pi * radius / (4 * nx * 3) 
# 0.8 factor for smallest cubed sphere spacing
# 0.8 factor for smallest nodal spacing (third order)
dx_min = 0.8 * 0.8 * dx_average
# 0.5 factor for 2 dimensions
dt = cfl * 0.5 * dx_min / wave_speed

def get_fn_template(day=None):
    suffix = ''

    if flux_type == "lmars":
        suffix = suffix + f'lmars'
    elif flux_type == "barth":
        suffix = suffix + f'barth'
    elif flux_type == "barth_normal_tangent":
        suffix = suffix + f'barth_normal_tangent'
    else:
        if flux_type == "old_tangent":
            suffix = suffix + 'old_tangent_diss'
        elif flux_type == "standard_tangent":
            suffix = suffix + 'tangent_diss'

        if h_diss:
            suffix = suffix + '_h_diss'

        if s_0 == 4000:
            suffix = suffix + '_big_mountain'
        elif s_0 != 3000:
            raise ValueError(f'suffix: expedcted one of 3000, 4000. Found {s_0}.')

    if day is None:
        return f"reduced_williamson_5_day_nx{nx}_p{poly_order}_{suffix}"
    else:
        return f"reduced_williamson_5_day_nx{nx}_p{poly_order}_{suffix}_{day}"

data_dir = os.path.join('data', get_fn_template())
plot_dir = os.path.join('plots', get_fn_template())

if rank == 0:
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)

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

    # print('b min max:', min(f.b.min() for f in solver.faces.values()), max(f.b.max() for f in solver.faces.values()))

    plt.savefig(f'./plots/williamson_5_orography.png')


def daily_diagnostics_fp():
    return os.path.join(data_dir, f"daily_diagnostics_{get_fn_template()}.npy")


def collect_daily_diagnostics(solver, day):
    energy = solver.integrate(solver.entropy())
    enstrophy = solver.integrate(solver.enstrophy())
    if rank != 0:
        return None
    return [float(day), float(energy), float(enstrophy)]


def save_daily_diagnostics(rows):
    if rank != 0:
        return
    np.save(daily_diagnostics_fp(), np.asarray(rows, dtype=np.float64))


def load_daily_diagnostics():
    return np.load(daily_diagnostics_fp())


def plot_daily_diagnostics():
    if rank != 0:
        return

    fp = daily_diagnostics_fp()
    if not os.path.exists(fp):
        print("No daily diagnostics found:", fp)
        return

    diagnostics = load_daily_diagnostics()
    days = diagnostics[:, 0]
    energy = diagnostics[:, 1]
    enstrophy = diagnostics[:, 2]

    plt.figure(figsize=(7, 4), dpi=400)
    plt.plot(days, (energy - energy[0]) / energy[0], label="Energy")
    plt.plot(days, (enstrophy - enstrophy[0]) / enstrophy[0], label="Enstrophy")
    plt.xlabel("Time (days)")
    plt.ylabel("Relative error")
    plt.yscale("symlog", linthresh=1.0e-15)
    plt.grid(True, which="both")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"daily_diagnostics_{get_fn_template()}.png"))


solver = DGCubedSphereSWE(
    poly_order, nx, ny, g, f,
    eps=0.0, a=0.5, ah=ah, radius=radius,
    dtype=np.float64, flux_type=flux_type,
    nprocx=nprocx, nprocy=nprocy,
)

for face in solver.faces.values():
    face.set_initial_condition(*initial_condition(face))

n_save = 100
if rank == 0:
    print('Initial dt:', dt)

if mode == 'run':
    daily_diagnostics = []
    initial_diagnostics = collect_daily_diagnostics(solver, 0)
    if rank == 0:
        daily_diagnostics.append(initial_diagnostics)

    t0 = time.time()
    for i in range(day):
        if rank == 0:
            print('Running day', i + 1)
        tend = solver.time + 3600 * 24
        #print('h min max:', min(f.h.min() for f in solver.faces.values()), max(f.h.max() for f in solver.faces.values()), solver.get_dt())
        is_nan = any(np.isnan(f.h).any() for f in solver.faces.values())
        if is_nan:
            print(f'NaN at day {i + 1}.')
            break
        while solver.time < tend:
            solver.time_step(dt=min(dt, tend - solver.time), order=34)

        day_diagnostics = collect_daily_diagnostics(solver, i + 1)
        if rank == 0:
            daily_diagnostics.append(day_diagnostics)

        if ((i + 1) % n_save == 0) or (i >= 720 and i < 1080):
            fn_template = get_fn_template(i + 1)
            if rank == 0:
                print('Saving:', fn_template)
            solver.save_restart(fn_template, data_dir)
            t1 = time.time()
            if rank == 0:
                print('Wall time for 1 day:', (t1 - t0) / n_save, '\n')
            t0 = time.time()
    # if rank == 0:
    #     print('Wall time for 1 day:', (t1 - t0) / 20, '\n')
    save_daily_diagnostics(daily_diagnostics)

if mode == 'plot':

    fn_template = get_fn_template(day)
    solver.store_diagnostics = True
    solver.load_restart(fn_template + '.npy', data_dir)
    plot_daily_diagnostics()

    lat = np.linspace(-90, 90, 4 * 512)[:, None]
    lon = np.linspace(-180, 180, 4 * 1024)[None, :]

    def _plot_func_helper(data, name, title, cmap, vmin=None, vmax=None):
        plt.figure(figsize=(10, 5), dpi=400)
        plt.title(title)
        plt.pcolormesh(lon.ravel(), lat.ravel(), data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.colorbar()

        plt.savefig(f'./{plot_dir}/{name}_{fn_template}.png')

    vort_plot = solver.evaluate_latlong(lat, lon, solver.vorticity(continuous=True), degrees=True)
    vort_plot -= 2 * 7.292e-5 * np.sin(lat * np.pi / 180)
    h_plot = solver.evaluate_latlong(
        lat, lon, 
        solver.continuous_projection(dict((name, face.h) for name, face in solver.faces.items())), 
        degrees=True
    )

    _plot_func_helper(vort_plot, 'vort', 'Relative vorticity', cmocean.cm.curl, vmin=-2.25e-5, vmax=2.25e-5)
    _plot_func_helper(h_plot, 'h', 'Height', cmocean.cm.deep)

    u_plot = solver.evaluate_latlong(
        lat, lon, 
        solver.continuous_projection(dict((name, face.u) for name, face in solver.faces.items())), 
        degrees=True
    )
    v_plot = solver.evaluate_latlong(
        lat, lon, 
        solver.continuous_projection(dict((name, face.v) for name, face in solver.faces.items())), 
        degrees=True
    )
    w_plot = solver.evaluate_latlong(
        lat, lon, 
        solver.continuous_projection(dict((name, face.w) for name, face in solver.faces.items())), 
        degrees=True
    )

    long_vec_x = np.cos(lon * np.pi / 180)
    long_vec_y = np.sin(lon * np.pi / 180)
    long_vec_z = 0 * lon

    lat_vec_x = -np.sin(lat * np.pi / 180) * np.sin(lon * np.pi / 180)
    lat_vec_y = np.sin(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
    lat_vec_z = np.cos(lat * np.pi / 180)

    zonal_vel = long_vec_x * u_plot + long_vec_y * v_plot + long_vec_z * w_plot
    meridional_vel = lat_vec_x * u_plot + lat_vec_y * v_plot + lat_vec_z * w_plot
    speed = np.sqrt(zonal_vel**2 + meridional_vel**2)

    _plot_func_helper(zonal_vel, 'zonal_vel', 'Zonal velocity', cmocean.cm.delta)
    _plot_func_helper(meridional_vel, 'meridional_vel', 'Meridional velocity', cmocean.cm.delta)
    _plot_func_helper(speed, 'speed', 'Speed', cmocean.cm.speed)

    print('Min h:', h_plot.min())
    print('Max vel:', speed.max())
    froude_number = speed / (np.sqrt(g * h_plot))
    print('Max froude number:', froude_number.max())
    print('Max wave speed:', (speed + np.sqrt(g * h_plot)).max())

    _plot_func_helper(froude_number, 'froude_number', 'Froude Number', cmocean.cm.speed, vmax=1.0)

    def _polar_plot_func_helper(data, name, title, cmap, vmin=None, vmax=None):
        plt.figure(figsize=(10, 5), dpi=400)
        plt.title(title)
        plt.tricontourf(solver.faces['zp'].xs.ravel(), solver.faces['zp'].ys.ravel(), data.ravel(), cmap=cmap, vmin=vmin, vmax=vmax, levels=100)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.colorbar()

        plt.savefig(f'./{plot_dir}/{name}_polar_{fn_template}.png')


    vort_plot = solver.vorticity(continuous=True)['zp']
    vort_plot -= solver.faces['zp'].f
    _polar_plot_func_helper(vort_plot, 'vort', 'Relative vorticity', cmocean.cm.curl, vmin=-2.25e-5, vmax=2.25e-5)

    speed = np.sqrt(solver.faces['zp'].u**2 + solver.faces['zp'].v**2 + solver.faces['zp'].w**2)
    froude_number = speed / np.sqrt(solver.faces['zp'].h * solver.faces['zp'].g)

    _polar_plot_func_helper(froude_number, 'froude_number', 'Froude Number', 'nipy_spectral', vmax=1.0)




if mode == 'restart':

    fn_template = get_fn_template(i_start)
    if rank == 0:
        print('Loading:', fn_template)
    solver.load_restart(fn_template + '.npy', data_dir)

    daily_diagnostics = []
    if rank == 0 and os.path.exists(daily_diagnostics_fp()):
        existing_diagnostics = load_daily_diagnostics()
        daily_diagnostics = existing_diagnostics[
            existing_diagnostics[:, 0] <= i_start
        ].tolist()

    restart_diagnostics = collect_daily_diagnostics(solver, i_start)
    if rank == 0 and (
        len(daily_diagnostics) == 0
        or daily_diagnostics[-1][0] != i_start
    ):
        daily_diagnostics.append(restart_diagnostics)

    for i in range(i_start, 3600):
        if rank == 0:
            print('\nRunning day', i + 1)
        tend = solver.time + 3600 * 24
        is_nan = any(np.isnan(f.h).any() for f in solver.faces.values())
        if is_nan:
            print(f'NaN at day {i+1}.')
            break
        while solver.time < tend:
            solver.time_step(dt=min(dt, tend - solver.time), order=34)

        day_diagnostics = collect_daily_diagnostics(solver, i + 1)
        if rank == 0:
            daily_diagnostics.append(day_diagnostics)

        if ((i + 1) % n_save == 0):
            fn_template = get_fn_template(i + 1)
            if rank == 0:
                print('Saving:', fn_template)
            solver.save_restart(fn_template, data_dir)
        # fn_template = f"reduced_williamson_5_day_{i + 1}.npy"
        # solver.save_restart(fn_template, 'data')

    save_daily_diagnostics(daily_diagnostics)

if mode == 'process-data':

    day = 0
    fn_template = get_fn_template(day)
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
        fn_template = get_fn_template(day)
        solver.load_restart(fn_template, 'data')

        vort = solver.vorticity()

        for name in solver.face_names:
            fp = solver.make_fp('vorticity', name, fn_template, 'data')
            print(fp)
            np.save(fp, vort[name] - solver.faces[name].f)


# h_plot_func=lambda s: s.h

# vort_plot_func = lambda s: s.vorticity() - s.f

# pv_plot_func = lambda s: (s.vorticity() - s.f) / (s.h)

# # exit(0)
# #days = np.array([1, 2, 3, 4]) * 360
# # days = list(days) + [1800,]
# days = [720,]
# for i, day in enumerate(days):
# # for i, day in enumerate([360, 400, 700]):
#     fn_template = get_fn_template(day)
#     solver.load_restart(fn_template, 'data')
#     print(min(f.h.min() for f in solver.faces.values()))
#     # plot_data(2 * i + 1, f'vort_day_{day}_nx{nx}_p{poly_order}', vort_plot_func, vmin=-3e-5, vmax=3e-5)
#     plot_data(2 * i + 1, f'vort_day_{day}_nx{nx}_p{poly_order}', pv_plot_func)
#     # plot_data(2 * i + 1, f'pv_day_{day}_nx{nx}_p{poly_order}', pv_plot_func)
#     plot_data(2 * i + 2, f'height_day_{day}_nx{nx}_p{poly_order}', h_plot_func)

#     # solver.load_restart(fn_template, 'data')
#     # plot_data(2 * i + 3, f'vort_day_{day}_nx{nx}_p{poly_order}_tangent_diss', vort_plot_func)
#     # # plot_data(2 * i + 1, f'pv_day_{day}_nx{nx}_p{poly_order}', pv_plot_func)
#     # plot_data(2 * i + 4, f'height_day_{day}_nx{nx}_p{poly_order}_tangent_diss', h_plot_func)



# plt.show()
