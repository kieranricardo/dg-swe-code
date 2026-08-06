from matplotlib import pyplot as plt
from dg_swe.dg_cubed_sphere_swe_numpy import DGCubedSphereSWENumpy
import numpy as np
import scipy
import os
import time
from mpi4py import MPI
import cmocean
import argparse


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

plt.rcParams['font.size'] = '12'


if size == 1:
    nprocx = nprocy = 1
else:
    nprocx = nprocy = int(np.sqrt(size // 6))


parser = argparse.ArgumentParser()
parser.add_argument('--order', type=int, help='Polynomial order')
parser.add_argument('--nx', type=int, help='Number of cells in horizontal')
parser.add_argument('--plot', action='store_true')
parser.add_argument('--day', type=int, help='Polynomial order')
args = parser.parse_args()


if args.plot:
    mode = 'plot'
    day = args.day
    assert size == 1
else:
    mode = 'run'    

nx = ny = args.nx + 1
poly_order = args.order

eps = 1.3
g = 9.80616
f = 7.292e-5
radius = 6.37122e6

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


def get_fn_template(a, flux_type, h_diss, day=None):
    suffix = ''

    if flux_type == "lmars":
        suffix = suffix + f'lmars'
    elif flux_type == "barth":
        suffix = suffix + f'barth'
    elif flux_type == "barth_normal_tangent":
        suffix = suffix + f'barth_normal_tangent'

    else:
        suffix = suffix + f'a_{a}'

        if flux_type == "old_tangent":
            suffix = suffix + '_old_tangent_diss'
        elif flux_type == "standard_tangent":
            suffix = suffix + '_tangent_diss'

        if h_diss:
            suffix = suffix + '_h_diss'

    if day is not None:
        suffix = suffix + f'_day_{day}'

    return f"galewsky_nx{nx-1}_p{poly_order}_{suffix}"


parameters_list = [
    dict(a=0.5, flux_type="barth_normal_tangent", h_diss=False),
]

if mode == 'run':

    for parameters in parameters_list:

        a = parameters['a']
        flux_type = parameters['flux_type']
        h_diss = parameters['h_diss']

        if parameters['h_diss']:
            ah = 0.5
        else:
            ah = 0.0

        data_dir = os.path.join('data', get_fn_template(**parameters))
        plot_dir = os.path.join('plots', get_fn_template(**parameters))

        if rank == 0:
            if not os.path.exists(data_dir): os.makedirs(data_dir)
            if not os.path.exists(plot_dir): os.makedirs(plot_dir)

        
        solver = DGCubedSphereSWENumpy(
            poly_order, nx, ny, g, f,
            eps, a=a, radius=radius,
            dtype=np.float64, flux_type=flux_type, ah=ah,
            nprocx=nprocx, nprocy=nprocy,
        )

        for face in solver.faces.values():
            face.set_initial_condition(*initial_condition(face))
        solver.boundaries()

        dt = 130 * (15 / nx) * (eps / 0.8)

        if rank == 0:
            print('Time step:', dt)
            print('Starting', get_fn_template(**parameters))
            print('a:', solver.faces['zp'].a, 'res:', nx, ny)
            print('ah:', solver.faces['zp'].ah, 'flux_type:', solver.faces['zp'].flux_type)

        for i in range(20):
            if rank == 0:
                print('Running day', i)
            tend = solver.time + 3600 * 24

            t0 = time.time()
            while solver.time < tend:
                solver.time_step(dt=min(dt, tend - solver.time), order=34)

            t1 = time.time()
            if rank == 0:
                print('Walltime:', t1 - t0, 's')

            comm.Barrier()
            fn_template = get_fn_template(day=i+1, **parameters)
            solver.save_restart(fn_template, data_dir)

        solver.save_diagnostics(fn_template, data_dir)

elif mode == 'plot':

    solver = DGCubedSphereSWENumpy(
        poly_order, nx, ny, g, f,
        eps, solution=None, a=0.5, radius=radius,
        dtype=np.float64, damping='adaptive'
    )

    lat = np.linspace(-90, 90, 4 * 512)[:, None]
    lon = np.linspace(-180, 180, 4 * 1024)[None, :]

    for parameters in parameters_list:

        data_dir = os.path.join('data', get_fn_template(**parameters))
        plot_dir = os.path.join('plots', get_fn_template(**parameters))
        if rank == 0:
            if not os.path.exists(data_dir): os.makedirs(data_dir)
            if not os.path.exists(plot_dir): os.makedirs(plot_dir)

        fn_template = get_fn_template(day=day, **parameters) 
        solver.load_restart(fn_template + '.npy', data_dir)

        def _plot_func_helper(data, name, title, cmap, vmin=None, vmax=None):
            plt.figure(figsize=(10, 5), dpi=400)
            plt.title(title)
            plt.pcolormesh(lon.ravel(), lat.ravel(), data, cmap=cmap, vmin=vmin, vmax=vmax)
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.colorbar()

            plt.savefig(f'./{plot_dir}/{name}_{fn_template}.png')

        rel_vort_siac = solver.siac_vorticity(
            include_coriolis=False, 
            boundary='sphere',
            quadrature_order=10,
            scale=0.75,
        )
        vort_plot_siac = solver.evaluate_latlong(lat, lon, rel_vort_siac, degrees=True)
        vort_plot = solver.evaluate_latlong(lat, lon, solver.vorticity(), degrees=True)
        vort_plot -= 2 * 7.292e-5 * np.sin(lat * np.pi / 180)
        h_plot = solver.evaluate_latlong(lat, lon, dict((name, face.h) for name, face in solver.faces.items()), degrees=True)

        _plot_func_helper(vort_plot_siac, 'siac_vort', 'Relative vorticity (SIAC)', cmocean.cm.curl, vmin=None, vmax=None)
        _plot_func_helper(vort_plot, 'vort', 'Relative vorticity', cmocean.cm.curl, vmin=None, vmax=None)
        
        _plot_func_helper(h_plot, 'h', 'Height', cmocean.cm.deep)

        u_plot = solver.evaluate_latlong(lat, lon, dict((name, face.u) for name, face in solver.faces.items()), degrees=True)
        v_plot = solver.evaluate_latlong(lat, lon, dict((name, face.v) for name, face in solver.faces.items()), degrees=True)
        w_plot = solver.evaluate_latlong(lat, lon, dict((name, face.w) for name, face in solver.faces.items()), degrees=True)

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
