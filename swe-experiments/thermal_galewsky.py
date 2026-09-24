from matplotlib import pyplot as plt
from dg_swe.dg_cubed_sphere_tswe import DGCubedSphereTSWE
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
parser.add_argument('--day', type=int, help='Plot day')
args = parser.parse_args()

if args.plot:
    mode = 'plot'
    day = args.day
    assert size == 1
else:
    mode = 'run'

nx = ny = args.nx
poly_order = args.order

cfl = 1.3
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

    b_pert = np.cos(lat) * np.exp(-(long / alpha) ** 2) * np.exp(-((lat_2 - lat) / beta) ** 2)
    hb = h * (g + b_pert)

    return u, v, w, h, hb


def get_fn_template(a, ah, upwind, tangent_diss, day=None):
    suffix = ''

    suffix = suffix + f'a_{a}'
    suffix = suffix + f'_ah_{ah}'
    if upwind:
        suffix = suffix + "_upwind"

    if tangent_diss:
        suffix = suffix + "_tangent_diss"

    if day is not None:
        suffix = suffix + f'_day_{day}'

    return f"thermal_galewsky_nx{nx}_p{poly_order}_{suffix}"


parameters_list = [
    dict(a=0.5, upwind=True, ah=0.0, tangent_diss=True),
]

if mode == 'run':

    for parameters in parameters_list:

        data_dir = os.path.join('data', get_fn_template(**parameters))
        plot_dir = os.path.join('plots', get_fn_template(**parameters))

        if rank == 0:
            if not os.path.exists(data_dir): os.makedirs(data_dir)
            if not os.path.exists(plot_dir): os.makedirs(plot_dir)

        solver = DGCubedSphereTSWE(
            poly_order, nx, ny, g, f,
            cfl, radius=radius,
            dtype=np.float64,
            nprocx=nprocx, nprocy=nprocy,
            a=parameters['a'],
            upwind=parameters['upwind'],
            ah=parameters['ah'],
            tangent_diss=parameters['tangent_diss']
        )

        for face in solver.faces.values():
            face.set_initial_condition(*initial_condition(face))
        solver.boundaries()

        dt = 130 * (15 / nx) * (cfl / 0.8)

        if rank == 0:
            print('Time step:', dt)
            print('Starting', get_fn_template(**parameters))
            print('a:', solver.faces['zp'].a, 'res:', nx, ny)
            print('ah:', solver.faces['zp'].ah, 'upwind:', solver.faces['zp'].upwind)
            print('flux type:', solver.faces['zp'].flux_type)

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
            fn_template = get_fn_template(day=i + 1, **parameters)
            solver.save_restart(fn_template, data_dir)

        solver.save_diagnostics(fn_template, data_dir)

elif mode == 'plot':

    solver = DGCubedSphereTSWE(
        poly_order, nx, ny, g, f,
        cfl, solution=None, a=0.5, radius=radius,
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

        h = solver.continuous_projection(dict((name, face.h) for name, face in solver.faces.items()))
        hb = solver.continuous_projection(dict((name, face.hb) for name, face in solver.faces.items()))
        b = dict((name, hb[name] / h[name]) for name in solver.faces.keys())

        u = solver.continuous_projection(dict((name, face.u) for name, face in solver.faces.items()))
        v = solver.continuous_projection(dict((name, face.v) for name, face in solver.faces.items()))
        w = solver.continuous_projection(dict((name, face.w) for name, face in solver.faces.items()))

        vort_plot = solver.evaluate_latlong(lat, lon, solver.vorticity(), degrees=True)
        vort_plot -= 2 * 7.292e-5 * np.sin(lat * np.pi / 180)
        h_plot = solver.evaluate_latlong(lat, lon, h, degrees=True)
        b_plot = solver.evaluate_latlong(lat, lon, b, degrees=True)

        _plot_func_helper(vort_plot, 'vort', 'Relative vorticity', cmocean.cm.curl, vmin=None, vmax=None)

        _plot_func_helper(h_plot, 'h', 'Height', cmocean.cm.deep)
        _plot_func_helper(b_plot, 'b', 'Buoyancy', cmocean.cm.thermal)

        u_plot = solver.evaluate_latlong(lat, lon, u, degrees=True)
        v_plot = solver.evaluate_latlong(lat, lon, v, degrees=True)
        w_plot = solver.evaluate_latlong(lat, lon, w, degrees=True)

        long_vec_x = np.cos(lon * np.pi / 180)
        long_vec_y = np.sin(lon * np.pi / 180)
        long_vec_z = 0 * lon

        lat_vec_x = -np.sin(lat * np.pi / 180) * np.sin(lon * np.pi / 180)
        lat_vec_y = np.sin(lat * np.pi / 180) * np.cos(lon * np.pi / 180)
        lat_vec_z = np.cos(lat * np.pi / 180)

        zonal_vel = long_vec_x * u_plot + long_vec_y * v_plot + long_vec_z * w_plot
        meridional_vel = lat_vec_x * u_plot + lat_vec_y * v_plot + lat_vec_z * w_plot
        speed = np.sqrt(zonal_vel ** 2 + meridional_vel ** 2)

        _plot_func_helper(zonal_vel, 'zonal_vel', 'Zonal velocity', cmocean.cm.delta)
        _plot_func_helper(meridional_vel, 'meridional_vel', 'Meridional velocity', cmocean.cm.delta)
        _plot_func_helper(speed, 'speed', 'Speed', cmocean.cm.speed)


        def _polar_plot_func_helper(xs, ys, data, name, title, cmap, vmin=None, vmax=None):

            plt.figure(figsize=(10, 5), dpi=400)
            plt.title(title)
            plt.tricontourf(xs, ys, data, cmap=cmap, vmin=vmin, vmax=vmax, levels=100)
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            plt.colorbar()

            plt.savefig(f'./{plot_dir}/{name}_polar_{fn_template}.png')


        xs = np.concatenate([face.xs.ravel() for face in solver.faces.values()])
        ys = np.concatenate([face.ys.ravel() for face in solver.faces.values()])
        zs = np.concatenate([face.zs.ravel() for face in solver.faces.values()])
        b_plot = np.concatenate([b[name].ravel() for name in solver.faces.keys()])

        mask = zs > 0
        _polar_plot_func_helper(xs[mask], ys[mask], b_plot[mask], 'b', 'Buoyancy', 'nipy_spectral')
