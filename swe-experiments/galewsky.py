from matplotlib import pyplot as plt
from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
from dg_swe.utils import Interpolate
import numpy as np
import scipy
import os

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

mode = 'run'
dev = 'cpu'

nx = ny = 15
eps = 0.8
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

if mode == 'run':

    exp_names = [f'DG_res_6x{nx}x{ny}', f'DG_cntr_res_6x{nx}x{ny}']

    for exp in exp_names:
        if 'cntr' in exp:
            a = 0.0
        else:
            a = 0.5
        solver = DGCubedSphereSWE(
            poly_order, nx, ny, g, f,
            eps, device=dev, solution=None, a=a, radius=radius,
            dtype=np.float64
        )
        for face in solver.faces.values():
            face.set_initial_condition(*initial_condition(face))

        solver.boundaries()
        print('Time step:', solver.get_dt())
        print('Starting', exp)
        print('a:', solver.faces['zp'].a, 'res:', nx, ny)

        for i in range(20):
            print('Running day', i)
            tend = solver.time + 3600 * 24
            while solver.time < tend:
                dt = solver.get_dt()
                dt = min(dt, tend - solver.time)
                solver.time_step(dt=dt)

            fn_template = f"{exp}_day_{i+1}.npy"
            solver.save_restart(fn_template, 'data')

        solver.save_diagnostics(fn_template, 'data')

elif mode == 'plot':

    solver = DGCubedSphereSWE(
        poly_order, nx, ny, g, f,
        eps, device=dev, solution=None, a=0.5, radius=radius,
        dtype=np.float64, damping='adaptive'
    )
    for face in solver.faces.values():
        face.set_initial_condition(*initial_condition(face))
    E0 = solver.integrate(solver.entropy())

    p = 12
    solver_hr = DGCubedSphereSWE(
        p, nx, ny, g, f,
        eps, device=dev, solution=None, a=0.5, radius=radius,
        dtype=np.float64, damping='adaptive'
    )

    interpolator = Interpolate(3, p)

    exp_names = [f'DG_res_6x{nx}x{ny}', f'DG_cntr_res_6x{nx}x{ny}']
    labels = ['Diss.', 'Cons.']

    for exp, label in zip(exp_names, labels):
        fn_template = f"{exp}_day_{20}.npy"
        solver.plot_diagnostics(fn_template, 'data', 1, label)

    plt.savefig(f'./plots/galewsky_conservation.png')

    vmin = -0.00015; vmax = 0.00015
    plot_func = lambda s: s.vorticity() - s.f

    def interpolate_plot_func(s):
        data = plot_func(solver.faces[s.name])
        return interpolator.torch_interpolate(data)

    day = 7

    for exp in exp_names:

        fn_template = f"{exp}_day_{day}.npy"
        solver.load_restart(fn_template, 'data')
        E = solver.integrate(solver.entropy())
        print(f'{exp} relative energy loss rate:', (E - E0) / (E0 * day * 24 * 3600))
        print(f'{exp} adjusted energy loss rate (Wm^-2):', 3e9 * (E - E0) / (E0 * day * 24 * 3600))

        for name, face in solver.faces.items():
            data = [face.u, face.v, face.w, face.h]
            data = [interpolator.torch_interpolate(tnsr).numpy() for tnsr in data]
            solver_hr.faces[name].set_initial_condition(*data)

        solver_hr.boundaries()

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_xlabel("x (km)")
        ax.set_ylabel("y (km)")

        im = solver_hr.triangular_plot(ax, vmin=vmin, vmax=vmax, latlong=False, plot_func=interpolate_plot_func)
        plt.colorbar(im[0])
        plt.savefig(f'./plots/vort_galewsky_{exp}_{int(day)}_days.png')

    plt.show()
