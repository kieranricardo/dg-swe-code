from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
import os
import time
from mpi4py import MPI
import cmocean
import argparse
from matplotlib.ticker import MaxNLocator

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
parser.add_argument('--flux_type', type=str, help='Flux type')
args = parser.parse_args()

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

days = [400, 1200, 2400, 3600]

solver = DGCubedSphereSWE(
    poly_order, nx, ny, g, f,
    eps=0.0, a=0.5, ah=ah, radius=radius,
    dtype=np.float64, flux_type=flux_type,
    nprocx=nprocx, nprocy=nprocy,
)

lat = np.linspace(-90, 90, 4 * 512)[:, None]
lon = np.linspace(-180, 180, 4 * 1024)[None, :]

h_plot_list = []
vort_plot_list = []
div_plot_list = []

for day in days:

    fn_template = get_fn_template(day)
    solver.load_restart(fn_template + '.npy', data_dir)

    vort_plot = solver.evaluate_latlong(lat, lon, solver.vorticity(continuous=True), degrees=True)
    vort_plot -= 2 * 7.292e-5 * np.sin(lat * np.pi / 180)

    h_plot = solver.evaluate_latlong(
        lat, lon, 
        solver.continuous_projection(dict((name, face.h) for name, face in solver.faces.items())), 
        degrees=True
    )

    div_plot = solver.evaluate_latlong(lat, lon, solver.divergence(continuous=True), degrees=True)

    vort_plot_list.append(vort_plot)
    h_plot_list.append(h_plot)
    div_plot_list.append(div_plot)


#### plot height
levs = np.linspace(100.0, 11000.0, 25)
fig, ax = plt.subplots(2, 2, figsize=(9, 5.7), layout="constrained")
mesh = ax[0, 0].pcolormesh(lon, lat, h_plot_list[0], 
                            cmap=cmocean.cm.deep, 
                            clim=[0.0, +10000.0])
cl = ax[0, 0].contour(lon.ravel(), lat.ravel(), h_plot_list[0], levs, colors=("k", ), 
                      linewidths=(0.5, ))
#cd = ax[0, 0].clabel(cl, fmt="%2.0f", colors="k", fontsize=8)
#ax[0, 0].set_xlabel(r"Lon [deg]", fontsize=8)
ax[0, 0].set_ylabel(r"Lat [deg]", fontsize=8)
#ax[0, 0].grid(True, which="both", ls=":", alpha=0.25)
ax[0, 0].tick_params(axis="both", labelsize=8)
ax[0, 0].yaxis.set_major_locator(MaxNLocator(5))

mesh = ax[0, 1].pcolormesh(lon.ravel(), lat.ravel(), h_plot_list[1], 
                            cmap=cmocean.cm.deep, 
                            clim=[0.0, +10000.0])
cl = ax[0, 1].contour(lon.ravel(), lat.ravel(), h_plot_list[1], levs, colors=("k", ), 
                      linewidths=(0.5, ))
ax[0, 1].tick_params(axis="both", labelsize=8)
ax[0, 1].yaxis.set_major_locator(MaxNLocator(5))

mesh = ax[1, 1].pcolormesh(lon, lat, h_plot_list[2], 
                            cmap=cmocean.cm.deep, 
                            clim=[0.0, +10000.0])
cl = ax[1, 1].contour(lon.ravel(), lat.ravel(), h_plot_list[2], levs, colors=("k", ), 
                      linewidths=(0.5, ))
#cd = ax[1, 1].clabel(cl, fmt="%2.0f", colors="k", fontsize=8)
ax[1, 1].set_xlabel(r"Lon [deg]", fontsize=8)
#ax[1, 1].set_ylabel(r"Lat [deg]", fontsize=8)
#ax[1, 1].grid(True, which="both", ls=":", alpha=0.25)
ax[1, 1].tick_params(axis="both", labelsize=8)
ax[1, 1].yaxis.set_major_locator(MaxNLocator(5))

mesh = ax[1, 0].pcolormesh(lon, lat, h_plot_list[3], 
                            cmap=cmocean.cm.deep, 
                            clim=[0.0, +10000.0])
cl = ax[1, 0].contour(lon.ravel(), lat.ravel(), h_plot_list[3], levs, colors=("k", ), 
                      linewidths=(0.5, ))
#cd = ax[1, 0].clabel(cl, fmt="%2.0f", colors="k", fontsize=8)
ax[1, 0].set_xlabel(r"Lon [deg]", fontsize=8)
ax[1, 0].set_ylabel(r"Lat [deg]", fontsize=8)
#ax[1, 0].grid(True, which="both", ls=":", alpha=0.25)
ax[1, 0].tick_params(axis="both", labelsize=8)
ax[1, 0].yaxis.set_major_locator(MaxNLocator(5))

cb = fig.colorbar(mesh, ax=ax[:, 0], pad=0.02, aspect=50, 
                  orientation="horizontal", location="bottom")
cb.ax.tick_params(labelsize=8)
cb.ax.xaxis.get_offset_text().set_fontsize(8)

plt.savefig(f'./{plot_dir}/height_{get_fn_template()}.png', dpi=300, bbox_inches="tight")

#-- rot(u)
fig, ax = plt.subplots(4, 2, figsize=(9, 11), layout="constrained")
mesh = ax[0, 0].pcolormesh(lon, lat, vort_plot_list[0], 
                            cmap=cmocean.cm.curl, 
                            clim=[-2.50E-05, +2.50E-05])
#ax[0, 0].set_xlabel(r"Lon [deg]", fontsize=8)
ax[0, 0].set_ylabel(r"Lat [deg]", fontsize=8)
#ax[0, 0].grid(True, which="both", ls=":", alpha=0.25)
ax[0, 0].tick_params(axis="both", labelsize=8)
ax[0, 0].yaxis.set_major_locator(MaxNLocator(5))

mesh = ax[1, 0].pcolormesh(lon, lat, vort_plot_list[1], 
                            cmap=cmocean.cm.curl, 
                            clim=[-2.50E-05, +2.50E-05])
#ax[1, 0].set_xlabel(r"Lon [deg]", fontsize=8)
ax[1, 0].set_ylabel(r"Lat [deg]", fontsize=8)
#ax[1, 0].grid(True, which="both", ls=":", alpha=0.25)
ax[1, 0].tick_params(axis="both", labelsize=8)
ax[1, 0].yaxis.set_major_locator(MaxNLocator(5))

mesh = ax[2, 0].pcolormesh(lon, lat, vort_plot_list[2], 
                            cmap=cmocean.cm.curl, 
                            clim=[-2.50E-05, +2.50E-05])
#ax[2, 0].set_xlabel(r"Lon [deg]", fontsize=8)
ax[2, 0].set_ylabel(r"Lat [deg]", fontsize=8)
#ax[2, 0].grid(True, which="both", ls=":", alpha=0.25)
ax[2, 0].tick_params(axis="both", labelsize=8)
ax[2, 0].yaxis.set_major_locator(MaxNLocator(5))

mesh = ax[3, 0].pcolormesh(lon, lat, vort_plot_list[3], 
                            cmap=cmocean.cm.curl, 
                            clim=[-2.50E-05, +2.50E-05])
ax[3, 0].set_xlabel(r"Lon [deg]", fontsize=8)
ax[3, 0].set_ylabel(r"Lat [deg]", fontsize=8)
#ax[3, 0].grid(True, which="both", ls=":", alpha=0.25)
ax[3, 0].tick_params(axis="both", labelsize=8)
ax[3, 0].yaxis.set_major_locator(MaxNLocator(5))

cb = fig.colorbar(mesh, ax=ax[:, 0], pad=0.01, aspect=50, 
                  orientation="horizontal", location="bottom")
cb.ax.tick_params(labelsize=8)
cb.ax.xaxis.get_offset_text().set_fontsize(8)

#-- div(u)
mesh = ax[0, 1].pcolormesh(lon, lat, div_plot_list[0], 
                            cmap=cmocean.cm.balance, 
                            clim=[-1.25E-06, +1.25E-06])
#ax[0, 1].set_xlabel(r"Lon [deg]", fontsize=8)
#ax[0, 1].set_ylabel(r"Lat [deg]", fontsize=8)
#ax[0, 1].grid(True, which="both", ls=":", alpha=0.25)
ax[0, 1].tick_params(axis="both", labelsize=8)
ax[0, 1].yaxis.set_major_locator(MaxNLocator(5))

mesh = ax[1, 1].pcolormesh(lon, lat, div_plot_list[1], 
                            cmap=cmocean.cm.balance, 
                            clim=[-1.25E-06, +1.25E-06])
#ax[1, 1].set_xlabel(r"Lon [deg]", fontsize=8)
#ax[1, 1].set_ylabel(r"Lat [deg]", fontsize=8)
#ax[1, 1].grid(True, which="both", ls=":", alpha=0.25)
ax[1, 1].tick_params(axis="both", labelsize=8)
ax[1, 1].yaxis.set_major_locator(MaxNLocator(5))

mesh = ax[2, 1].pcolormesh(lon, lat, div_plot_list[2], 
                            cmap=cmocean.cm.balance, 
                            clim=[-1.25E-06, +1.25E-06])
#ax[2, 1].set_xlabel(r"Lon [deg]", fontsize=8)
#ax[2, 1].set_ylabel(r"Lat [deg]", fontsize=8)
#ax[2, 1].grid(True, which="both", ls=":", alpha=0.25)
ax[2, 1].tick_params(axis="both", labelsize=8)
ax[2, 1].yaxis.set_major_locator(MaxNLocator(5))

mesh = ax[3, 1].pcolormesh(lon, lat, div_plot_list[3], 
                            cmap=cmocean.cm.balance, 
                            clim=[-1.25E-06, +1.25E-06])
ax[3, 1].set_xlabel(r"Lon [deg]", fontsize=8)
#ax[3, 1].set_ylabel(r"Lat [deg]", fontsize=8)
#ax[3, 1].grid(True, which="both", ls=":", alpha=0.25)
ax[3, 1].tick_params(axis="both", labelsize=8)
ax[3, 1].yaxis.set_major_locator(MaxNLocator(5))

cb = fig.colorbar(mesh, ax=ax[:, 1], pad=0.01, aspect=50, 
                  orientation="horizontal", location="bottom")
cb.ax.tick_params(labelsize=8)
cb.ax.xaxis.get_offset_text().set_fontsize(8)

plt.savefig(f'./{plot_dir}/rotdiv_{get_fn_template()}.png', dpi=300, bbox_inches="tight")
