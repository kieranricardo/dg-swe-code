from matplotlib import pyplot as plt
from dg_swe.dg_cubed_sphere_linear_swe import DGCubedSphereLinearSWE
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter as MovieWriter
import torch
import os

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

mode = 'plot'
dev = 'cpu'
xlim = np.pi
ylim = np.pi

nx = ny = 10
eps = 1.0
H = 8
g = 8
f = 8
poly_order = 3

def initial_condition(face):
    lat, long = face.geometry.lat_long(face.xs, face.ys, face.zs)
    stream = 0.1 * np.sin(long) * np.sin(lat)

    stream = torch.from_numpy(stream)
    u, v, w = face.curl_k(0 * stream)
    h = 1.2 * stream * face.f / g
    return u.numpy(), v.numpy(), w.numpy(), h.numpy()


# from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
# solver = DGCubedSphereSWE(
#     poly_order, nx, ny, g, f,
#     eps, device=dev, solution=None, a=0.5, radius=1,
#     dtype=np.float64, damping=None
# )
#
# for face in solver.faces.values():
#     face.set_initial_condition(*initial_condition(face))
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.set_xlabel("x (km)")
# ax.set_ylabel("y (km)")
#
# vmin = min(f.h.min() for f in solver.faces.values())
# vmax = max(f.h.max() for f in solver.faces.values())
# im = solver.triangular_plot(ax, latlong=False, vmin=vmin, vmax=vmax, plot_func=lambda s: s.h, n=20)
# plt.colorbar(im[0])
# plt.savefig('./plots/geostrophic_balance_ic.png')

solver = DGCubedSphereLinearSWE(
    poly_order, nx, ny, g, f,
    eps, H=H, device=dev, solution=None, a=0.5, radius=1,
    dtype=np.float64, damping=None,
)



for face in solver.faces.values():
    face.set_initial_condition(*initial_condition(face))


# while solver.time < 2.0:
#     solver.time_step()
#
# exit(0)

currtime = 0.0
def update_plot(frame_number, plot, solver, ax):
    global currtime
    if solver.time > currtime + 1.0:
        print("Time:", solver.time, 's.')
        currtime = solver.time

    h_max = max(f.h_plot().max() for f in solver.faces.values())
    h_min = min(f.h_plot().min() for f in solver.faces.values())
    diff = h_max - h_min
    avg = 0.5 * (h_max + h_min)
    vmin = h_min
    vmax = h_max

    for _ in range(iplot):
        #
        solver.time_step()

    for aa in plot[0]:
        aa.remove()
    plot[0] = solver.plot_solution(ax, vmin=vmin, vmax=vmax, cmap='viridis', plot_func=plot_func, dim=3)


plot_func = lambda s: s.h_plot() #/ s.h #s.vorticity - s.f

h_max = max(f.h_plot().max() for f in solver.faces.values())
h_min = min(f.h_plot().min() for f in solver.faces.values())
diff = h_max - h_min
avg = 0.5 * (h_max + h_min)
vmin = h_min
vmax = h_max

solver.time_step()

mode = "movie"
if mode == "plot":
    iplot = 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")


    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plot = [solver.plot_solution(ax, cmap='viridis', plot_func=plot_func, dim=3, vmin=vmin, vmax=vmax)]
    ani = animation.FuncAnimation(
        fig, update_plot, 1, fargs=(plot, solver, ax), interval=10
    )

    plt.show()
    print("Simulation time (unit less):", solver.time)

elif mode == "movie":
    iplot = 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plot = [solver.plot_solution(ax, cmap='viridis', plot_func=plot_func, dim=3, vmin=vmin, vmax=vmax)]

    moviewriter = MovieWriter(fps=30)

    with moviewriter.saving(fig, f"./plots/linear_sw.mp4", dpi=100):

        while solver.time <= 1:

            update_plot(0, plot, solver, ax)
            moviewriter.grab_frame()
