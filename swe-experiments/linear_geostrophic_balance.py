from matplotlib import pyplot as plt
from dg_swe.linear_cubed_sphere_swe import LinearCubedSphereSWE
import numpy as np
import torch
import os

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

mode = 'plot'
dev = 'cpu'
xlim = np.pi
ylim = np.pi

nx = ny = 5
eps = 0.2
H = 0.2
g = 8
f = 8
poly_order = 3

def initial_condition(face):
    lat, long = face.geometry.lat_long(face.xs, face.ys, face.zs)
    stream = 0.1 * np.sin(long) * np.sin(lat)

    stream = torch.from_numpy(stream)
    u, v, w = face.curl_k(stream)
    h = stream * face.f / g
    return u.numpy(), v.numpy(), w.numpy(), h.numpy()



solver = LinearCubedSphereSWE(
    poly_order, nx, ny, g, f,
    eps, H=H, device=dev, solution=None, a=0.5, radius=1,
    dtype=np.float64, damping=None,
)

solver0 = LinearCubedSphereSWE(
    poly_order, nx, ny, g, f,
    eps, H=H, device=dev, solution=None, a=0.5, radius=1,
    dtype=np.float64, damping=None
)

for face in solver.faces.values():
    face.set_initial_condition(*initial_condition(face))

for face in solver0.faces.values():
    face.set_initial_condition(*initial_condition(face))

h_norm = np.sqrt(sum(face.integrate(face.h ** 2) for face in solver.faces.values()))
u_norm = np.sqrt(sum(face.integrate(face.u ** 2 + face.v ** 2) for face in solver.faces.values()))

tend = 2
h_errors = []
vel_errors = []
times = []

for _ in range(1000):

    h_error = sum(f1.integrate((f1.h - f2.h) ** 2) for f1, f2 in zip(solver.faces.values(), solver0.faces.values()))
    h_error = np.sqrt(h_error) / h_norm

    u_error = sum(f1.integrate((f1.u - f2.u) ** 2 + (f1.v - f2.v) ** 2) for f1, f2 in zip(solver.faces.values(), solver0.faces.values()))
    u_error = np.sqrt(u_error) / u_norm

    h_errors.append(h_error)
    vel_errors.append(u_error)
    times.append(solver.time)
    solver.time_step()


plt.figure(1)

plt.ylabel("Relative L2 error")
plt.xlabel("Time")

plt.semilogy(times, h_errors, label='D')
plt.semilogy(times, vel_errors, label='u')
plt.legend()
plt.grid()
plt.savefig('./plots/linear_geostrophic_balance_error.png')
plt.show()