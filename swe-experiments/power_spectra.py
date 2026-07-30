from matplotlib import pyplot as plt
from dg_tswe.utils import Interpolate
from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
from scipy.special import sph_harm
import numpy as np
import scipy
import os
import pickle

plt.rcParams['font.size'] = '14'

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

compute = True
eps = 0.8
g = 9.80616 / 250
f = 7.292e-5
radius = 6.37122e6
u_0 = 0.5
h_0 = 5960.0
s_0 = 3000
poly_order = 3

tangent_diss = True
h_diss = True

if h_diss:
    ah = 0.5
else:
    ah = 0.0

def get_fn_template(day):
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

nx = ny = 64
solver = DGCubedSphereSWE(
    poly_order, nx, ny, g, f,
    eps, a=0.5, ah=ah, radius=radius,
    dtype=np.float64, tangent_diss=tangent_diss
)


fn_template = get_fn_template(720)
print('Loading', fn_template)
solver.load_restart(fn_template, 'data')

lat = {n: f.geometry.lat_long(f.xs, f.ys, f.zs)[0] + 0.5 * np.pi for n, f in solver.faces.items()}
lon = {n: f.geometry.lat_long(f.xs, f.ys, f.zs)[1] for n, f in solver.faces.items()}

weights = {n: (f.weights * abs(f.J)).numpy() / radius**2 for n, f in solver.faces.items()}

KE = {n: (0.5 * f.h * (f.u**2 + f.v**2 + f.w**2)).numpy() for n, f in solver.faces.items()}

max_n = 100
fp = os.path.join('data', f'{fn_template}_power_spectra_n{max_n}.pkl')

if compute:
    coeffs = []

    for n1 in range(max_n):
        coeffs_ = []
        for m1 in range(-n1, n1+1):
            coeff = sum(
                (KE[n] * np.conjugate(sph_harm(m1, n1, lon[n], lat[n])) * weights[n]).sum()
                for n, f in solver.faces.items()
            )
            coeffs_.append(coeff)

        coeffs.append(coeffs_)

    # coeffs = np.array(coeffs)
    with open(fp, 'wb') as f:
        pickle.dump(coeffs, f, pickle.HIGHEST_PROTOCOL)

with open(fp, 'rb') as f:
    coeffs = pickle.load(f)


spectrum = np.array([np.sqrt(sum(abs(x)**2 for x in coeffs_)) for coeffs_ in coeffs])
ks = np.arange(15, len(spectrum))
# coeffs = np.load(fp)

line = ks**(-5 / 3)
line *= spectrum[15] / line[0]

plt.semilogy(spectrum)
plt.loglog(ks, line, linestyle='dotted', label=r'$-5/3$')
plt.grid()
plt.xlabel('Spherical wavenumber')
plt.ylabel('Power')
plt.legend()
fp = os.path.join('plots', f'{fn_template}_power_spectra_n{max_n}.png')
plt.savefig(fp)
plt.show()

