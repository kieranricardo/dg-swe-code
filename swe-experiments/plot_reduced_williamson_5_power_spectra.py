from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import linregress
from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
import os
import time
import cmocean
import argparse
from matplotlib.ticker import MaxNLocator
import pickle


plt.rcParams['font.size'] = '12'

nxs = [20, 40, 80, 160]
resolutions = [120, 60, 30, 15]

parser = argparse.ArgumentParser()
parser.add_argument('--order', type=int, help='Polynomial order')
parser.add_argument('--flux_type', type=str, help='Flux type')
args = parser.parse_args()

poly_order = args.order
flux_type = args.flux_type


def get_fn_template(nx, day=None):
    suffix = ''

    if flux_type == "lmars":
        suffix = suffix + 'lmars'
    elif flux_type == "barth":
        suffix = suffix + 'barth'
    elif flux_type == "barth_normal_tangent":
        suffix = suffix + 'barth_normal_tangent'
    elif flux_type == "old_tangent":
        suffix = suffix + 'old_tangent_diss'
    elif flux_type == "standard_tangent":
        suffix = suffix + 'tangent_diss'

    if day is None:
        return f"reduced_williamson_5_day_nx{nx}_p{poly_order}_{suffix}"
    else:
        return f"reduced_williamson_5_day_nx{nx}_p{poly_order}_{suffix}_{day}"


fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)


axes[0].set_title('Kinetic energy')
axes[0].set_ylabel('Power')

axes[1].set_title('Enstrophy')
axes[1].set_ylabel('Power')

ks = np.arange(100, 300)
axes[0].loglog(ks, 3e3 * (ks / ks[0])**(-3.0), '--', color='black', label='$n^{-3}$')
ks = np.arange(150, 500)
axes[1].loglog(ks, 0.5e-14 * (ks / ks[0])**(-1.0), '--', color='black', label='$n^{-1}$')

for nx, resolution in zip(nxs, resolutions):

    data_dir = os.path.join('data', get_fn_template(nx))
    spectra_fp = os.path.join(data_dir, f'spectra.pkl')

    try:
        with open(spectra_fp, 'rb') as file:
                spectra = pickle.load(file)
    except Exception:
        print(spectra_fp)
        break

    axes[0].loglog(spectra['ke'], label=f'{resolution} km')
    axes[1].loglog(spectra['enstrophy'], label=f'{resolution} km')


for ax in axes:
    ax.grid()
    ax.set_xlabel('Spherical wavenumber')
    ax.legend()

plt.savefig(f'./plots/reduced_williamson_5_spectra_{flux_type}.png')




