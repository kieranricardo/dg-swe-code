from matplotlib import pyplot as plt
from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
import numpy as np
import os
import pickle

try:
    from scipy.special import sph_harm_y

    def spherical_harmonic(m, n, lon, colat):
        return sph_harm_y(n, m, colat, lon)
except ImportError:
    from scipy.special import sph_harm

    def spherical_harmonic(m, n, lon, colat):
        return sph_harm(m, n, lon, colat)


plt.rcParams['font.size'] = '14'

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

nx = ny = 64
max_n = 100
nlat = max(4 * max_n + 1, 2 * (ny - 1) * (poly_order + 1) + 1)
nlon = max(8 * max_n + 2, 4 * (nx - 1) * (poly_order + 1) + 2)
day = 16
exp_name = f'DG_res_6x{nx}x{ny}'
data_dir = os.environ.get(
    'SWE_DATA_DIR',
    '/Users/u5824685/Documents/repos/dg-tswe-paper/tswe-experiments/data'
)

def get_fn_template(day):
    suffix = ''
    if tangent_diss:
        suffix = suffix + 'tangent_diss'

    if h_diss:
        suffix = suffix + '_h_diss'

    if s_0 == 4000:
        suffix = suffix + '_big_mountain'
    elif s_0 != 3000:
        raise ValueError(f'suffix: expected one of 3000, 4000. Found {s_0}.')

    return f"reduced_williamson_5_day_{day}_nx{nx}_p{poly_order}_{suffix}.npy"


def make_latlon_grid(nlat, nlon):
    mu, mu_weights = np.polynomial.legendre.leggauss(nlat)
    colat = np.arccos(mu)
    lat = 0.5 * np.pi - colat
    lon = np.linspace(0.0, 2 * np.pi, nlon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

    return lat_grid, lon_grid, colat, mu_weights


def evaluate_ke_latlon(solver, lat_grid, lon_grid):
    # Match power_spectra.py: form nodal KE first, then evaluate that DG field.
    ke_coeffs = {
        name: 0.5 * face.h * (face.u ** 2 + face.v ** 2 + face.w ** 2)
        for name, face in solver.faces.items()
    }

    return solver.evaluate_latlong(lat_grid, lon_grid, ke_coeffs)


def spherical_harmonic_coefficients_fft(values, colat, mu_weights, max_n):
    if values.ndim != 2:
        raise ValueError(f"values: expected a 2D lat/lon array. Found shape {values.shape}.")

    nlat_, nlon_ = values.shape
    if colat.shape != (nlat_,):
        raise ValueError(f"colat: expected shape {(nlat_,)}. Found {colat.shape}.")
    if mu_weights.shape != (nlat_,):
        raise ValueError(f"mu_weights: expected shape {(nlat_,)}. Found {mu_weights.shape}.")
    if nlon_ < 2 * max_n - 1:
        raise ValueError(f"nlon: need at least {2 * max_n - 1} longitudes for max_n={max_n}. Found {nlon_}.")

    dlon = 2 * np.pi / nlon_
    lon_fft = np.fft.fft(values, axis=1) * dlon

    coeffs = []
    lon0 = 0.0
    for n in range(max_n):
        coeffs_n = []
        for m in range(-n, n + 1):
            lon_integral = lon_fft[:, m % nlon_]
            lat_basis = np.conjugate(spherical_harmonic(m, n, lon0, colat))
            coeffs_n.append(np.sum(lon_integral * lat_basis * mu_weights))
        coeffs.append(coeffs_n)

    return coeffs


def plot_spectrum(spectrum, fn_template):
    fig, ax = plt.subplots()
    print('Spectrum:', spectrum[-10:])
    ax.semilogy(spectrum)

    if len(spectrum) > 15:
        ks = np.arange(15, len(spectrum))
        line = ks ** (-5 / 3)
        line *= spectrum[15] / line[0]
        ax.loglog(ks, line, linestyle='dotted', label=r'$-5/3$')

    ax.grid()
    ax.set_xlabel('Spherical wavenumber')
    ax.set_ylabel('Power')
    ax.legend()

    fp = os.path.join('plots', f'{fn_template}_power_spectra_fft_n{max_n}_nlat{nlat}_nlon{nlon}.png')
    # fig.savefig(fp)
    plt.show()


def main():

    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    if not os.path.exists('./data'):
        os.makedirs('./data')

    solver = DGCubedSphereSWE(
        poly_order, nx, ny, g, f,
        eps, a=0.5, ah=ah, radius=radius,
        dtype=np.float64, tangent_diss=tangent_diss
    )

    fn_template = f"{exp_name}_day_{day}.npy"
    fp = os.path.join('data', f'{fn_template}_power_spectra_fft_n{max_n}_nlat{nlat}_nlon{nlon}.pkl')

    if compute:
        print('Loading', fn_template)
        solver.load_restart(fn_template, data_dir)

        print(f'Evaluating KE on {nlat} x {nlon} lat/lon grid')
        lat_grid, lon_grid, colat, mu_weights = make_latlon_grid(nlat, nlon)
        ke = evaluate_ke_latlon(solver, lat_grid, lon_grid)

        print('Computing spherical harmonic coefficients with longitude FFT')
        coeffs = spherical_harmonic_coefficients_fft(ke, colat, mu_weights, max_n)

        with open(fp, 'wb') as file:
            pickle.dump(coeffs, file, pickle.HIGHEST_PROTOCOL)

    with open(fp, 'rb') as file:
        coeffs = pickle.load(file)

    spectrum = np.array([np.sqrt(sum(abs(x) ** 2 for x in coeffs_n)) for coeffs_n in coeffs])
    plot_spectrum(spectrum, fn_template)


if __name__ == '__main__':
    main()
