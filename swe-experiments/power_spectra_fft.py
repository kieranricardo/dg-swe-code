from matplotlib import pyplot as plt
from dg_swe.dg_cubed_sphere_swe_numpy import DGCubedSphereSWENumpy
import numpy as np
import os
import pickle


plt.rcParams['font.size'] = '14'

use_siac = False
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

nx = ny = 112 + 1
max_n = 800
nlat = max(4 * max_n + 1, 2 * (ny - 1) * (poly_order + 1) + 1)
nlon = max(8 * max_n + 2, 4 * (nx - 1) * (poly_order + 1) + 2)


def get_fn_template(day=None):
    suffix = ''
    if tangent_diss:
        suffix = suffix + 'tangent_diss'

    if h_diss:
        suffix = suffix + '_h_diss'

    if s_0 == 4000:
        suffix = suffix + '_big_mountain'
    elif s_0 != 3000:
        raise ValueError(f'suffix: expedcted one of 3000, 4000. Found {s_0}.')

    if day is None:
        return f"reduced_williamson_5_day_nx{nx-1}_p{poly_order}_{suffix}"
    else:
        return f"reduced_williamson_5_day_nx{nx-1}_p{poly_order}_{suffix}_{day}"

# day = 16
# exp_name = f'DG_res_tang_diss_6x{nx}x{ny}'
# fn_template = f"{exp_name}_day_{day}.npy"
# out_name = 'galewsky'

fn_template = get_fn_template(1080)

data_dir = os.path.join('data', get_fn_template())
plot_dir = os.path.join('plots', get_fn_template())

if use_siac:
    out_fp = os.path.join(plot_dir, f'siac_spectra_{fn_template}.png')
else:
    out_fp = os.path.join(plot_dir, f'spectra_{fn_template}.png')


def make_latlon_grid(nlat, nlon):
    mu, mu_weights = np.polynomial.legendre.leggauss(nlat)
    colat = np.arccos(mu)
    lat = 0.5 * np.pi - colat
    lon = np.linspace(0.0, 2 * np.pi, nlon, endpoint=False)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing='ij')

    return lat_grid, lon_grid, colat, mu_weights


def evaluate_ke_latlon(solver, lat_grid, lon_grid):
    # Match power_spectra.py: form nodal KE first, then evaluate that DG field.
    if use_siac:

        h_siac = solver.siac_filter(
            dict((name, solver.faces[name].h) for name in solver.face_names),
            boundary='sphere',
            quadrature_order=10,
            scale=0.75
        )

        u_siac = solver.siac_filter(
            dict((name, solver.faces[name].u) for name in solver.face_names),
            boundary='sphere',
            quadrature_order=10,
            scale=0.75
        )

        v_siac = solver.siac_filter(
            dict((name, solver.faces[name].v) for name in solver.face_names),
            boundary='sphere',
            quadrature_order=10,
            scale=0.75
        )

        w_siac = solver.siac_filter(
            dict((name, solver.faces[name].w) for name in solver.face_names),
            boundary='sphere',
            quadrature_order=10,
            scale=0.75
        )

        ke_coeffs = dict(
            (name, 0.5 * h_siac[name] * (u_siac[name]**2 + v_siac[name]**2 + w_siac[name]**2))
            for name in solver.face_names
        )

    else:
        ke_coeffs = {
            name: 0.5 * face.h * (face.u ** 2 + face.v ** 2 + face.w ** 2)
            for name, face in solver.faces.items()
        }

    return solver.evaluate_latlong(lat_grid, lon_grid, ke_coeffs)


def evaluate_enstrophy_latlon(solver, lat_grid, lon_grid):

    if use_siac:
        rel_vort_siac = solver.siac_vorticity(
            include_coriolis=False, 
            boundary='sphere',
            quadrature_order=10,
            scale=0.75
        )

        h_siac = solver.siac_filter(
            dict((name, solver.faces[name].h) for name in solver.face_names),
            boundary='sphere',
            quadrature_order=10,
            scale=0.75
        )

        enstrophy_coeffs = dict(
            (name, rel_vort_siac[name]**2 / h_siac[name])
            for name in solver.face_names
        )

    else:
        vort = solver.vorticity()
        enstrophy_coeffs = {
            name: (vort[name] - face.f)**2 / face.h 
            for name, face in solver.faces.items()
        }

    return solver.evaluate_latlong(lat_grid, lon_grid, enstrophy_coeffs)


def normalized_associated_legendre(n, m, x):
    if m < 0 or m > n:
        raise ValueError(f"Need 0 <= m <= n. Found m={m}, n={n}.")

    x = np.asarray(x, dtype=np.float64)
    sin_colat = np.sqrt(np.maximum(0.0, 1.0 - x ** 2))

    pmm = np.full_like(x, 1 / np.sqrt(4 * np.pi), dtype=np.float64)
    for k in range(1, m + 1):
        pmm *= -np.sqrt((2 * k + 1) / (2 * k)) * sin_colat

    if n == m:
        return pmm

    p_prev = pmm
    p_curr = np.sqrt(2 * m + 3) * x * pmm
    if n == m + 1:
        return p_curr

    for ell in range(m + 2, n + 1):
        a = np.sqrt((2 * ell + 1) * (2 * ell - 1) / ((ell - m) * (ell + m)))
        b = np.sqrt(
            (2 * ell + 1) * (ell + m - 1) * (ell - m - 1)
            / ((2 * ell - 3) * (ell - m) * (ell + m))
        )
        p_next = a * x * p_curr - b * p_prev
        p_prev = p_curr
        p_curr = p_next

    return p_curr


def spherical_harmonic_latitude(m, n, colat):
    m_abs = abs(m)
    lat_basis = normalized_associated_legendre(n, m_abs, np.cos(colat))

    if m < 0 and m_abs % 2:
        lat_basis = -lat_basis

    return lat_basis


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

    x = np.cos(colat)
    sin_colat = np.sqrt(np.maximum(0.0, 1.0 - x ** 2))
    dlon = 2 * np.pi / nlon_

    if np.isrealobj(values):
        lon_fft = np.fft.rfft(values, axis=1) * dlon
        use_conjugate_negative_modes = True
    else:
        lon_fft = np.fft.fft(values, axis=1) * dlon
        use_conjugate_negative_modes = False

    coeffs = [[None] * (2 * n + 1) for n in range(max_n)]
    pmm = np.full(nlat_, 1 / np.sqrt(4 * np.pi), dtype=np.float64)

    for m_abs in range(max_n):
        lat_basis_by_n = np.empty((max_n - m_abs, nlat_), dtype=np.float64)
        lat_basis_by_n[0] = pmm

        if m_abs + 1 < max_n:
            lat_basis_by_n[1] = np.sqrt(2 * m_abs + 3) * x * pmm

            for i, ell in enumerate(range(m_abs + 2, max_n), start=2):
                a = np.sqrt((2 * ell + 1) * (2 * ell - 1) / ((ell - m_abs) * (ell + m_abs)))
                b = np.sqrt(
                    (2 * ell + 1) * (ell + m_abs - 1) * (ell - m_abs - 1)
                    / ((2 * ell - 3) * (ell - m_abs) * (ell + m_abs))
                )
                lat_basis_by_n[i] = a * x * lat_basis_by_n[i - 1] - b * lat_basis_by_n[i - 2]

        if m_abs == 0:
            modes = (lon_fft[:, 0] * mu_weights)[:, None]
            integrals = lat_basis_by_n @ modes
            for n, coeff in enumerate(integrals[:, 0]):
                coeffs[n][n] = coeff
        else:
            positive_mode = lon_fft[:, m_abs]
            if use_conjugate_negative_modes:
                negative_mode = np.conj(positive_mode)
            else:
                negative_mode = lon_fft[:, -m_abs]

            modes = np.column_stack((negative_mode * mu_weights, positive_mode * mu_weights))
            integrals = lat_basis_by_n @ modes

            if m_abs % 2:
                integrals[:, 0] *= -1

            for i, n in enumerate(range(m_abs, max_n)):
                coeffs[n][n - m_abs] = integrals[i, 0]
                coeffs[n][n + m_abs] = integrals[i, 1]

        if m_abs + 1 < max_n:
            pmm *= -np.sqrt((2 * m_abs + 3) / (2 * m_abs + 2)) * sin_colat

    return coeffs


def spectrum_from_coeffs(coeffs):
    return np.array([np.sqrt(sum(abs(x) ** 2 for x in coeffs_n)) for coeffs_n in coeffs])


def add_reference_line(ax, spectrum, idx, exponent, label):
    if len(spectrum) > idx:
        ks = np.arange(idx, len(spectrum))
        line = ks ** exponent
        line *= spectrum[idx] / line[0]
        ax.loglog(ks, line, linestyle='dotted', label=label)


def plot_spectra(ke_spectrum, enstrophy_spectrum, fn_template):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].semilogy(ke_spectrum)
    # add_reference_line(axes[0], ke_spectrum, 30, -5/3, r'$n^{-5/3}$')
    add_reference_line(axes[0], ke_spectrum, 75, -3.0, r'$n^{-3}$')
    axes[0].set_title('Kinetic energy')
    axes[0].set_ylabel('Power')

    axes[1].semilogy(enstrophy_spectrum)
    add_reference_line(axes[1], enstrophy_spectrum, 100, -1.0, r'$n^{-1}$')
    axes[1].set_title('Enstrophy')
    axes[1].set_ylabel('Power')

    for ax in axes:
        ax.grid()
        ax.set_xlabel('Spherical wavenumber')
        ax.legend()

    fig.savefig(out_fp)
    # plt.show()


def main():

    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    solver = DGCubedSphereSWENumpy(
        poly_order, nx, ny, g, f,
        eps, a=0.5, ah=ah, radius=radius,
        dtype=np.float64, tangent_diss=tangent_diss
    )

    if use_siac:
        coeffs_fp = os.path.join(data_dir, f'siac_coeffs_{fn_template}.pkl')
    else:
        coeffs_fp = os.path.join(data_dir, f'coeffs_{fn_template}.pkl')
    if compute:
        print('Loading', fn_template)
        solver.load_restart(fn_template + '.npy', data_dir)

        print(f'Making {nlat} x {nlon} lat/lon grid')
        lat_grid, lon_grid, colat, mu_weights = make_latlon_grid(nlat, nlon)

        print('Evaluating KE on lat/lon grid')
        ke = evaluate_ke_latlon(solver, lat_grid, lon_grid)

        print('Computing KE spherical harmonic coefficients with longitude FFT')
        ke_coeffs = spherical_harmonic_coefficients_fft(ke, colat, mu_weights, max_n)
        del ke

        print('Evaluating enstrophy on lat/lon grid')
        enstrophy = evaluate_enstrophy_latlon(solver, lat_grid, lon_grid)

        print('Computing enstrophy spherical harmonic coefficients with longitude FFT')
        enstrophy_coeffs = spherical_harmonic_coefficients_fft(enstrophy, colat, mu_weights, max_n)
        del enstrophy

        with open(coeffs_fp, 'wb') as file:
            pickle.dump({'ke': ke_coeffs, 'enstrophy': enstrophy_coeffs}, file, pickle.HIGHEST_PROTOCOL)

    with open(coeffs_fp, 'rb') as file:
        coeffs = pickle.load(file)

    ke_spectrum = spectrum_from_coeffs(coeffs['ke'])
    enstrophy_spectrum = spectrum_from_coeffs(coeffs['enstrophy'])
    plot_spectra(ke_spectrum, enstrophy_spectrum, fn_template)


if __name__ == '__main__':
    main()
