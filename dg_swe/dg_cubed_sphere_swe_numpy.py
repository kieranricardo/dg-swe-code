import os
from math import comb, factorial

import numpy as np

from dg_swe.geometry import EquiangularFace, SadournyFace, face_name_from_cartesian, lat_long_to_cartesian

try:
    from numba import njit
except ImportError:
    njit = None


def _to_numpy(arr, dtype=None, copy=False):
    if hasattr(arr, "detach"):
        arr = arr.detach().cpu().numpy()
    out = np.asarray(arr)
    if dtype is not None and out.dtype != np.dtype(dtype):
        return out.astype(dtype, copy=copy)
    if copy:
        return out.copy()
    return out


def _norm_l2(vec):
    return np.sqrt(sum(a * a for a in vec))


VALID_FLUX_TYPES = (
    "standard",
    "standard_tangent",
    "old_tangent",
    "lmars",
    "barth",
    "barth_normal_tangent",
)


def _validate_flux_type(flux_type):
    if flux_type not in VALID_FLUX_TYPES:
        raise ValueError(
            f"flux_type: expected one of {VALID_FLUX_TYPES}. Found {flux_type!r}."
        )
    return flux_type


def _reject_legacy_flux_kwargs(kwargs):
    legacy_keys = ("tangent_diss", "old_tangent_diss", "lmars", "barth_diss")
    found = [key for key in legacy_keys if key in kwargs]
    if found:
        keys = ", ".join(found)
        raise TypeError(f"Legacy flux switch keyword(s) removed: {keys}. Use flux_type instead.")


def _positive_part_power(x, degree):
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0.0
    if degree == 0:
        out[mask] = 1.0
    else:
        out[mask] = x[mask] ** degree
    return out


def centered_cardinal_bspline(x, order, derivative=0):
    """
    Centered cardinal B-spline of the requested order.

    ``order`` is degree + 1, so order 1 is the unit box on
    [-1/2, 1/2] and order 2 is the centered tent function.
    """
    if order < 1:
        raise ValueError(f"order must be positive; found {order}.")
    if derivative < 0:
        raise ValueError(f"derivative must be non-negative; found {derivative}.")

    x_arr = np.asarray(x, dtype=float)
    scalar = x_arr.ndim == 0
    x_flat = x_arr.reshape(1) if scalar else x_arr

    if derivative >= order:
        out = np.zeros_like(x_flat, dtype=float)
    elif derivative > 0:
        out = np.zeros_like(x_flat, dtype=float)
        for j in range(derivative + 1):
            out += (
                (-1) ** j
                * comb(derivative, j)
                * centered_cardinal_bspline(
                    x_flat + 0.5 * derivative - j, order - derivative
                )
            )
    else:
        degree = order - 1
        out = np.zeros_like(x_flat, dtype=float)
        for j in range(order + 1):
            out += (
                (-1) ** j
                * comb(order, j)
                * _positive_part_power(x_flat + 0.5 * order - j, degree)
            )
        out /= factorial(degree)

    if scalar:
        return out[0]
    return out.reshape(x_arr.shape)


def _centered_uniform_sum_moments(order, max_degree):
    moments = np.zeros(max_degree + 1, dtype=float)
    moments[0] = 1.0

    uniform = np.zeros(max_degree + 1, dtype=float)
    for degree in range(max_degree + 1):
        if degree % 2 == 0:
            uniform[degree] = 1.0 / ((degree + 1) * 2.0 ** degree)

    for _ in range(order):
        next_moments = np.zeros_like(moments)
        for degree in range(max_degree + 1):
            total = 0.0
            for j in range(degree + 1):
                total += comb(degree, j) * moments[j] * uniform[degree - j]
            next_moments[degree] = total
        moments = next_moments

    return moments


def _shifted_bspline_moments(order, shift, max_degree):
    centered_moments = _centered_uniform_sum_moments(order, max_degree)
    moments = np.zeros(max_degree + 1, dtype=float)
    for degree in range(max_degree + 1):
        total = 0.0
        for j in range(degree + 1):
            total += comb(degree, j) * shift ** (degree - j) * centered_moments[j]
        moments[degree] = total
    return moments


class SIACKernel:
    def __init__(
        self,
        poly_order=None,
        *,
        spline_order=None,
        num_shifts=None,
        reproduction_degree=None,
        shifts=None,
        coeffs=None,
    ):
        if spline_order is None:
            if poly_order is None:
                raise ValueError("poly_order or spline_order must be provided.")
            spline_order = poly_order + 1

        if shifts is None:
            if num_shifts is None:
                if poly_order is None:
                    raise ValueError("poly_order, num_shifts, or shifts must be provided.")
                num_shifts = poly_order
            shifts = np.arange(-num_shifts, num_shifts + 1, dtype=float)
        else:
            shifts = np.asarray(shifts, dtype=float)

        if reproduction_degree is None:
            reproduction_degree = shifts.size - 1

        if reproduction_degree >= shifts.size:
            raise ValueError(
                "reproduction_degree must be less than the number of SIAC shifts; "
                f"found reproduction_degree={reproduction_degree}, shifts={shifts.size}."
            )

        self.spline_order = int(spline_order)
        self.reproduction_degree = int(reproduction_degree)
        self.shifts = shifts

        if coeffs is None:
            coeffs = self._solve_coeffs()
        self.coeffs = np.asarray(coeffs, dtype=float)
        if self.coeffs.shape != self.shifts.shape:
            raise ValueError(
                f"coeffs: expected shape {self.shifts.shape}. Found {self.coeffs.shape}."
            )

        self._breakpoints = None

    def _solve_coeffs(self):
        matrix = np.zeros((self.reproduction_degree + 1, self.shifts.size), dtype=float)
        for col, shift in enumerate(self.shifts):
            matrix[:, col] = _shifted_bspline_moments(
                self.spline_order, shift, self.reproduction_degree
            )

        rhs = np.zeros(self.reproduction_degree + 1, dtype=float)
        rhs[0] = 1.0

        if matrix.shape[0] == matrix.shape[1]:
            return np.linalg.solve(matrix, rhs)
        return np.linalg.lstsq(matrix, rhs, rcond=None)[0]

    def values(self, x, derivative=0):
        x_arr = np.asarray(x, dtype=float)
        out = np.zeros_like(x_arr, dtype=float)
        for coeff, shift in zip(self.coeffs, self.shifts):
            out += coeff * centered_cardinal_bspline(
                x_arr - shift, self.spline_order, derivative=derivative
            )
        return out

    @property
    def breakpoints(self):
        if self._breakpoints is None:
            knots = []
            for shift in self.shifts:
                left = shift - 0.5 * self.spline_order
                knots.extend(left + np.arange(self.spline_order + 1))
            self._breakpoints = np.unique(np.round(knots, decimals=14))
        return self._breakpoints

    def quadrature(self, quadrature_order, derivative=0):
        if quadrature_order < 1:
            raise ValueError(f"quadrature_order must be positive; found {quadrature_order}.")

        points_1d, weights_1d = np.polynomial.legendre.leggauss(quadrature_order)
        points = []
        weights = []
        for left, right in zip(self.breakpoints[:-1], self.breakpoints[1:]):
            if right <= left:
                continue
            midpoint = 0.5 * (left + right)
            half_width = 0.5 * (right - left)
            interval_points = midpoint + half_width * points_1d
            interval_weights = half_width * weights_1d * self.values(
                interval_points, derivative=derivative
            )
            points.append(interval_points)
            weights.append(interval_weights)

        return np.concatenate(points), np.concatenate(weights)

    def cache_key(self):
        return (
            self.spline_order,
            self.reproduction_degree,
            tuple(np.round(self.shifts, decimals=14)),
            tuple(np.round(self.coeffs, decimals=14)),
        )


def siac_kernel(poly_order, **kwargs):
    return SIACKernel(poly_order=poly_order, **kwargs)


def gll(N, iterative=True):
    return gLLNodesAndWeights(N + 1)


def lagrange(N, i, x, xi):
    fac = 1
    for j in range(-1, N):
        if j != i:
            fac = fac * ((x - xi[j + 1]) / (xi[i + 1] - xi[j + 1]))
    return fac


def lagrange_basis_values(x, xi):
    x = np.asarray(x)
    xi = np.asarray(xi)

    x_flat = x.reshape(-1)
    basis = np.ones((x_flat.size, xi.size), dtype=np.result_type(x, xi, float))

    for i in range(xi.size):
        for j in range(xi.size):
            if i != j:
                basis[:, i] *= (x_flat - xi[j]) / (xi[i] - xi[j])

    return basis.reshape(x.shape + (xi.size,))


def lagrange1st(N, xi):
    out = np.zeros([N + 1, N + 1])
    d = np.zeros([N + 1, N + 1])

    for i in range(-1, N):
        for j in range(-1, N):
            if i != j:
                d[i + 1, j + 1] = legendre(N, xi[i + 1]) / legendre(N, xi[j + 1]) / (xi[i + 1] - xi[j + 1])
            if i == -1 and j == -1:
                d[i + 1, j + 1] = -0.25 * N * (N + 1)
            if i == N - 1 and j == N - 1:
                d[i + 1, j + 1] = 0.25 * N * (N + 1)

    for n in range(-1, N):
        for i in range(-1, N):
            total = 0
            for j in range(-1, N):
                total += d[i + 1, j + 1] * lagrange(N, n, xi[j + 1], xi)
            out[n + 1, i + 1] = total
    return out


def legendre(N, x):
    P = np.zeros(2 * N)

    if N == 0:
        P[0] = 1
    elif N == 1:
        P[1] = x
    else:
        P[0] = 1
        P[1] = x
    for i in range(2, N + 1):
        P[i] = ((2 * i - 1) * x * P[i - 1] - (i - 1) * P[i - 2]) / float(i)

    return P[N]


def gLLNodesAndWeights(n, epsilon=1e-15):
    if n < 2:
        raise ValueError("n must be larger than 1")

    x = np.empty(n)
    w = np.empty(n)

    x[0] = -1
    x[n - 1] = 1
    w[0] = 2.0 / (n * (n - 1))
    w[n - 1] = w[0]

    n_2 = n // 2
    for i in range(1, n_2):
        xi = (1 - (3 * (n - 2)) / (8 * (n - 1) ** 3)) * np.cos((4 * i + 1) * np.pi / (4 * (n - 1) + 1))
        error = 1.0

        while error > epsilon:
            y = dLgP(n - 1, xi)
            y1 = d2LgP(n - 1, xi)
            y2 = d3LgP(n - 1, xi)
            dx = 2 * y * y1 / (2 * y1 ** 2 - y * y2)
            xi -= dx
            error = abs(dx)

        x[i] = -xi
        x[n - i - 1] = xi

        w[i] = 2 / (n * (n - 1) * lgP(n - 1, x[i]) ** 2)
        w[n - i - 1] = w[i]

    if n % 2 != 0:
        x[n_2] = 0
        w[n_2] = 2.0 / ((n * (n - 1)) * lgP(n - 1, np.array(x[n_2])) ** 2)

    return x, w


def dLgP(n, xi):
    return n * (lgP(n - 1, xi) - xi * lgP(n, xi)) / (1 - xi ** 2)


def d2LgP(n, xi):
    return (2 * xi * dLgP(n, xi) - n * (n + 1) * lgP(n, xi)) / (1 - xi ** 2)


def d3LgP(n, xi):
    return (4 * xi * d2LgP(n, xi) - (n * (n + 1) - 2) * dLgP(n, xi)) / (1 - xi ** 2)


def lgP(n, xi):
    xi = np.asarray(xi)
    scalar = xi.ndim == 0
    xi_arr = xi.reshape(1) if scalar else xi
    if n == 0:
        out = np.ones(xi_arr.size)
        return out[0] if scalar else out
    if n == 1:
        return xi_arr[0] if scalar else xi_arr

    fP = np.ones(xi_arr.size)
    sP = xi_arr.copy()
    nP = np.empty(xi_arr.size)
    for i in range(2, n + 1):
        nP = ((2 * i - 1) * xi_arr * sP - (i - 1) * fP) / i
        fP = sP
        sP = nP
    return nP[0] if scalar else nP


def cross_product(vec1, vec2):
    return [
        vec1[1] * vec2[2] - vec1[2] * vec2[1],
        vec1[2] * vec2[0] - vec1[0] * vec2[2],
        vec1[0] * vec2[1] - vec1[1] * vec2[0],
    ]


if njit is not None:
    @njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
    def _solve_numba_kernel(
        u, v, w, h, b,
        D, endpoint_weight, J,
        vert_upper_edge_factor, vert_lower_edge_factor,
        horz_right_edge_factor, horz_left_edge_factor,
        dxidx, dxidy, dxidz, detadx, detady, detadz,
        dxidx_up, dxidy_up, dxidz_up, dxidx_down, dxidy_down, dxidz_down,
        dxidx_right, dxidy_right, dxidz_right, dxidx_left, dxidy_left, dxidz_left,
        detadx_up, detady_up, detadz_up, detadx_down, detady_down, detadz_down,
        detadx_right, detady_right, detadz_right, detadx_left, detady_left, detadz_left,
        dxdxi, dydxi, dzdxi, dxdeta, dydeta, dzdeta,
        f,
        u_up, v_up, w_up, h_up, u_down, v_down, w_down, h_down,
        u_right, v_right, w_right, h_right, u_left, v_left, w_left, h_left,
        eta_x_up, eta_y_up, eta_z_up, eta_x_down, eta_y_down, eta_z_down,
        xi_x_right, xi_y_right, xi_z_right, xi_x_left, xi_y_left, xi_z_left,
        dxdxi_up, dydxi_up, dzdxi_up, dxdxi_down, dydxi_down, dzdxi_down,
        dxdxi_right, dydxi_right, dzdxi_right, dxdxi_left, dydxi_left, dzdxi_left,
        dxdeta_up, dydeta_up, dzdeta_up, dxdeta_down, dydeta_down, dzdeta_down,
        dxdeta_right, dydeta_right, dzdeta_right, dxdeta_left, dydeta_left, dzdeta_left,
        g, a, ah, tangent_diss,
    ):
        ny, nx, n, _ = u.shape

        h_xcontra_J = np.empty_like(u)
        h_ycontra_J = np.empty_like(u)
        uv_flux = np.empty_like(u)
        u_contra = np.empty_like(u)
        v_contra = np.empty_like(u)
        u_cov = np.empty_like(u)
        v_cov = np.empty_like(u)

        h_k = np.empty_like(u)
        u_k = np.empty_like(u)
        v_k = np.empty_like(u)
        w_k = np.empty_like(u)

        inv_endpoint_weight = 1.0 / endpoint_weight
        tangent_scale = 0.5 if tangent_diss else 0.0

        for ey in range(ny):
            for ex in range(nx):
                for eta in range(n):
                    for xi in range(n):
                        uu = u[ey, ex, eta, xi]
                        vv = v[ey, ex, eta, xi]
                        ww = w[ey, ex, eta, xi]
                        hh = h[ey, ex, eta, xi]
                        ux = uu * dxidx[ey, ex, eta, xi] + vv * dxidy[ey, ex, eta, xi] + ww * dxidz[ey, ex, eta, xi]
                        uy = uu * detadx[ey, ex, eta, xi] + vv * detady[ey, ex, eta, xi] + ww * detadz[ey, ex, eta, xi]
                        j_val = J[ey, ex, eta, xi]
                        u_contra[ey, ex, eta, xi] = ux
                        v_contra[ey, ex, eta, xi] = uy
                        hx = hh * ux
                        hy = hh * uy
                        h_xcontra_J[ey, ex, eta, xi] = hx * j_val
                        h_ycontra_J[ey, ex, eta, xi] = hy * j_val
                        uv_flux[ey, ex, eta, xi] = 0.5 * (uu * uu + vv * vv + ww * ww) + g * (hh + b[ey, ex, eta, xi])
                        u_cov[ey, ex, eta, xi] = uu * dxdxi[ey, ex, eta, xi] + vv * dydxi[ey, ex, eta, xi] + ww * dzdxi[ey, ex, eta, xi]
                        v_cov[ey, ex, eta, xi] = uu * dxdeta[ey, ex, eta, xi] + vv * dydeta[ey, ex, eta, xi] + ww * dzdeta[ey, ex, eta, xi]

        for ey in range(ny):
            for ex in range(nx):
                for eta in range(n):
                    for xi in range(n):
                        ddxi_h = 0.0
                        ddeta_h = 0.0
                        ddxi_uv = 0.0
                        ddeta_uv = 0.0
                        ddxi_vcov = 0.0
                        ddeta_ucov = 0.0
                        for l in range(n):
                            d_xi = D[l, xi]
                            d_eta = D[l, eta]
                            ddxi_h += h_xcontra_J[ey, ex, eta, l] * d_xi
                            ddeta_h += d_eta * h_ycontra_J[ey, ex, l, xi]
                            ddxi_uv += uv_flux[ey, ex, eta, l] * d_xi
                            ddeta_uv += d_eta * uv_flux[ey, ex, l, xi]
                            ddxi_vcov += v_cov[ey, ex, eta, l] * d_xi
                            ddeta_ucov += d_eta * u_cov[ey, ex, l, xi]
                        j_val = J[ey, ex, eta, xi]
                        h_k[ey, ex, eta, xi] = -(ddxi_h + ddeta_h) / j_val
                        abs_vort_cov = ddxi_vcov - ddeta_ucov + f[ey, ex, eta, xi] * j_val
                        u_k[ey, ex, eta, xi] = -ddxi_uv + v_contra[ey, ex, eta, xi] * abs_vort_cov
                        v_k[ey, ex, eta, xi] = -ddeta_uv - u_contra[ey, ex, eta, xi] * abs_vort_cov

        for ey in range(ny + 1):
            for ex in range(nx):
                for xi in range(n):
                    uu_up = u_up[ey, ex, xi]
                    vv_up = v_up[ey, ex, xi]
                    ww_up = w_up[ey, ex, xi]
                    hh_up = h_up[ey, ex, xi]
                    uu_down = u_down[ey, ex, xi]
                    vv_down = v_down[ey, ex, xi]
                    ww_down = w_down[ey, ex, xi]
                    hh_down = h_down[ey, ex, xi]

                    h_up_flux_val = hh_up * (
                        uu_up * eta_x_up[ey, ex, xi]
                        + vv_up * eta_y_up[ey, ex, xi]
                        + ww_up * eta_z_up[ey, ex, xi]
                    )
                    h_down_flux_val = hh_down * (
                        uu_down * eta_x_down[ey, ex, xi]
                        + vv_down * eta_y_down[ey, ex, xi]
                        + ww_down * eta_z_down[ey, ex, xi]
                    )
                    uv_up_flux_val = 0.5 * (
                        uu_up * uu_up + vv_up * vv_up + ww_up * ww_up
                    ) + g * hh_up
                    uv_down_flux_val = 0.5 * (
                        uu_down * uu_down + vv_down * vv_down + ww_down * ww_down
                    ) + g * hh_down

                    h_sum = hh_up + hh_down
                    inv_h_sum = 1.0 / h_sum
                    c_snd = 0.5 * (np.sqrt(g * hh_up) + np.sqrt(g * hh_down))
                    c_adv = (h_up_flux_val + h_down_flux_val) * inv_h_sum
                    abs_c_adv = abs(c_adv)

                    h_flux_val = 0.5 * (h_up_flux_val + h_down_flux_val) - ah * abs_c_adv * (hh_up - hh_down)
                    uv_flux_val = 0.5 * (uv_up_flux_val + uv_down_flux_val) - a * (
                        c_snd + abs_c_adv
                    ) * (h_up_flux_val - h_down_flux_val) * (2.0 * inv_h_sum)

                    u_cov_up_val = (
                        uu_up * dxdxi_up[ey, ex, xi]
                        + vv_up * dydxi_up[ey, ex, xi]
                        + ww_up * dzdxi_up[ey, ex, xi]
                    )
                    u_cov_down_val = (
                        uu_down * dxdxi_down[ey, ex, xi]
                        + vv_down * dydxi_down[ey, ex, xi]
                        + ww_down * dzdxi_down[ey, ex, xi]
                    )
                    u_contra_up_val = (
                        uu_up * dxidx_up[ey, ex, xi]
                        + vv_up * dxidy_up[ey, ex, xi]
                        + ww_up * dxidz_up[ey, ex, xi]
                    )
                    u_contra_down_val = (
                        uu_down * dxidx_down[ey, ex, xi]
                        + vv_down * dxidy_down[ey, ex, xi]
                        + ww_down * dxidz_down[ey, ex, xi]
                    )
                    v_contra_up_val = (
                        uu_up * detadx_up[ey, ex, xi]
                        + vv_up * detady_up[ey, ex, xi]
                        + ww_up * detadz_up[ey, ex, xi]
                    )
                    v_contra_down_val = (
                        uu_down * detadx_down[ey, ex, xi]
                        + vv_down * detady_down[ey, ex, xi]
                        + ww_down * detadz_down[ey, ex, xi]
                    )

                    adv_sign = 1.0 - 2.0 * (c_adv < 0.0)
                    avg_tan_cov = 0.5 * (
                        u_cov_up_val + u_cov_down_val
                    ) - tangent_scale * adv_sign * (
                        u_cov_up_val - u_cov_down_val
                    )

                    u_flux_up_val = v_contra_up_val * avg_tan_cov
                    u_flux_down_val = v_contra_down_val * avg_tan_cov
                    v_flux_up_val = uv_flux_val - u_contra_up_val * avg_tan_cov
                    v_flux_down_val = uv_flux_val - u_contra_down_val * avg_tan_cov

                    if ey > 0:
                        cell_y = ey - 1
                        h_k[cell_y, ex, n - 1, xi] -= (
                            h_flux_val - h_down_flux_val
                        ) * vert_upper_edge_factor[cell_y, ex, xi]
                        u_k[cell_y, ex, n - 1, xi] -= (
                            u_flux_down_val - v_contra_down_val * u_cov_down_val
                        ) * inv_endpoint_weight
                        v_k[cell_y, ex, n - 1, xi] -= (
                            v_flux_down_val
                            - (uv_down_flux_val - u_contra_down_val * u_cov_down_val)
                        ) * inv_endpoint_weight

                    if ey < ny:
                        h_k[ey, ex, 0, xi] += (
                            h_flux_val - h_up_flux_val
                        ) * vert_lower_edge_factor[ey, ex, xi]
                        u_k[ey, ex, 0, xi] += (
                            u_flux_up_val - v_contra_up_val * u_cov_up_val
                        ) * inv_endpoint_weight
                        v_k[ey, ex, 0, xi] += (
                            v_flux_up_val
                            - (uv_up_flux_val - u_contra_up_val * u_cov_up_val)
                        ) * inv_endpoint_weight

        for ey in range(ny):
            for ex in range(nx + 1):
                for eta in range(n):
                    uu_right = u_right[ey, ex, eta]
                    vv_right = v_right[ey, ex, eta]
                    ww_right = w_right[ey, ex, eta]
                    hh_right = h_right[ey, ex, eta]
                    uu_left = u_left[ey, ex, eta]
                    vv_left = v_left[ey, ex, eta]
                    ww_left = w_left[ey, ex, eta]
                    hh_left = h_left[ey, ex, eta]

                    h_right_flux_val = hh_right * (
                        uu_right * xi_x_right[ey, ex, eta]
                        + vv_right * xi_y_right[ey, ex, eta]
                        + ww_right * xi_z_right[ey, ex, eta]
                    )
                    h_left_flux_val = hh_left * (
                        uu_left * xi_x_left[ey, ex, eta]
                        + vv_left * xi_y_left[ey, ex, eta]
                        + ww_left * xi_z_left[ey, ex, eta]
                    )
                    uv_right_flux_val = 0.5 * (
                        uu_right * uu_right + vv_right * vv_right + ww_right * ww_right
                    ) + g * hh_right
                    uv_left_flux_val = 0.5 * (
                        uu_left * uu_left + vv_left * vv_left + ww_left * ww_left
                    ) + g * hh_left

                    h_sum = hh_right + hh_left
                    inv_h_sum = 1.0 / h_sum
                    c_snd = 0.5 * (np.sqrt(g * hh_right) + np.sqrt(g * hh_left))
                    c_adv = (h_right_flux_val + h_left_flux_val) * inv_h_sum
                    abs_c_adv = abs(c_adv)

                    h_flux_val = 0.5 * (h_right_flux_val + h_left_flux_val) - ah * abs_c_adv * (hh_right - hh_left)
                    uv_flux_val = 0.5 * (uv_right_flux_val + uv_left_flux_val) - a * (
                        c_snd + abs_c_adv
                    ) * (h_right_flux_val - h_left_flux_val) * (2.0 * inv_h_sum)

                    v_cov_right_val = (
                        uu_right * dxdeta_right[ey, ex, eta]
                        + vv_right * dydeta_right[ey, ex, eta]
                        + ww_right * dzdeta_right[ey, ex, eta]
                    )
                    v_cov_left_val = (
                        uu_left * dxdeta_left[ey, ex, eta]
                        + vv_left * dydeta_left[ey, ex, eta]
                        + ww_left * dzdeta_left[ey, ex, eta]
                    )
                    u_contra_right_val = (
                        uu_right * dxidx_right[ey, ex, eta]
                        + vv_right * dxidy_right[ey, ex, eta]
                        + ww_right * dxidz_right[ey, ex, eta]
                    )
                    u_contra_left_val = (
                        uu_left * dxidx_left[ey, ex, eta]
                        + vv_left * dxidy_left[ey, ex, eta]
                        + ww_left * dxidz_left[ey, ex, eta]
                    )
                    v_contra_right_val = (
                        uu_right * detadx_right[ey, ex, eta]
                        + vv_right * detady_right[ey, ex, eta]
                        + ww_right * detadz_right[ey, ex, eta]
                    )
                    v_contra_left_val = (
                        uu_left * detadx_left[ey, ex, eta]
                        + vv_left * detady_left[ey, ex, eta]
                        + ww_left * detadz_left[ey, ex, eta]
                    )

                    adv_sign = 1.0 - 2.0 * (c_adv < 0.0)
                    avg_tan_cov = 0.5 * (
                        v_cov_right_val + v_cov_left_val
                    ) - tangent_scale * adv_sign * (
                        v_cov_right_val - v_cov_left_val
                    )

                    u_flux_right_val = uv_flux_val - v_contra_right_val * avg_tan_cov
                    u_flux_left_val = uv_flux_val - v_contra_left_val * avg_tan_cov
                    v_flux_right_val = u_contra_right_val * avg_tan_cov
                    v_flux_left_val = u_contra_left_val * avg_tan_cov

                    if ex > 0:
                        cell_x = ex - 1
                        h_k[ey, cell_x, eta, n - 1] -= (
                            h_flux_val - h_left_flux_val
                        ) * horz_right_edge_factor[ey, cell_x, eta]
                        u_k[ey, cell_x, eta, n - 1] -= (
                            u_flux_left_val
                            - (uv_left_flux_val - v_contra_left_val * v_cov_left_val)
                        ) * inv_endpoint_weight
                        v_k[ey, cell_x, eta, n - 1] -= (
                            v_flux_left_val - u_contra_left_val * v_cov_left_val
                        ) * inv_endpoint_weight

                    if ex < nx:
                        h_k[ey, ex, eta, 0] += (
                            h_flux_val - h_right_flux_val
                        ) * horz_left_edge_factor[ey, ex, eta]
                        u_k[ey, ex, eta, 0] += (
                            u_flux_right_val
                            - (
                                uv_right_flux_val
                                - v_contra_right_val * v_cov_right_val
                            )
                        ) * inv_endpoint_weight
                        v_k[ey, ex, eta, 0] += (
                            v_flux_right_val - u_contra_right_val * v_cov_right_val
                        ) * inv_endpoint_weight

        for ey in range(ny):
            for ex in range(nx):
                for eta in range(n):
                    for xi in range(n):
                        uk_cov = u_k[ey, ex, eta, xi]
                        vk_cov = v_k[ey, ex, eta, xi]
                        u_k[ey, ex, eta, xi] = uk_cov * dxidx[ey, ex, eta, xi] + vk_cov * detadx[ey, ex, eta, xi]
                        v_k[ey, ex, eta, xi] = uk_cov * dxidy[ey, ex, eta, xi] + vk_cov * detady[ey, ex, eta, xi]
                        w_k[ey, ex, eta, xi] = uk_cov * dxidz[ey, ex, eta, xi] + vk_cov * detadz[ey, ex, eta, xi]

        return u_k, v_k, w_k, h_k

    @njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
    def _solve_numba_lmars_kernel(
        u, v, w, h, b,
        D, endpoint_weight, J,
        vert_upper_edge_factor, vert_lower_edge_factor,
        horz_right_edge_factor, horz_left_edge_factor,
        dxidx, dxidy, dxidz, detadx, detady, detadz,
        dxidx_up, dxidy_up, dxidz_up, dxidx_down, dxidy_down, dxidz_down,
        dxidx_right, dxidy_right, dxidz_right, dxidx_left, dxidy_left, dxidz_left,
        detadx_up, detady_up, detadz_up, detadx_down, detady_down, detadz_down,
        detadx_right, detady_right, detadz_right, detadx_left, detady_left, detadz_left,
        dxdxi, dydxi, dzdxi, dxdeta, dydeta, dzdeta,
        f,
        u_up, v_up, w_up, h_up, u_down, v_down, w_down, h_down,
        u_right, v_right, w_right, h_right, u_left, v_left, w_left, h_left,
        eta_x_up, eta_y_up, eta_z_up, eta_x_down, eta_y_down, eta_z_down,
        xi_x_right, xi_y_right, xi_z_right, xi_x_left, xi_y_left, xi_z_left,
        dxdxi_up, dydxi_up, dzdxi_up, dxdxi_down, dydxi_down, dzdxi_down,
        dxdxi_right, dydxi_right, dzdxi_right, dxdxi_left, dydxi_left, dzdxi_left,
        dxdeta_up, dydeta_up, dzdeta_up, dxdeta_down, dydeta_down, dzdeta_down,
        dxdeta_right, dydeta_right, dzdeta_right, dxdeta_left, dydeta_left, dzdeta_left,
        g, a, ah, tangent_diss,
    ):
        ny, nx, n, _ = u.shape

        h_xcontra_J = np.empty_like(u)
        h_ycontra_J = np.empty_like(u)
        uv_flux = np.empty_like(u)
        u_contra = np.empty_like(u)
        v_contra = np.empty_like(u)
        u_cov = np.empty_like(u)
        v_cov = np.empty_like(u)

        h_k = np.empty_like(u)
        u_k = np.empty_like(u)
        v_k = np.empty_like(u)
        w_k = np.empty_like(u)

        inv_endpoint_weight = 1.0 / endpoint_weight
        tangent_scale = 0.5
        a = 0.5
        ah = 0.5

        for ey in range(ny):
            for ex in range(nx):
                for eta in range(n):
                    for xi in range(n):
                        uu = u[ey, ex, eta, xi]
                        vv = v[ey, ex, eta, xi]
                        ww = w[ey, ex, eta, xi]
                        hh = h[ey, ex, eta, xi]
                        ux = uu * dxidx[ey, ex, eta, xi] + vv * dxidy[ey, ex, eta, xi] + ww * dxidz[ey, ex, eta, xi]
                        uy = uu * detadx[ey, ex, eta, xi] + vv * detady[ey, ex, eta, xi] + ww * detadz[ey, ex, eta, xi]
                        j_val = J[ey, ex, eta, xi]
                        u_contra[ey, ex, eta, xi] = ux
                        v_contra[ey, ex, eta, xi] = uy
                        hx = hh * ux
                        hy = hh * uy
                        h_xcontra_J[ey, ex, eta, xi] = hx * j_val
                        h_ycontra_J[ey, ex, eta, xi] = hy * j_val
                        uv_flux[ey, ex, eta, xi] = 0.5 * (uu * uu + vv * vv + ww * ww) + g * (hh + b[ey, ex, eta, xi])
                        u_cov[ey, ex, eta, xi] = uu * dxdxi[ey, ex, eta, xi] + vv * dydxi[ey, ex, eta, xi] + ww * dzdxi[ey, ex, eta, xi]
                        v_cov[ey, ex, eta, xi] = uu * dxdeta[ey, ex, eta, xi] + vv * dydeta[ey, ex, eta, xi] + ww * dzdeta[ey, ex, eta, xi]

        for ey in range(ny):
            for ex in range(nx):
                for eta in range(n):
                    for xi in range(n):
                        ddxi_h = 0.0
                        ddeta_h = 0.0
                        ddxi_uv = 0.0
                        ddeta_uv = 0.0
                        ddxi_vcov = 0.0
                        ddeta_ucov = 0.0
                        for l in range(n):
                            d_xi = D[l, xi]
                            d_eta = D[l, eta]
                            ddxi_h += h_xcontra_J[ey, ex, eta, l] * d_xi
                            ddeta_h += d_eta * h_ycontra_J[ey, ex, l, xi]
                            ddxi_uv += uv_flux[ey, ex, eta, l] * d_xi
                            ddeta_uv += d_eta * uv_flux[ey, ex, l, xi]
                            ddxi_vcov += v_cov[ey, ex, eta, l] * d_xi
                            ddeta_ucov += d_eta * u_cov[ey, ex, l, xi]
                        j_val = J[ey, ex, eta, xi]
                        h_k[ey, ex, eta, xi] = -(ddxi_h + ddeta_h) / j_val
                        abs_vort_cov = ddxi_vcov - ddeta_ucov + f[ey, ex, eta, xi] * j_val
                        u_k[ey, ex, eta, xi] = -ddxi_uv + v_contra[ey, ex, eta, xi] * abs_vort_cov
                        v_k[ey, ex, eta, xi] = -ddeta_uv - u_contra[ey, ex, eta, xi] * abs_vort_cov

        for ey in range(ny + 1):
            for ex in range(nx):
                for xi in range(n):
                    uu_up = u_up[ey, ex, xi]
                    vv_up = v_up[ey, ex, xi]
                    ww_up = w_up[ey, ex, xi]
                    hh_up = h_up[ey, ex, xi]
                    uu_down = u_down[ey, ex, xi]
                    vv_down = v_down[ey, ex, xi]
                    ww_down = w_down[ey, ex, xi]
                    hh_down = h_down[ey, ex, xi]

                    vel_up = (
                        uu_up * eta_x_up[ey, ex, xi]
                        + vv_up * eta_y_up[ey, ex, xi]
                        + ww_up * eta_z_up[ey, ex, xi]
                    )
                    vel_down = (
                        uu_down * eta_x_down[ey, ex, xi]
                        + vv_down * eta_y_down[ey, ex, xi]
                        + ww_down * eta_z_down[ey, ex, xi]
                    )

                    h_up_flux_val = hh_up * vel_up
                    h_down_flux_val = hh_down * vel_down
                    uv_up_flux_val = 0.5 * (
                        uu_up * uu_up + vv_up * vv_up + ww_up * ww_up
                    ) + g * hh_up
                    uv_down_flux_val = 0.5 * (
                        uu_down * uu_down + vv_down * vv_down + ww_down * ww_down
                    ) + g * hh_down

                    h_sum = hh_up + hh_down
                    inv_h_sum = 1.0 / h_sum
                    c_snd_down = np.sqrt(g * hh_down)
                    c_snd_up = np.sqrt(g * hh_up)
                    c_snd = 0.5 * (c_snd_down + c_snd_up) - 0.25 * (vel_up - vel_down)
                    h_star = c_snd**2 / g
                    c_adv = 0.5 * (vel_up + vel_down) + c_snd_down - c_snd_up
                    abs_c_adv = abs(c_adv)

                    # h_flux_val = 0.5 * c_adv * h_sum - ah * abs_c_adv * (hh_up - hh_down)
                    # uv_flux_val = 0.5 * (uv_up_flux_val + uv_down_flux_val) - a * (
                    #     c_snd
                    # ) * (h_up_flux_val - h_down_flux_val) * (2.0 * inv_h_sum)
                    h_flux_val = c_adv * h_star
                    uv_flux_val = 0.5 * (uv_up_flux_val + uv_down_flux_val) - 0.5 * (0.5 * vel_up**2 + 0.5 * vel_down**2 + g * hh_up + g * hh_down)
                    uv_flux_val = uv_flux_val + 0.5 * c_adv**2 + g * h_star

                    u_cov_up_val = (
                        uu_up * dxdxi_up[ey, ex, xi]
                        + vv_up * dydxi_up[ey, ex, xi]
                        + ww_up * dzdxi_up[ey, ex, xi]
                    )
                    u_cov_down_val = (
                        uu_down * dxdxi_down[ey, ex, xi]
                        + vv_down * dydxi_down[ey, ex, xi]
                        + ww_down * dzdxi_down[ey, ex, xi]
                    )
                    u_contra_up_val = (
                        uu_up * dxidx_up[ey, ex, xi]
                        + vv_up * dxidy_up[ey, ex, xi]
                        + ww_up * dxidz_up[ey, ex, xi]
                    )
                    u_contra_down_val = (
                        uu_down * dxidx_down[ey, ex, xi]
                        + vv_down * dxidy_down[ey, ex, xi]
                        + ww_down * dxidz_down[ey, ex, xi]
                    )
                    v_contra_up_val = (
                        uu_up * detadx_up[ey, ex, xi]
                        + vv_up * detady_up[ey, ex, xi]
                        + ww_up * detadz_up[ey, ex, xi]
                    )
                    v_contra_down_val = (
                        uu_down * detadx_down[ey, ex, xi]
                        + vv_down * detady_down[ey, ex, xi]
                        + ww_down * detadz_down[ey, ex, xi]
                    )

                    adv_sign = 1.0 - 2.0 * (c_adv < 0.0)
                    avg_tan_cov = 0.5 * (
                        u_cov_up_val + u_cov_down_val
                    ) - tangent_scale * adv_sign * (
                        u_cov_up_val - u_cov_down_val
                    )

                    u_flux_up_val = v_contra_up_val * avg_tan_cov
                    u_flux_down_val = v_contra_down_val * avg_tan_cov
                    v_flux_up_val = uv_flux_val - u_contra_up_val * avg_tan_cov
                    v_flux_down_val = uv_flux_val - u_contra_down_val * avg_tan_cov

                    if ey > 0:
                        cell_y = ey - 1
                        h_k[cell_y, ex, n - 1, xi] -= (
                            h_flux_val - h_down_flux_val
                        ) * vert_upper_edge_factor[cell_y, ex, xi]
                        u_k[cell_y, ex, n - 1, xi] -= (
                            u_flux_down_val - v_contra_down_val * u_cov_down_val
                        ) * inv_endpoint_weight
                        v_k[cell_y, ex, n - 1, xi] -= (
                            v_flux_down_val
                            - (uv_down_flux_val - u_contra_down_val * u_cov_down_val)
                        ) * inv_endpoint_weight

                    if ey < ny:
                        h_k[ey, ex, 0, xi] += (
                            h_flux_val - h_up_flux_val
                        ) * vert_lower_edge_factor[ey, ex, xi]
                        u_k[ey, ex, 0, xi] += (
                            u_flux_up_val - v_contra_up_val * u_cov_up_val
                        ) * inv_endpoint_weight
                        v_k[ey, ex, 0, xi] += (
                            v_flux_up_val
                            - (uv_up_flux_val - u_contra_up_val * u_cov_up_val)
                        ) * inv_endpoint_weight

        for ey in range(ny):
            for ex in range(nx + 1):
                for eta in range(n):
                    uu_right = u_right[ey, ex, eta]
                    vv_right = v_right[ey, ex, eta]
                    ww_right = w_right[ey, ex, eta]
                    hh_right = h_right[ey, ex, eta]
                    uu_left = u_left[ey, ex, eta]
                    vv_left = v_left[ey, ex, eta]
                    ww_left = w_left[ey, ex, eta]
                    hh_left = h_left[ey, ex, eta]

                    vel_right = (
                        uu_right * xi_x_right[ey, ex, eta]
                        + vv_right * xi_y_right[ey, ex, eta]
                        + ww_right * xi_z_right[ey, ex, eta]
                    )
                    vel_left = (
                        uu_left * xi_x_left[ey, ex, eta]
                        + vv_left * xi_y_left[ey, ex, eta]
                        + ww_left * xi_z_left[ey, ex, eta]
                    )
                    
                    h_right_flux_val = hh_right * vel_right
                    h_left_flux_val = hh_left * vel_left
                    uv_right_flux_val = 0.5 * (
                        uu_right * uu_right + vv_right * vv_right + ww_right * ww_right
                    ) + g * hh_right
                    uv_left_flux_val = 0.5 * (
                        uu_left * uu_left + vv_left * vv_left + ww_left * ww_left
                    ) + g * hh_left

                    h_sum = hh_right + hh_left
                    inv_h_sum = 1.0 / h_sum
                    c_snd_left = np.sqrt(g * hh_left)
                    c_snd_right = np.sqrt(g * hh_right)
                    c_snd = 0.5 * (c_snd_right + c_snd_left) - 0.25 * (vel_right - vel_left)
                    h_star = c_snd * c_snd / g
                    c_adv = 0.5 * (vel_right + vel_left) + c_snd_left - c_snd_right
                    abs_c_adv = abs(c_adv)

                    # h_flux_val = 0.5 * c_adv * h_sum - ah * abs_c_adv * (hh_right - hh_left)
                    # uv_flux_val = 0.5 * (uv_right_flux_val + uv_left_flux_val) - a * (
                    #     c_snd
                    # ) * (h_right_flux_val - h_left_flux_val) * (2.0 * inv_h_sum)
                    h_flux_val = c_adv * h_star
                    uv_flux_val = 0.5 * (uv_right_flux_val + uv_left_flux_val) - 0.5 * (0.5 * vel_right**2 + 0.5 * vel_left**2 + g * hh_right + g * hh_left)
                    uv_flux_val = uv_flux_val + 0.5 * c_adv**2 + g * h_star

                    v_cov_right_val = (
                        uu_right * dxdeta_right[ey, ex, eta]
                        + vv_right * dydeta_right[ey, ex, eta]
                        + ww_right * dzdeta_right[ey, ex, eta]
                    )
                    v_cov_left_val = (
                        uu_left * dxdeta_left[ey, ex, eta]
                        + vv_left * dydeta_left[ey, ex, eta]
                        + ww_left * dzdeta_left[ey, ex, eta]
                    )
                    u_contra_right_val = (
                        uu_right * dxidx_right[ey, ex, eta]
                        + vv_right * dxidy_right[ey, ex, eta]
                        + ww_right * dxidz_right[ey, ex, eta]
                    )
                    u_contra_left_val = (
                        uu_left * dxidx_left[ey, ex, eta]
                        + vv_left * dxidy_left[ey, ex, eta]
                        + ww_left * dxidz_left[ey, ex, eta]
                    )
                    v_contra_right_val = (
                        uu_right * detadx_right[ey, ex, eta]
                        + vv_right * detady_right[ey, ex, eta]
                        + ww_right * detadz_right[ey, ex, eta]
                    )
                    v_contra_left_val = (
                        uu_left * detadx_left[ey, ex, eta]
                        + vv_left * detady_left[ey, ex, eta]
                        + ww_left * detadz_left[ey, ex, eta]
                    )

                    adv_sign = 1.0 - 2.0 * (c_adv < 0.0)
                    avg_tan_cov = 0.5 * (
                        v_cov_right_val + v_cov_left_val
                    ) - tangent_scale * adv_sign * (
                        v_cov_right_val - v_cov_left_val
                    )

                    u_flux_right_val = uv_flux_val - v_contra_right_val * avg_tan_cov
                    u_flux_left_val = uv_flux_val - v_contra_left_val * avg_tan_cov
                    v_flux_right_val = u_contra_right_val * avg_tan_cov
                    v_flux_left_val = u_contra_left_val * avg_tan_cov

                    if ex > 0:
                        cell_x = ex - 1
                        h_k[ey, cell_x, eta, n - 1] -= (
                            h_flux_val - h_left_flux_val
                        ) * horz_right_edge_factor[ey, cell_x, eta]
                        u_k[ey, cell_x, eta, n - 1] -= (
                            u_flux_left_val
                            - (uv_left_flux_val - v_contra_left_val * v_cov_left_val)
                        ) * inv_endpoint_weight
                        v_k[ey, cell_x, eta, n - 1] -= (
                            v_flux_left_val - u_contra_left_val * v_cov_left_val
                        ) * inv_endpoint_weight

                    if ex < nx:
                        h_k[ey, ex, eta, 0] += (
                            h_flux_val - h_right_flux_val
                        ) * horz_left_edge_factor[ey, ex, eta]
                        u_k[ey, ex, eta, 0] += (
                            u_flux_right_val
                            - (
                                uv_right_flux_val
                                - v_contra_right_val * v_cov_right_val
                            )
                        ) * inv_endpoint_weight
                        v_k[ey, ex, eta, 0] += (
                            v_flux_right_val - u_contra_right_val * v_cov_right_val
                        ) * inv_endpoint_weight

        for ey in range(ny):
            for ex in range(nx):
                for eta in range(n):
                    for xi in range(n):
                        uk_cov = u_k[ey, ex, eta, xi]
                        vk_cov = v_k[ey, ex, eta, xi]
                        u_k[ey, ex, eta, xi] = uk_cov * dxidx[ey, ex, eta, xi] + vk_cov * detadx[ey, ex, eta, xi]
                        v_k[ey, ex, eta, xi] = uk_cov * dxidy[ey, ex, eta, xi] + vk_cov * detady[ey, ex, eta, xi]
                        w_k[ey, ex, eta, xi] = uk_cov * dxidz[ey, ex, eta, xi] + vk_cov * detadz[ey, ex, eta, xi]

        return u_k, v_k, w_k, h_k

    @njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
    def _apply_old_tangent_diss_numba(
        u_k, v_k, w_k, endpoint_weight,
        dxidx, dxidy, dxidz, detadx, detady, detadz,
        u_up, v_up, w_up, h_up, u_down, v_down, w_down, h_down,
        u_right, v_right, w_right, h_right, u_left, v_left, w_left, h_left,
        eta_x_up, eta_y_up, eta_z_up, eta_x_down, eta_y_down, eta_z_down,
        xi_x_right, xi_y_right, xi_z_right, xi_x_left, xi_y_left, xi_z_left,
        J_eta, J_xi,
        dxdxi_up, dydxi_up, dzdxi_up, dxdxi_down, dydxi_down, dzdxi_down,
        dxdxi_right, dydxi_right, dzdxi_right, dxdxi_left, dydxi_left, dzdxi_left,
        dxdeta_up, dydeta_up, dzdeta_up, dxdeta_down, dydeta_down, dzdeta_down,
        dxdeta_right, dydeta_right, dzdeta_right, dxdeta_left, dydeta_left, dzdeta_left,
    ):
        ny, nx, n, _ = u_k.shape
        inv_endpoint_weight = 1.0 / endpoint_weight

        for ey in range(ny + 1):
            for ex in range(nx):
                for xi in range(n):
                    uu_up = u_up[ey, ex, xi]
                    vv_up = v_up[ey, ex, xi]
                    ww_up = w_up[ey, ex, xi]
                    hh_up = h_up[ey, ex, xi]
                    uu_down = u_down[ey, ex, xi]
                    vv_down = v_down[ey, ex, xi]
                    ww_down = w_down[ey, ex, xi]
                    hh_down = h_down[ey, ex, xi]

                    h_up_flux_val = hh_up * (
                        uu_up * eta_x_up[ey, ex, xi]
                        + vv_up * eta_y_up[ey, ex, xi]
                        + ww_up * eta_z_up[ey, ex, xi]
                    )
                    h_down_flux_val = hh_down * (
                        uu_down * eta_x_down[ey, ex, xi]
                        + vv_down * eta_y_down[ey, ex, xi]
                        + ww_down * eta_z_down[ey, ex, xi]
                    )
                    inv_h_sum = 1.0 / (hh_up + hh_down)
                    old_c_adv = 0.5 * abs(
                        h_up_flux_val / hh_up + h_down_flux_val / hh_down
                    )

                    u_cov_up_val = (
                        uu_up * dxdxi_up[ey, ex, xi]
                        + vv_up * dydxi_up[ey, ex, xi]
                        + ww_up * dzdxi_up[ey, ex, xi]
                    )
                    u_cov_down_val = (
                        uu_down * dxdxi_down[ey, ex, xi]
                        + vv_down * dydxi_down[ey, ex, xi]
                        + ww_down * dzdxi_down[ey, ex, xi]
                    )
                    v_cov_up_val = (
                        uu_up * dxdeta_up[ey, ex, xi]
                        + vv_up * dydeta_up[ey, ex, xi]
                        + ww_up * dzdeta_up[ey, ex, xi]
                    )
                    v_cov_down_val = (
                        uu_down * dxdeta_down[ey, ex, xi]
                        + vv_down * dydeta_down[ey, ex, xi]
                        + ww_down * dzdeta_down[ey, ex, xi]
                    )
                    old_u_diss = -old_c_adv * (
                        hh_up * u_cov_up_val - hh_down * u_cov_down_val
                    ) * inv_h_sum
                    old_v_diss = -old_c_adv * (
                        hh_up * v_cov_up_val - hh_down * v_cov_down_val
                    ) * inv_h_sum
                    old_vel_diss = old_c_adv * (
                        h_up_flux_val - h_down_flux_val
                    ) * inv_h_sum

                    if ey > 0:
                        cell_y = ey - 1
                        eta_idx = n - 1
                        eta_scale = J_eta[cell_y, ex, eta_idx, xi]
                        du_cov = -old_u_diss * eta_scale * inv_endpoint_weight
                        dv_cov = -(old_v_diss * eta_scale + old_vel_diss) * inv_endpoint_weight
                        u_k[cell_y, ex, eta_idx, xi] += (
                            du_cov * dxidx[cell_y, ex, eta_idx, xi]
                            + dv_cov * detadx[cell_y, ex, eta_idx, xi]
                        )
                        v_k[cell_y, ex, eta_idx, xi] += (
                            du_cov * dxidy[cell_y, ex, eta_idx, xi]
                            + dv_cov * detady[cell_y, ex, eta_idx, xi]
                        )
                        w_k[cell_y, ex, eta_idx, xi] += (
                            du_cov * dxidz[cell_y, ex, eta_idx, xi]
                            + dv_cov * detadz[cell_y, ex, eta_idx, xi]
                        )

                    if ey < ny:
                        eta_idx = 0
                        eta_scale = J_eta[ey, ex, eta_idx, xi]
                        du_cov = old_u_diss * eta_scale * inv_endpoint_weight
                        dv_cov = (old_v_diss * eta_scale + old_vel_diss) * inv_endpoint_weight
                        u_k[ey, ex, eta_idx, xi] += (
                            du_cov * dxidx[ey, ex, eta_idx, xi]
                            + dv_cov * detadx[ey, ex, eta_idx, xi]
                        )
                        v_k[ey, ex, eta_idx, xi] += (
                            du_cov * dxidy[ey, ex, eta_idx, xi]
                            + dv_cov * detady[ey, ex, eta_idx, xi]
                        )
                        w_k[ey, ex, eta_idx, xi] += (
                            du_cov * dxidz[ey, ex, eta_idx, xi]
                            + dv_cov * detadz[ey, ex, eta_idx, xi]
                        )

        for ey in range(ny):
            for ex in range(nx + 1):
                for eta in range(n):
                    uu_right = u_right[ey, ex, eta]
                    vv_right = v_right[ey, ex, eta]
                    ww_right = w_right[ey, ex, eta]
                    hh_right = h_right[ey, ex, eta]
                    uu_left = u_left[ey, ex, eta]
                    vv_left = v_left[ey, ex, eta]
                    ww_left = w_left[ey, ex, eta]
                    hh_left = h_left[ey, ex, eta]

                    h_right_flux_val = hh_right * (
                        uu_right * xi_x_right[ey, ex, eta]
                        + vv_right * xi_y_right[ey, ex, eta]
                        + ww_right * xi_z_right[ey, ex, eta]
                    )
                    h_left_flux_val = hh_left * (
                        uu_left * xi_x_left[ey, ex, eta]
                        + vv_left * xi_y_left[ey, ex, eta]
                        + ww_left * xi_z_left[ey, ex, eta]
                    )
                    inv_h_sum = 1.0 / (hh_right + hh_left)
                    old_c_adv = 0.5 * abs(
                        h_right_flux_val / hh_right + h_left_flux_val / hh_left
                    )

                    u_cov_right_val = (
                        uu_right * dxdxi_right[ey, ex, eta]
                        + vv_right * dydxi_right[ey, ex, eta]
                        + ww_right * dzdxi_right[ey, ex, eta]
                    )
                    u_cov_left_val = (
                        uu_left * dxdxi_left[ey, ex, eta]
                        + vv_left * dydxi_left[ey, ex, eta]
                        + ww_left * dzdxi_left[ey, ex, eta]
                    )
                    v_cov_right_val = (
                        uu_right * dxdeta_right[ey, ex, eta]
                        + vv_right * dydeta_right[ey, ex, eta]
                        + ww_right * dzdeta_right[ey, ex, eta]
                    )
                    v_cov_left_val = (
                        uu_left * dxdeta_left[ey, ex, eta]
                        + vv_left * dydeta_left[ey, ex, eta]
                        + ww_left * dzdeta_left[ey, ex, eta]
                    )
                    old_u_diss = -old_c_adv * (
                        hh_right * u_cov_right_val - hh_left * u_cov_left_val
                    ) * inv_h_sum
                    old_v_diss = -old_c_adv * (
                        hh_right * v_cov_right_val - hh_left * v_cov_left_val
                    ) * inv_h_sum
                    old_vel_diss = old_c_adv * (
                        h_right_flux_val - h_left_flux_val
                    ) * inv_h_sum

                    if ex > 0:
                        cell_x = ex - 1
                        xi_idx = n - 1
                        xi_scale = J_xi[ey, cell_x, eta, xi_idx]
                        du_cov = -(old_u_diss * xi_scale + old_vel_diss) * inv_endpoint_weight
                        dv_cov = -old_v_diss * xi_scale * inv_endpoint_weight
                        u_k[ey, cell_x, eta, xi_idx] += (
                            du_cov * dxidx[ey, cell_x, eta, xi_idx]
                            + dv_cov * detadx[ey, cell_x, eta, xi_idx]
                        )
                        v_k[ey, cell_x, eta, xi_idx] += (
                            du_cov * dxidy[ey, cell_x, eta, xi_idx]
                            + dv_cov * detady[ey, cell_x, eta, xi_idx]
                        )
                        w_k[ey, cell_x, eta, xi_idx] += (
                            du_cov * dxidz[ey, cell_x, eta, xi_idx]
                            + dv_cov * detadz[ey, cell_x, eta, xi_idx]
                        )

                    if ex < nx:
                        xi_idx = 0
                        xi_scale = J_xi[ey, ex, eta, xi_idx]
                        du_cov = (old_u_diss * xi_scale + old_vel_diss) * inv_endpoint_weight
                        dv_cov = old_v_diss * xi_scale * inv_endpoint_weight
                        u_k[ey, ex, eta, xi_idx] += (
                            du_cov * dxidx[ey, ex, eta, xi_idx]
                            + dv_cov * detadx[ey, ex, eta, xi_idx]
                        )
                        v_k[ey, ex, eta, xi_idx] += (
                            du_cov * dxidy[ey, ex, eta, xi_idx]
                            + dv_cov * detady[ey, ex, eta, xi_idx]
                        )
                        w_k[ey, ex, eta, xi_idx] += (
                            du_cov * dxidz[ey, ex, eta, xi_idx]
                            + dv_cov * detadz[ey, ex, eta, xi_idx]
                        )

    @njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
    def _apply_barth_diss_numba(
        u_k, v_k, w_k, h_k, g,
        vert_upper_edge_factor, vert_lower_edge_factor,
        horz_right_edge_factor, horz_left_edge_factor,
        u_up, v_up, w_up, h_up, u_down, v_down, w_down, h_down,
        u_right, v_right, w_right, h_right, u_left, v_left, w_left, h_left,
        eta_x_up, eta_y_up, eta_z_up, eta_x_down, eta_y_down, eta_z_down,
        xi_x_right, xi_y_right, xi_z_right, xi_x_left, xi_y_left, xi_z_left,
        dxdxi_up, dydxi_up, dzdxi_up, dxdxi_down, dydxi_down, dzdxi_down,
        dxdxi_right, dydxi_right, dzdxi_right, dxdxi_left, dydxi_left, dzdxi_left,
        dxdeta_up, dydeta_up, dzdeta_up, dxdeta_down, dydeta_down, dzdeta_down,
        dxdeta_right, dydeta_right, dzdeta_right, dxdeta_left, dydeta_left, dzdeta_left,
        normal_only,
    ):
        ny, nx, n, _ = u_k.shape

        for ey in range(ny + 1):
            for ex in range(nx):
                for xi in range(n):
                    uu_l = u_down[ey, ex, xi]
                    vv_l = v_down[ey, ex, xi]
                    ww_l = w_down[ey, ex, xi]
                    hh_l = h_down[ey, ex, xi]
                    uu_r = u_up[ey, ex, xi]
                    vv_r = v_up[ey, ex, xi]
                    ww_r = w_up[ey, ex, xi]
                    hh_r = h_up[ey, ex, xi]

                    nx_face = 0.5 * (eta_x_down[ey, ex, xi] + eta_x_up[ey, ex, xi])
                    ny_face = 0.5 * (eta_y_down[ey, ex, xi] + eta_y_up[ey, ex, xi])
                    nz_face = 0.5 * (eta_z_down[ey, ex, xi] + eta_z_up[ey, ex, xi])
                    n_norm = np.sqrt(nx_face * nx_face + ny_face * ny_face + nz_face * nz_face)
                    nx_face = nx_face / n_norm
                    ny_face = ny_face / n_norm
                    nz_face = nz_face / n_norm

                    tx_face = 0.5 * (dxdxi_down[ey, ex, xi] + dxdxi_up[ey, ex, xi])
                    ty_face = 0.5 * (dydxi_down[ey, ex, xi] + dydxi_up[ey, ex, xi])
                    tz_face = 0.5 * (dzdxi_down[ey, ex, xi] + dzdxi_up[ey, ex, xi])
                    t_norm = np.sqrt(tx_face * tx_face + ty_face * ty_face + tz_face * tz_face)
                    tx_face = tx_face / t_norm
                    ty_face = ty_face / t_norm
                    tz_face = tz_face / t_norm

                    h_avg = 0.5 * (hh_l + hh_r)
                    c = np.sqrt(g * h_avg)
                    u_avg = 0.5 * (uu_l + uu_r)
                    v_avg = 0.5 * (vv_l + vv_r)
                    w_avg = 0.5 * (ww_l + ww_r)
                    du = uu_r - uu_l
                    dv = vv_r - vv_l
                    dw = ww_r - ww_l
                    dh = hh_r - hh_l

                    u_n = u_avg * nx_face + v_avg * ny_face + w_avg * nz_face
                    du_n = du * nx_face + dv * ny_face + dw * nz_face
                    du_t = du * tx_face + dv * ty_face + dw * tz_face

                    mu_m = abs(u_n - c)
                    mu_0 = abs(u_n)
                    mu_p = abs(u_n + c)
                    amp_m = (g * dh - c * du_n) / (2.0 * g)
                    amp_p = (g * dh + c * du_n) / (2.0 * g)

                    diss_h = mu_m * amp_m + mu_p * amp_p
                    if normal_only:
                        diss_un = mu_m * amp_m * (u_n - c) + mu_p * amp_p * (u_n + c)
                    else:
                        amp_0 = h_avg * du_t
                        diss_u = (
                            mu_m * amp_m * (u_avg - c * nx_face)
                            + mu_p * amp_p * (u_avg + c * nx_face)
                            + mu_0 * amp_0 * tx_face
                        )
                        diss_v = (
                            mu_m * amp_m * (v_avg - c * ny_face)
                            + mu_p * amp_p * (v_avg + c * ny_face)
                            + mu_0 * amp_0 * ty_face
                        )
                        diss_w = (
                            mu_m * amp_m * (w_avg - c * nz_face)
                            + mu_p * amp_p * (w_avg + c * nz_face)
                            + mu_0 * amp_0 * tz_face
                        )

                    if ey > 0:
                        cell_y = ey - 1
                        eta_idx = n - 1
                        scale = 0.5 * vert_upper_edge_factor[cell_y, ex, xi]
                        dh_k = scale * diss_h
                        h_k[cell_y, ex, eta_idx, xi] += dh_k
                        if normal_only:
                            un_l = uu_l * nx_face + vv_l * ny_face + ww_l * nz_face
                            dun = (scale * diss_un - un_l * dh_k) / hh_l
                            u_k[cell_y, ex, eta_idx, xi] += dun * nx_face
                            v_k[cell_y, ex, eta_idx, xi] += dun * ny_face
                            w_k[cell_y, ex, eta_idx, xi] += dun * nz_face
                        else:
                            dm_u = scale * diss_u
                            dm_v = scale * diss_v
                            dm_w = scale * diss_w
                            u_k[cell_y, ex, eta_idx, xi] += (dm_u - uu_l * dh_k) / hh_l
                            v_k[cell_y, ex, eta_idx, xi] += (dm_v - vv_l * dh_k) / hh_l
                            w_k[cell_y, ex, eta_idx, xi] += (dm_w - ww_l * dh_k) / hh_l

                    if ey < ny:
                        eta_idx = 0
                        scale = -0.5 * vert_lower_edge_factor[ey, ex, xi]
                        dh_k = scale * diss_h
                        h_k[ey, ex, eta_idx, xi] += dh_k
                        if normal_only:
                            un_r = uu_r * nx_face + vv_r * ny_face + ww_r * nz_face
                            dun = (scale * diss_un - un_r * dh_k) / hh_r
                            u_k[ey, ex, eta_idx, xi] += dun * nx_face
                            v_k[ey, ex, eta_idx, xi] += dun * ny_face
                            w_k[ey, ex, eta_idx, xi] += dun * nz_face
                        else:
                            dm_u = scale * diss_u
                            dm_v = scale * diss_v
                            dm_w = scale * diss_w
                            u_k[ey, ex, eta_idx, xi] += (dm_u - uu_r * dh_k) / hh_r
                            v_k[ey, ex, eta_idx, xi] += (dm_v - vv_r * dh_k) / hh_r
                            w_k[ey, ex, eta_idx, xi] += (dm_w - ww_r * dh_k) / hh_r

        for ey in range(ny):
            for ex in range(nx + 1):
                for eta in range(n):
                    uu_l = u_left[ey, ex, eta]
                    vv_l = v_left[ey, ex, eta]
                    ww_l = w_left[ey, ex, eta]
                    hh_l = h_left[ey, ex, eta]
                    uu_r = u_right[ey, ex, eta]
                    vv_r = v_right[ey, ex, eta]
                    ww_r = w_right[ey, ex, eta]
                    hh_r = h_right[ey, ex, eta]

                    nx_face = 0.5 * (xi_x_left[ey, ex, eta] + xi_x_right[ey, ex, eta])
                    ny_face = 0.5 * (xi_y_left[ey, ex, eta] + xi_y_right[ey, ex, eta])
                    nz_face = 0.5 * (xi_z_left[ey, ex, eta] + xi_z_right[ey, ex, eta])
                    n_norm = np.sqrt(nx_face * nx_face + ny_face * ny_face + nz_face * nz_face)
                    nx_face = nx_face / n_norm
                    ny_face = ny_face / n_norm
                    nz_face = nz_face / n_norm

                    tx_face = 0.5 * (dxdeta_left[ey, ex, eta] + dxdeta_right[ey, ex, eta])
                    ty_face = 0.5 * (dydeta_left[ey, ex, eta] + dydeta_right[ey, ex, eta])
                    tz_face = 0.5 * (dzdeta_left[ey, ex, eta] + dzdeta_right[ey, ex, eta])
                    t_norm = np.sqrt(tx_face * tx_face + ty_face * ty_face + tz_face * tz_face)
                    tx_face = tx_face / t_norm
                    ty_face = ty_face / t_norm
                    tz_face = tz_face / t_norm

                    h_avg = 0.5 * (hh_l + hh_r)
                    c = np.sqrt(g * h_avg)
                    u_avg = 0.5 * (uu_l + uu_r)
                    v_avg = 0.5 * (vv_l + vv_r)
                    w_avg = 0.5 * (ww_l + ww_r)
                    du = uu_r - uu_l
                    dv = vv_r - vv_l
                    dw = ww_r - ww_l
                    dh = hh_r - hh_l

                    u_n = u_avg * nx_face + v_avg * ny_face + w_avg * nz_face
                    du_n = du * nx_face + dv * ny_face + dw * nz_face
                    du_t = du * tx_face + dv * ty_face + dw * tz_face

                    mu_m = abs(u_n - c)
                    mu_0 = abs(u_n)
                    mu_p = abs(u_n + c)
                    amp_m = (g * dh - c * du_n) / (2.0 * g)
                    amp_p = (g * dh + c * du_n) / (2.0 * g)

                    diss_h = mu_m * amp_m + mu_p * amp_p
                    if normal_only:
                        diss_un = mu_m * amp_m * (u_n - c) + mu_p * amp_p * (u_n + c)
                    else:
                        amp_0 = h_avg * du_t
                        diss_u = (
                            mu_m * amp_m * (u_avg - c * nx_face)
                            + mu_p * amp_p * (u_avg + c * nx_face)
                            + mu_0 * amp_0 * tx_face
                        )
                        diss_v = (
                            mu_m * amp_m * (v_avg - c * ny_face)
                            + mu_p * amp_p * (v_avg + c * ny_face)
                            + mu_0 * amp_0 * ty_face
                        )
                        diss_w = (
                            mu_m * amp_m * (w_avg - c * nz_face)
                            + mu_p * amp_p * (w_avg + c * nz_face)
                            + mu_0 * amp_0 * tz_face
                        )

                    if ex > 0:
                        cell_x = ex - 1
                        xi_idx = n - 1
                        scale = 0.5 * horz_right_edge_factor[ey, cell_x, eta]
                        dh_k = scale * diss_h
                        h_k[ey, cell_x, eta, xi_idx] += dh_k
                        if normal_only:
                            un_l = uu_l * nx_face + vv_l * ny_face + ww_l * nz_face
                            dun = (scale * diss_un - un_l * dh_k) / hh_l
                            u_k[ey, cell_x, eta, xi_idx] += dun * nx_face
                            v_k[ey, cell_x, eta, xi_idx] += dun * ny_face
                            w_k[ey, cell_x, eta, xi_idx] += dun * nz_face
                        else:
                            dm_u = scale * diss_u
                            dm_v = scale * diss_v
                            dm_w = scale * diss_w
                            u_k[ey, cell_x, eta, xi_idx] += (dm_u - uu_l * dh_k) / hh_l
                            v_k[ey, cell_x, eta, xi_idx] += (dm_v - vv_l * dh_k) / hh_l
                            w_k[ey, cell_x, eta, xi_idx] += (dm_w - ww_l * dh_k) / hh_l

                    if ex < nx:
                        xi_idx = 0
                        scale = -0.5 * horz_left_edge_factor[ey, ex, eta]
                        dh_k = scale * diss_h
                        h_k[ey, ex, eta, xi_idx] += dh_k
                        if normal_only:
                            un_r = uu_r * nx_face + vv_r * ny_face + ww_r * nz_face
                            dun = (scale * diss_un - un_r * dh_k) / hh_r
                            u_k[ey, ex, eta, xi_idx] += dun * nx_face
                            v_k[ey, ex, eta, xi_idx] += dun * ny_face
                            w_k[ey, ex, eta, xi_idx] += dun * nz_face
                        else:
                            dm_u = scale * diss_u
                            dm_v = scale * diss_v
                            dm_w = scale * diss_w
                            u_k[ey, ex, eta, xi_idx] += (dm_u - uu_r * dh_k) / hh_r
                            v_k[ey, ex, eta, xi_idx] += (dm_v - vv_r * dh_k) / hh_r
                            w_k[ey, ex, eta, xi_idx] += (dm_w - ww_r * dh_k) / hh_r

    @njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
    def _solve_numba_old_tangent_kernel(
        u, v, w, h, b,
        D, endpoint_weight, J,
        vert_upper_edge_factor, vert_lower_edge_factor,
        horz_right_edge_factor, horz_left_edge_factor,
        dxidx, dxidy, dxidz, detadx, detady, detadz,
        dxidx_up, dxidy_up, dxidz_up, dxidx_down, dxidy_down, dxidz_down,
        dxidx_right, dxidy_right, dxidz_right, dxidx_left, dxidy_left, dxidz_left,
        detadx_up, detady_up, detadz_up, detadx_down, detady_down, detadz_down,
        detadx_right, detady_right, detadz_right, detadx_left, detady_left, detadz_left,
        dxdxi, dydxi, dzdxi, dxdeta, dydeta, dzdeta,
        f,
        u_up, v_up, w_up, h_up, u_down, v_down, w_down, h_down,
        u_right, v_right, w_right, h_right, u_left, v_left, w_left, h_left,
        eta_x_up, eta_y_up, eta_z_up, eta_x_down, eta_y_down, eta_z_down,
        xi_x_right, xi_y_right, xi_z_right, xi_x_left, xi_y_left, xi_z_left,
        J_eta, J_xi,
        dxdxi_up, dydxi_up, dzdxi_up, dxdxi_down, dydxi_down, dzdxi_down,
        dxdxi_right, dydxi_right, dzdxi_right, dxdxi_left, dydxi_left, dzdxi_left,
        dxdeta_up, dydeta_up, dzdeta_up, dxdeta_down, dydeta_down, dzdeta_down,
        dxdeta_right, dydeta_right, dzdeta_right, dxdeta_left, dydeta_left, dzdeta_left,
        g, a, ah,
    ):
        u_k, v_k, w_k, h_k = _solve_numba_kernel(
            u, v, w, h, b,
            D, endpoint_weight, J,
            vert_upper_edge_factor, vert_lower_edge_factor,
            horz_right_edge_factor, horz_left_edge_factor,
            dxidx, dxidy, dxidz, detadx, detady, detadz,
            dxidx_up, dxidy_up, dxidz_up, dxidx_down, dxidy_down, dxidz_down,
            dxidx_right, dxidy_right, dxidz_right, dxidx_left, dxidy_left, dxidz_left,
            detadx_up, detady_up, detadz_up, detadx_down, detady_down, detadz_down,
            detadx_right, detady_right, detadz_right, detadx_left, detady_left, detadz_left,
            dxdxi, dydxi, dzdxi, dxdeta, dydeta, dzdeta,
            f,
            u_up, v_up, w_up, h_up, u_down, v_down, w_down, h_down,
            u_right, v_right, w_right, h_right, u_left, v_left, w_left, h_left,
            eta_x_up, eta_y_up, eta_z_up, eta_x_down, eta_y_down, eta_z_down,
            xi_x_right, xi_y_right, xi_z_right, xi_x_left, xi_y_left, xi_z_left,
            dxdxi_up, dydxi_up, dzdxi_up, dxdxi_down, dydxi_down, dzdxi_down,
            dxdxi_right, dydxi_right, dzdxi_right, dxdxi_left, dydxi_left, dzdxi_left,
            dxdeta_up, dydeta_up, dzdeta_up, dxdeta_down, dydeta_down, dzdeta_down,
            dxdeta_right, dydeta_right, dzdeta_right, dxdeta_left, dydeta_left, dzdeta_left,
            g, a, ah, False,
        )
        _apply_old_tangent_diss_numba(
            u_k, v_k, w_k, endpoint_weight,
            dxidx, dxidy, dxidz, detadx, detady, detadz,
            u_up, v_up, w_up, h_up, u_down, v_down, w_down, h_down,
            u_right, v_right, w_right, h_right, u_left, v_left, w_left, h_left,
            eta_x_up, eta_y_up, eta_z_up, eta_x_down, eta_y_down, eta_z_down,
            xi_x_right, xi_y_right, xi_z_right, xi_x_left, xi_y_left, xi_z_left,
            J_eta, J_xi,
            dxdxi_up, dydxi_up, dzdxi_up, dxdxi_down, dydxi_down, dzdxi_down,
            dxdxi_right, dydxi_right, dzdxi_right, dxdxi_left, dydxi_left, dzdxi_left,
            dxdeta_up, dydeta_up, dzdeta_up, dxdeta_down, dydeta_down, dzdeta_down,
            dxdeta_right, dydeta_right, dzdeta_right, dxdeta_left, dydeta_left, dzdeta_left,
        )
        return u_k, v_k, w_k, h_k
else:
    _solve_numba_kernel = None
    _solve_numba_lmars_kernel = None
    _apply_barth_diss_numba = None
    _solve_numba_old_tangent_kernel = None


class DGCubedSphereSWENumpy:
    def __init__(
            self, poly_order, nx, ny, g, f, eps, radius=1.0, device='cpu',
            solution=None, a=0.0, ah=0.0, dtype=np.float64,
            flux_type="standard", nprocx=1, nprocy=1, comm=None, **kwargs):

        _reject_legacy_flux_kwargs(kwargs)
        self.face_names = ['zp', 'zn', 'xp', 'xn', 'yp', 'yn']
        flux_type = _validate_flux_type(flux_type)
        self.nprocx = nprocx
        self.nprocy = nprocy
        self.comm = self._get_comm(comm, nprocx, nprocy)
        self.rank = self.comm.Get_rank() if self.comm is not None else 0
        self.size = self.comm.Get_size() if self.comm is not None else 1
        self.parallel = self.size > 1 or nprocx * nprocy > 1
        if self.parallel and self.comm is None:
            raise ImportError("mpi4py is required when nprocx*nprocy > 1.")
        if self.parallel:
            if nprocx != nprocy:
                raise ValueError("MPI cubed-sphere partitioning expects nprocx == nprocy.")
            if self.size != self.nproc:
                raise ValueError(f"MPI size must be {self.nproc}; found {self.size}.")
            self.tile_idx = self.rank // (self.nprocx * self.nprocy)
            self.face_name = self.face_names[self.tile_idx]
            self.active_face_names = [self.face_name]
        else:
            self.tile_idx = 0
            self.face_name = None
            self.active_face_names = self.face_names

        face_nx = nx
        face_ny = ny
        face_partition = {}
        if self.parallel:
            global_nx, local_nx, x_min, x_max = self._local_axis_partition(
                nx, self.nprocx, self.x_proc_idx, "nx"
            )
            global_ny, local_ny, y_min, y_max = self._local_axis_partition(
                ny, self.nprocy, self.y_proc_idx, "ny"
            )
            face_nx = local_nx + 1
            face_ny = local_ny + 1
            face_partition = {
                "global_nx": global_nx,
                "global_ny": global_ny,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            }

        self.faces = {
            name: DGCubedSphereFaceNumpy(
                name, poly_order, face_nx, face_ny, g, f, radius, eps, device, a=a, ah=ah, dtype=dtype,
                bc='', flux_type=flux_type,
                x_proc_idx=self.x_proc_idx if self.parallel else 0,
                y_proc_idx=self.y_proc_idx if self.parallel else 0,
                nprocx=self.nprocx if self.parallel else 1,
                nprocy=self.nprocy if self.parallel else 1,
                **face_partition,
            )
            for name in self.active_face_names
        }
        if self.parallel:
            self._init_mpi_boundary_exchange()

        self.time = 0
        self.cdt = min(self.faces[n].cdt for n in self.active_face_names)
        self.flux_type = flux_type

        self.time_list = []
        self.energy_list = []
        self.enstrophy_list = []
        self.mass_list = []
        self.vorticity_list = []
        self.vorticity_diagnostic = False # calculates a continuous diagnostic vorticity for plotting

    @staticmethod
    def _get_comm(comm, nprocx, nprocy):
        if comm is not None:
            return comm
        mpi_world_size = max(
            int(os.environ.get(name, "1"))
            for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MPI_LOCALNRANKS")
        )
        if nprocx * nprocy == 1 and mpi_world_size == 1:
            return None
        try:
            from mpi4py import MPI
            return MPI.COMM_WORLD
        except ImportError as exc:
            if nprocx * nprocy == 1:
                return None
            raise ImportError("mpi4py is required when nprocx*nprocy > 1.") from exc
        

    @property
    def nproc(self):
        return 6 * self.nprocx * self.nprocy

    @staticmethod
    def _local_axis_partition(num_points, nproc, proc_idx, name):
        num_elements = num_points - 1
        if num_elements % nproc != 0:
            raise ValueError(
                f"{name} - 1 must be divisible by nproc; got {name}={num_points}, nproc={nproc}."
            )

        local_elements = num_elements // nproc
        start = proc_idx * local_elements
        stop = start + local_elements
        dx = 1.0 / num_elements
        return (
            num_elements,
            local_elements,
            -0.5 + start * dx,
            -0.5 + stop * dx,
        )

    @property
    def x_proc_idx(self):
        return (self.rank - self.tile_idx * self.nprocx * self.nprocy) // self.nprocy

    @property
    def y_proc_idx(self):
        return (self.rank - self.tile_idx * self.nprocx * self.nprocy) % self.nprocy

    def _side_connection(self, side):
        face = self.faces[self.face_name]
        for name, (i1, i2) in face.connections:
            if i1 == side:
                return name, i2
        raise ValueError(f"No cubed-sphere connection for side {side}.")

    @property
    def prev_procx(self):
        if self.x_proc_idx == 0:
            return self.get_proc(self._side_connection(2), self.y_proc_idx)
        return self.rank - self.nprocy

    @property
    def next_procx(self):
        if self.x_proc_idx == self.nprocx - 1:
            return self.get_proc(self._side_connection(0), self.y_proc_idx)
        return self.rank + self.nprocy

    @property
    def prev_procy(self):
        if self.y_proc_idx == 0:
            return self.get_proc(self._side_connection(3), self.x_proc_idx)
        return self.rank - 1

    @property
    def next_procy(self):
        if self.y_proc_idx == self.nprocy - 1:
            return self.get_proc(self._side_connection(1), self.x_proc_idx)
        return self.rank + 1

    def get_proc(self, conn, tang_idx):
        name, bdry_code = conn
        tile_idx = self.face_names.index(name) * self.nprocx * self.nprocy
        if bdry_code == 0:
            return tile_idx + (self.nprocx - 1) * self.nprocy + tang_idx
        elif bdry_code == 1:
            return tile_idx + tang_idx * self.nprocy + (self.nprocy - 1)
        elif bdry_code == 2:
            return tile_idx + tang_idx
        elif bdry_code == 3:
            return tile_idx + tang_idx * self.nprocy
        else:
            raise ValueError

    def set_vort(self, sol):
        for name in self.active_face_names:
            face = self.faces[name]
            face.vort = face.dg_vort(*sol[name])

    def boundaries(self, sol=None):

        if sol is None:
            sol = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}

        if self.vorticity_diagnostic:
            self.set_vort(sol)

        if self.parallel:
            reqs = self.fill_boundaries(sol)
            self.recv_boundaries(reqs)
            return

        for name in self.active_face_names:

            face = self.faces[name]

            for con in face.connections:
                n, (i1, i2) = con

                neighbour = self.faces[n]
                u, v, w, h = sol[n]
                vort = neighbour.vort
                self._assign_edge_state(face, i1, self._edge_state(neighbour, (u, v, w, h), i2))

    @staticmethod
    def _edge_state(face, state, side):
        u, v, w, h = state
        vort = face.vort
        if side == 0:
            data = (u[:, -1, :, -1], v[:, -1, :, -1], w[:, -1, :, -1], h[:, -1, :, -1], vort[:, -1, :, -1])
        elif side == 1:
            data = (u[-1, :, -1], v[-1, :, -1], w[-1, :, -1], h[-1, :, -1], vort[-1, :, -1])
        elif side == 2:
            data = (u[:, 0, :, 0], v[:, 0, :, 0], w[:, 0, :, 0], h[:, 0, :, 0], vort[:, 0, :, 0])
        elif side == 3:
            data = (u[0, :, 0], v[0, :, 0], w[0, :, 0], h[0, :, 0], vort[0, :, 0])
        else:
            raise ValueError(f"Unknown boundary side {side}.")
        return np.ascontiguousarray(np.stack(data))

    @staticmethod
    def _pack_edge_state(face, state, side, out):
        u, v, w, h = state
        vort = face.vort
        if side == 0:
            out[0] = u[:, -1, :, -1]
            out[1] = v[:, -1, :, -1]
            out[2] = w[:, -1, :, -1]
            out[3] = h[:, -1, :, -1]
            out[4] = vort[:, -1, :, -1]
        elif side == 1:
            out[0] = u[-1, :, -1]
            out[1] = v[-1, :, -1]
            out[2] = w[-1, :, -1]
            out[3] = h[-1, :, -1]
            out[4] = vort[-1, :, -1]
        elif side == 2:
            out[0] = u[:, 0, :, 0]
            out[1] = v[:, 0, :, 0]
            out[2] = w[:, 0, :, 0]
            out[3] = h[:, 0, :, 0]
            out[4] = vort[:, 0, :, 0]
        elif side == 3:
            out[0] = u[0, :, 0]
            out[1] = v[0, :, 0]
            out[2] = w[0, :, 0]
            out[3] = h[0, :, 0]
            out[4] = vort[0, :, 0]
        else:
            raise ValueError(f"Unknown boundary side {side}.")

    @staticmethod
    def _assign_edge_state(face, side, data):
        u, v, w, h, vort = data
        if side == 0:
            face.u_right[:, -1] = u
            face.v_right[:, -1] = v
            face.w_right[:, -1] = w
            face.h_right[:, -1] = h
            face.vort_right[:, -1] = vort
        elif side == 1:
            face.u_up[-1] = u
            face.v_up[-1] = v
            face.w_up[-1] = w
            face.h_up[-1] = h
            face.vort_up[-1] = vort
        elif side == 2:
            face.u_left[:, 0] = u
            face.v_left[:, 0] = v
            face.w_left[:, 0] = w
            face.h_left[:, 0] = h
            face.vort_left[:, 0] = vort
        elif side == 3:
            face.u_down[0] = u
            face.v_down[0] = v
            face.w_down[0] = w
            face.h_down[0] = h
            face.vort_down[0] = vort
        else:
            raise ValueError(f"Unknown boundary side {side}.")

    def _init_mpi_boundary_exchange(self):
        face = self.faces[self.face_name]
        nvars = 5
        dtype = face.dtype

        self.right_boundary_x = np.zeros((nvars, face.ny, face.n), dtype=dtype)
        self.left_boundary_x = np.zeros_like(self.right_boundary_x)
        self.right_boundary_x_send = np.zeros_like(self.right_boundary_x)
        self.left_boundary_x_send = np.zeros_like(self.right_boundary_x)

        self.right_boundary_y = np.zeros((nvars, face.nx, face.n), dtype=dtype)
        self.left_boundary_y = np.zeros_like(self.right_boundary_y)
        self.right_boundary_y_send = np.zeros_like(self.right_boundary_y)
        self.left_boundary_y_send = np.zeros_like(self.right_boundary_y)

        self.req_right_boundary_x_send = self.comm.Send_init(self.right_boundary_x_send, dest=self.next_procx)
        self.req_right_boundary_x_recv = self.comm.Recv_init(self.right_boundary_x, source=self.next_procx)
        self.req_left_boundary_x_send = self.comm.Send_init(self.left_boundary_x_send, dest=self.prev_procx)
        self.req_left_boundary_x_recv = self.comm.Recv_init(self.left_boundary_x, source=self.prev_procx)

        self.req_right_boundary_y_send = self.comm.Send_init(self.right_boundary_y_send, dest=self.next_procy)
        self.req_right_boundary_y_recv = self.comm.Recv_init(self.right_boundary_y, source=self.next_procy)
        self.req_left_boundary_y_send = self.comm.Send_init(self.left_boundary_y_send, dest=self.prev_procy)
        self.req_left_boundary_y_recv = self.comm.Recv_init(self.left_boundary_y, source=self.prev_procy)

    def fill_right_boundary_x(self, sol):
        face = self.faces[self.face_name]
        self._pack_edge_state(face, sol[self.face_name], 0, self.right_boundary_x_send)
        self.req_right_boundary_x_recv.Start()
        self.req_right_boundary_x_send.Start()
        return self.req_right_boundary_x_recv

    def fill_left_boundary_x(self, sol):
        face = self.faces[self.face_name]
        self._pack_edge_state(face, sol[self.face_name], 2, self.left_boundary_x_send)
        self.req_left_boundary_x_recv.Start()
        self.req_left_boundary_x_send.Start()
        return self.req_left_boundary_x_recv

    def fill_right_boundary_y(self, sol):
        face = self.faces[self.face_name]
        self._pack_edge_state(face, sol[self.face_name], 1, self.right_boundary_y_send)
        self.req_right_boundary_y_recv.Start()
        self.req_right_boundary_y_send.Start()
        return self.req_right_boundary_y_recv

    def fill_left_boundary_y(self, sol):
        face = self.faces[self.face_name]
        self._pack_edge_state(face, sol[self.face_name], 3, self.left_boundary_y_send)
        self.req_left_boundary_y_recv.Start()
        self.req_left_boundary_y_send.Start()
        return self.req_left_boundary_y_recv

    def fill_boundaries(self, sol):
        reqs = [
            self.fill_left_boundary_x(sol),
            self.fill_right_boundary_x(sol),
            self.fill_left_boundary_y(sol),
            self.fill_right_boundary_y(sol),
            self.req_right_boundary_x_send,
            self.req_left_boundary_x_send,
            self.req_right_boundary_y_send,
            self.req_left_boundary_y_send,
        ]
        return reqs

    def recv_boundaries(self, reqs):
        for req in reqs:
            if req is not None:
                req.Wait()

        face = self.faces[self.face_name]
        self._assign_edge_state(face, 2, self.left_boundary_x)
        self._assign_edge_state(face, 0, self.right_boundary_x)
        self._assign_edge_state(face, 3, self.left_boundary_y)
        self._assign_edge_state(face, 1, self.right_boundary_y)

    def get_dt(self):
        return min(face.get_dt() for face in self.faces.values())

    def positivity_preserving_limiter(self, state, prev_state):
        for n in self.active_face_names:
            # if state[n][3].min() < 0:
                # print('0 detected')

            cell_means = (state[n][3] * self.faces[n].Jw).sum(axis=(2, 3)) / self.faces[n].Jw.sum(axis=(2, 3))
            cell_diffs = state[n][3] - cell_means[..., None, None]

            cell_mins = state[n][3].min(axis=(2, 3))
            diff_min = cell_mins - cell_means

            target_min = np.minimum(10.0, cell_means)
            needs_limiting = cell_mins <= target_min
            scale = np.ones_like(cell_means)
            scale[needs_limiting] = (
                target_min[needs_limiting] - cell_means[needs_limiting]
            ) / diff_min[needs_limiting]
            scale = np.clip(scale, 0.0, 1.0)

            # scale = (cell_mins > 5.0)

            state[n][3][:] = cell_means[..., None, None] + scale[..., None, None] * cell_diffs

            if cell_means.min() > 0:
                pass
                #print('Fixable')
            else:
                print("Negative height detected at time:", self.time)
                speed = np.sqrt(sum(state[n][i]**2 for i in range(3)))
                print("Maximum velocity:", speed.max())
                self.comm.Abort(1)

        return state


    def time_step(self, dt=None, order=3, forcing=None):

        if self.vorticity_diagnostic:
            self.time_list.append(self.time)
            self.energy_list.append(self.integrate(self.entropy()))
            self.enstrophy_list.append(self.integrate(self.enstrophy()))
            self.vorticity_list.append(self.integrate(self.vorticity()))
            self.mass_list.append(self.integrate(self.mass()))

        self.h = {n: f.h for n, f in self.faces.items()}  # only needs to be done once
        if dt is None:
            dt = self.get_dt()

        if order == 3:

            u = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}
            self.boundaries(u)
            k_1 = {n: self.faces[n].solve(*u[n], self.time, dt) for n in self.active_face_names}

            u_1 = {n: tuple(u[n][i] + dt * k_1[n][i] for i in range(4)) for n in self.active_face_names}
            u_1 = self.positivity_preserving_limiter(u_1, prev_state=u)
            self.boundaries(u_1)
            k_2 = {n: self.faces[n].solve(*u_1[n], self.time, dt) for n in self.active_face_names}

            u_2 = {n: tuple(0.75 * u[n][i] + 0.25 * (u_1[n][i] + dt * k_2[n][i]) for i in range(4)) for n in self.active_face_names}
            u_2 = self.positivity_preserving_limiter(u_2, prev_state=u_1)
            self.boundaries(u_2)
            k_3 = {n: self.faces[n].solve(*u_2[n], self.time, dt) for n in self.active_face_names}

            for n in self.active_face_names:
                self.faces[n].u = (self.faces[n].u + 2 * (u_2[n][0] + dt * k_3[n][0])) / 3
                self.faces[n].v = (self.faces[n].v + 2 * (u_2[n][1] + dt * k_3[n][1])) / 3
                self.faces[n].w = (self.faces[n].w + 2 * (u_2[n][2] + dt * k_3[n][2])) / 3
                self.faces[n].h = (self.faces[n].h + 2 * (u_2[n][3] + dt * k_3[n][3])) / 3

            u = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}
            u = self.positivity_preserving_limiter(u, prev_state=u_2)
            self.boundaries(u)

        elif order == 34:
            u = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}
            self.boundaries(u)
            k_1 = {n: self.faces[n].solve(*u[n], self.time, dt) for n in self.active_face_names}

            u_1 = {n: tuple(u[n][i] + 0.5 * dt * k_1[n][i] for i in range(4)) for n in self.active_face_names}
            u_1 = self.positivity_preserving_limiter(u_1, prev_state=u)
            self.boundaries(u_1)
            k_2 = {n: self.faces[n].solve(*u_1[n], self.time, dt) for n in self.active_face_names}

            u_2 = {n: tuple(u_1[n][i] + 0.5 * dt * k_2[n][i] for i in range(4)) for n in self.active_face_names}
            u_2 = self.positivity_preserving_limiter(u_2, prev_state=u_1)
            self.boundaries(u_2)
            k_3 = {n: self.faces[n].solve(*u_2[n], self.time, dt) for n in self.active_face_names}

            u_3 = {n: tuple((2 / 3) * u[n][i] + (1 / 3) * u_2[n][i] + (1 / 6) * dt * k_3[n][i] for i in range(4)) for n in self.active_face_names}
            u_3 = self.positivity_preserving_limiter(u_3, prev_state=u_2)
            self.boundaries(u_3)
            k_4 = {n: self.faces[n].solve(*u_3[n], self.time, dt) for n in self.active_face_names}

            for n in self.active_face_names:
                self.faces[n].u = u_3[n][0] + 0.5 * dt * k_4[n][0]
                self.faces[n].v = u_3[n][1] + 0.5 * dt * k_4[n][1]
                self.faces[n].w = u_3[n][2] + 0.5 * dt * k_4[n][2]
                self.faces[n].h = u_3[n][3] + 0.5 * dt * k_4[n][3]

            u = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}
            u = self.positivity_preserving_limiter(u, prev_state=u_3)
            self.boundaries(u)

        for n in self.active_face_names:
            self.faces[n].time += dt

        self.time += dt

    def plot_solution(self, ax, vmin=None, vmax=None, plot_func=None, dim=3, cmap='nipy_spectral'):

        if dim == 3:
            return [self.faces[name].plot_solution(ax, vmin, vmax, plot_func, dim, cmap) for name in self.active_face_names]
        elif dim == 2:
            return [self.faces[name].plot_solution(ax, vmin, vmax, plot_func, dim, cmap) for name in self.active_face_names if name != 'zn']
        else:
            raise ValueError(f"dim: expected one of 2, 3. Found {dim}.")

    def evaluate_latlong(self, lat, long, coeffs, degrees=False):
        """
        Evaluate nodal DG coefficients at latitude/longitude points.

        coeffs may be a dict keyed by face name, or a six-face array in
        self.face_names order: ['zp', 'zn', 'xp', 'xn', 'yp', 'yn'].
        """
        lat, long = np.broadcast_arrays(lat, long)
        if degrees:
            lat = np.deg2rad(lat)
            long = np.deg2rad(long)

        lat_shape = lat.shape
        lat_flat = lat.ravel()
        long_flat = long.ravel()

        x, y, z = lat_long_to_cartesian(lat_flat, long_flat, radius=next(iter(self.faces.values())).geometry.radius)
        face_names = face_name_from_cartesian(x, y, z).ravel()

        out = None
        for name in self.active_face_names:
            mask = face_names == name
            if not mask.any():
                continue

            face = self.faces[name]
            x1, y1 = face.geometry.to_cubed_sphere(x[mask], y[mask], z[mask])
            face_values = face.evaluate(x1, y1, self._face_coeffs(coeffs, name))

            if out is None:
                out = np.empty(lat_flat.shape, dtype=face_values.dtype)
            out[mask] = face_values

        if out is None:
            out = np.empty(lat_flat.shape)

        return out.reshape(lat_shape)

    def _face_coeffs(self, coeffs, name):
        if isinstance(coeffs, dict):
            return coeffs[name]
        return coeffs[self.face_names.index(name)]

    def _coeffs_by_face(self, coeffs):
        if coeffs is None:
            return {name: self.faces[name].h for name in self.active_face_names}
        if isinstance(coeffs, str):
            return {name: getattr(self.faces[name], coeffs) for name in self.active_face_names}
        if isinstance(coeffs, dict):
            return coeffs
        return {
            name: coeffs[self.face_names.index(name)]
            for name in self.active_face_names
        }

    def evaluate_cartesian(self, x, y, z, coeffs):
        """
        Evaluate scalar DG coefficients at Cartesian points on the sphere.

        ``coeffs`` accepts the same forms as ``evaluate_latlong``: a dict keyed
        by face name, or an array whose first axis follows ``self.face_names``.
        """
        x, y, z = np.broadcast_arrays(x, y, z)
        out = self._evaluate_cartesian_many(x, y, z, [coeffs])[0]
        return out

    def _evaluate_cartesian_many(self, x, y, z, coeffs_list):
        coeffs_by_face = [self._coeffs_by_face(coeffs) for coeffs in coeffs_list]
        return self._evaluate_cartesian_many_by_face(x, y, z, coeffs_by_face)

    def _evaluate_cartesian_many_by_face(self, x, y, z, coeffs_by_face):
        if self.parallel:
            raise NotImplementedError(
                "Sphere-wide Cartesian evaluation is only available in serial runs."
            )

        x, y, z = np.broadcast_arrays(x, y, z)
        out_shape = x.shape
        x_flat = x.ravel()
        y_flat = y.ravel()
        z_flat = z.ravel()
        face_names = face_name_from_cartesian(x_flat, y_flat, z_flat).ravel()

        outputs = [None for _ in coeffs_by_face]

        for name in self.active_face_names:
            mask = face_names == name
            if not mask.any():
                continue

            face = self.faces[name]
            x1, y1 = face.geometry.to_cubed_sphere(x_flat[mask], y_flat[mask], z_flat[mask])
            for idx, coeffs in enumerate(coeffs_by_face):
                values = face.evaluate(x1, y1, coeffs[name])
                if outputs[idx] is None:
                    outputs[idx] = np.empty(x_flat.shape, dtype=values.dtype)
                outputs[idx][mask] = values

        for idx, out in enumerate(outputs):
            if out is None:
                outputs[idx] = np.empty(x_flat.shape)
            outputs[idx] = outputs[idx].reshape(out_shape)

        return outputs

    def _siac_extended_scalar_coeffs(self, face, coeffs_by_face, pad_x, pad_y):
        ext_shape = (face.ny + 2 * pad_y, face.nx + 2 * pad_x, face.n, face.n)
        coeffs_ext = np.empty(ext_shape, dtype=face.dtype)
        coeffs_ext[
            pad_y:pad_y + face.ny,
            pad_x:pad_x + face.nx,
        ] = coeffs_by_face[face.name]

        if pad_x == 0 and pad_y == 0:
            return coeffs_ext

        x1, y1 = face._siac_extended_coordinates(pad_x, pad_y)
        ghost_mask, x_nudge, y_nudge = face._siac_extension_ghost_mask(pad_x, pad_y)
        node_mask = np.broadcast_to(ghost_mask[..., None, None], ext_shape)
        x_eval = x1[node_mask] + np.broadcast_to(x_nudge[..., None, None], ext_shape)[node_mask]
        y_eval = y1[node_mask] + np.broadcast_to(y_nudge[..., None, None], ext_shape)[node_mask]
        x, y, z = face.geometry.to_cartesian(x_eval, y_eval)
        coeffs_ext[node_mask] = self._evaluate_cartesian_many_by_face(
            x, y, z, [coeffs_by_face]
        )[0]
        return coeffs_ext

    def _siac_extended_covariant_velocity(self, face, pad_x, pad_y):
        ext_shape = (face.ny + 2 * pad_y, face.nx + 2 * pad_x, face.n, face.n)
        u_ext = np.empty(ext_shape, dtype=face.dtype)
        v_ext = np.empty(ext_shape, dtype=face.dtype)

        u_cov, v_cov = face.local_covariant_velocity()
        u_ext[pad_y:pad_y + face.ny, pad_x:pad_x + face.nx] = u_cov
        v_ext[pad_y:pad_y + face.ny, pad_x:pad_x + face.nx] = v_cov

        if pad_x == 0 and pad_y == 0:
            return u_ext, v_ext

        x1, y1 = face._siac_extended_coordinates(pad_x, pad_y)
        ghost_mask, x_nudge, y_nudge = face._siac_extension_ghost_mask(pad_x, pad_y)
        node_mask = np.broadcast_to(ghost_mask[..., None, None], ext_shape)
        x_eval = x1[node_mask] + np.broadcast_to(x_nudge[..., None, None], ext_shape)[node_mask]
        y_eval = y1[node_mask] + np.broadcast_to(y_nudge[..., None, None], ext_shape)[node_mask]

        x, y, z = face.geometry.to_cartesian(x_eval, y_eval)
        velocity_coeffs = [
            {name: source_face.u for name, source_face in self.faces.items()},
            {name: source_face.v for name, source_face in self.faces.items()},
            {name: source_face.w for name, source_face in self.faces.items()},
        ]
        u_sample, v_sample, w_sample = self._evaluate_cartesian_many_by_face(
            x, y, z, velocity_coeffs
        )
        (
            dxdx1,
            dxdy1,
            _,
            dydx1,
            dydy1,
            _,
            dzdx1,
            dzdy1,
            _,
        ) = face.geometry.covariant_basis(x_eval, y_eval)

        u_ext[node_mask] = u_sample * dxdx1 + v_sample * dydx1 + w_sample * dzdx1
        v_ext[node_mask] = u_sample * dxdy1 + v_sample * dydy1 + w_sample * dzdy1
        return u_ext, v_ext

    def siac_filter(
        self,
        coeffs=None,
        *,
        derivative=(0, 0),
        width=None,
        scale=1.0,
        kernel=None,
        quadrature_order=None,
        boundary="sphere",
    ):
        """
        Apply a tensor-product SIAC filter to scalar DG coefficients.

        ``boundary="sphere"`` samples through cubed-sphere panel seams in
        serial. ``boundary="face"`` uses each face's fast local tensor-product
        matrix path and clips samples at that face's edge.
        """
        if kernel is None:
            kernel = siac_kernel(next(iter(self.faces.values())).poly_order)
        if quadrature_order is None:
            quadrature_order = max(next(iter(self.faces.values())).poly_order + 3, 6)

        if boundary == "face":
            coeffs_by_face = self._coeffs_by_face(coeffs)
            return {
                name: face.siac_filter(
                    coeffs_by_face[name],
                    derivative=derivative,
                    width=width,
                    scale=scale,
                    kernel=kernel,
                    quadrature_order=quadrature_order,
                )
                for name, face in self.faces.items()
            }
        if boundary != "sphere":
            raise ValueError(
                f"boundary: expected one of ['sphere', 'face']. Found {boundary}."
            )
        return self._siac_filter_sphere(
            coeffs,
            derivative=derivative,
            width=width,
            scale=scale,
            kernel=kernel,
            quadrature_order=quadrature_order,
        )

    def _siac_filter_sphere(
        self,
        coeffs,
        *,
        derivative,
        width,
        scale,
        kernel,
        quadrature_order,
    ):
        if self.parallel:
            raise NotImplementedError(
                "Sphere-wide SIAC filtering is only available in serial runs. "
                "Use boundary='face' for local per-rank post-processing."
            )

        coeffs_by_face = self._coeffs_by_face(coeffs)
        dx_order, dy_order = derivative
        out = {}

        for name, face in self.faces.items():
            Hx, Hy = face._siac_widths(width, scale)
            pad_x, pad_y = face._siac_padding(Hx, Hy, kernel)
            coeffs_ext = self._siac_extended_scalar_coeffs(
                face, coeffs_by_face, pad_x, pad_y
            )
            x_matrix = face._siac_extended_matrix(
                "x", dx_order, Hx, kernel, quadrature_order, pad_x
            )
            y_matrix = face._siac_extended_matrix(
                "y", dy_order, Hy, kernel, quadrature_order, pad_y
            )
            out[name] = face._apply_siac_matrices(
                coeffs_ext, x_matrix, y_matrix
            ).astype(face.dtype, copy=False)

        return out

    def siac_vorticity(
        self,
        *,
        width=None,
        scale=1.0,
        kernel=None,
        quadrature_order=None,
        boundary="sphere",
        include_coriolis=True,
    ):
        """
        Compute vorticity using SIAC derivative filters on covariant velocity.
        """
        if kernel is None:
            kernel = siac_kernel(next(iter(self.faces.values())).poly_order)
        if quadrature_order is None:
            quadrature_order = max(next(iter(self.faces.values())).poly_order + 3, 6)

        if boundary == "face":
            return {
                name: face.siac_vorticity(
                    width=width,
                    scale=scale,
                    kernel=kernel,
                    quadrature_order=quadrature_order,
                    include_coriolis=include_coriolis,
                )
                for name, face in self.faces.items()
            }
        if boundary != "sphere":
            raise ValueError(
                f"boundary: expected one of ['sphere', 'face']. Found {boundary}."
            )
        if self.parallel:
            raise NotImplementedError(
                "Sphere-wide SIAC vorticity is only available in serial runs. "
                "Use boundary='face' for local per-rank post-processing."
            )

        out = {}
        for name, face in self.faces.items():
            Hx, Hy = face._siac_widths(width, scale)
            pad_x, pad_y = face._siac_padding(Hx, Hy, kernel)
            u_ext, v_ext = self._siac_extended_covariant_velocity(face, pad_x, pad_y)
            x_filter = face._siac_extended_matrix(
                "x", 0, Hx, kernel, quadrature_order, pad_x
            )
            y_filter = face._siac_extended_matrix(
                "y", 0, Hy, kernel, quadrature_order, pad_y
            )
            x_derivative = face._siac_extended_matrix(
                "x", 1, Hx, kernel, quadrature_order, pad_x
            )
            y_derivative = face._siac_extended_matrix(
                "y", 1, Hy, kernel, quadrature_order, pad_y
            )
            dvy_dx = face._apply_siac_matrices(v_ext, x_derivative, y_filter)
            dux_dy = face._apply_siac_matrices(u_ext, x_filter, y_derivative)
            vort = (dvy_dx - dux_dy) / face.local_surface_jacobian()
            if include_coriolis:
                vort = vort + face.f
            out[name] = vort.astype(face.dtype, copy=False)

        return out

    def triangular_plot(self, ax, vmin=None, vmax=None, plot_func=None, cmap='nipy_spectral', latlong=False, n=None, lines=False, levels=None):
        data = [plot_func(face).ravel() for face in self.faces.values()]
        if not latlong:
            x_coords = [face.xs.ravel() for face in self.faces.values()]
            y_coords = [face.ys.ravel() for face in self.faces.values()]
            z_coords = [face.zs.ravel() for face in self.faces.values()]
            z_coords = np.concatenate(z_coords)
        else:
            x_coords = [face.geometry.lat_long(face.xs, face.ys, face.zs)[1].ravel() for face in self.faces.values()]
            y_coords = [face.geometry.lat_long(face.xs, face.ys, face.zs)[0].ravel() for face in self.faces.values()]

        data = np.concatenate(data)
        y_coords = np.concatenate(y_coords)
        x_coords = np.concatenate(x_coords)

        if latlong:
            y_coords *= 180 / np.pi
            x_coords *= 180 / np.pi
            mask = (10 <= y_coords) & (y_coords <= 80)
        else:
            mask = z_coords > 0

        if n is None:
            n = int(0.5 * (vmax - vmin) / 1e-5)

        if levels is None:
            levels = np.linspace(vmin, vmax, n)

        if lines:
            out = ax.tricontour(
                -y_coords[mask], x_coords[mask], data[mask], colors='black',
                levels=levels, negative_linestyles='dashed', linewidths=0.5
            )
        else:
            out = ax.tricontourf(
                -y_coords[mask], x_coords[mask], data[mask], cmap=cmap, levels=levels
            )
        return [out]

    def latlong_triangular_plot(self, ax, vmin=None, vmax=None, plot_func=None, cmap='nipy_spectral', n=None, lines=False, levels=None):
        data = [plot_func(face).ravel() for face in self.faces.values()]

        x_coords = [face.geometry.lat_long(face.xs, face.ys, face.zs)[1].ravel() for face in self.faces.values()]
        y_coords = [face.geometry.lat_long(face.xs, face.ys, face.zs)[0].ravel() for face in self.faces.values()]

        data = np.concatenate(data)
        y_coords = np.concatenate(y_coords)
        x_coords = np.concatenate(x_coords)

        y_coords *= 180 / np.pi
        x_coords *= 180 / np.pi
        # mask = (10 <= y_coords) & (y_coords <= 80)
        mask = np.ones_like(x_coords) > 0


        if levels is None:
            if n is None:
                n = int(0.5 * (vmax - vmin) / 1e-5)

            levels = np.linspace(vmin, vmax, n)

        if lines:
            out = ax.tricontour(
                x_coords[mask], y_coords[mask], data[mask], colors='black',
                levels=levels, negative_linestyles='dashed', linewidths=0.5, inline=True
            )
            fntsz = 8
            label_set = list(levels[::5])
            if not levels[1] in label_set:
                label_set.append(levels[1])
            if not levels[-3] in label_set:
                label_set.append(levels[-3])

            ax.clabel(out, label_set, inline=True, fontsize=fntsz)
            # ax.clabel(out, out.levels[::5], inline=True, fontsize=fntsz)
            # ax.clabel(out, out.levels[:1], inline=True, fontsize=fntsz)
            # ax.clabel(out, out.levels[-1:], inline=True, fontsize=fntsz)
        else:
            out = ax.tricontourf(
                x_coords[mask], y_coords[mask], data[mask], cmap=cmap, levels=levels, vmin=vmin, vmax=vmax,
            )
        return [out]

    def integrate(self, q):
        return sum(f.integrate(q[n]) for n, f in self.faces.items())

    def entropy(self):
        return {n: f.entropy() for n, f in self.faces.items()}

    def enstrophy(self):
        return {n: f.enstrophy() for n, f in self.faces.items()}

    def vorticity(self):
        return {n: f.h * f.q(f.u, f.v, f.w, f.h) for n, f in self.faces.items()}

    def mass(self):
        return {n: f.h for n, f in self.faces.items()}

    def _restart_tile(self, var):
        face = self.faces[self.face_name]
        return (
            self.face_name,
            self.x_proc_idx,
            self.y_proc_idx,
            np.ascontiguousarray(_to_numpy(getattr(face, var))),
        )

    def _restart_tile_slice(self, face):
        y0 = face.y_proc_idx * face.ny
        x0 = face.x_proc_idx * face.nx
        return np.s_[y0:y0 + face.ny, x0:x0 + face.nx, :, :]

    def _assemble_restart_face(self, tiles):
        expected_tiles = self.nprocx * self.nprocy
        if len(tiles) != expected_tiles:
            raise ValueError(
                f"Restart gather expected {expected_tiles} tiles per face; found {len(tiles)}."
            )

        local_shape = tiles[0][2].shape
        if len(local_shape) != 4:
            raise ValueError(f"Restart tile must be 4D; found shape {local_shape}.")

        local_ny, local_nx, n_eta, n_xi = local_shape
        out = np.empty(
            (local_ny * self.nprocy, local_nx * self.nprocx, n_eta, n_xi),
            dtype=tiles[0][2].dtype,
        )

        seen = set()
        for x_idx, y_idx, tile in tiles:
            if not (0 <= x_idx < self.nprocx and 0 <= y_idx < self.nprocy):
                raise ValueError(
                    f"Restart tile index ({x_idx}, {y_idx}) is outside "
                    f"({self.nprocx}, {self.nprocy})."
                )
            if tile.shape != local_shape:
                raise ValueError(
                    f"Restart tiles for a face must all have shape {local_shape}; "
                    f"found {tile.shape}."
                )
            key = (x_idx, y_idx)
            if key in seen:
                raise ValueError(f"Duplicate restart tile index {key}.")
            seen.add(key)

            y0 = y_idx * local_ny
            x0 = x_idx * local_nx
            out[y0:y0 + local_ny, x0:x0 + local_nx] = tile

        if len(seen) != expected_tiles:
            raise ValueError(
                f"Restart gather found {len(seen)} unique tiles; expected {expected_tiles}."
            )
        return out

    def _save_parallel_restart_var(self, var, fn_template, directory):
        gathered = self.comm.gather(self._restart_tile(var), root=0)
        if self.rank != 0:
            return

        by_face = {name: [] for name in self.face_names}
        for name, x_idx, y_idx, tile in gathered:
            by_face[name].append((x_idx, y_idx, _to_numpy(tile)))

        for name in self.face_names:
            data = self._assemble_restart_face(by_face[name])
            np.save(self.make_fp(var, name, fn_template, directory), data)

    def _load_restart_data(self, var, name, fn_template, directory):
        data = np.load(self.make_fp(var, name, fn_template, directory))
        face = self.faces[name]
        expected_shape = (face.global_ny, face.global_nx, face.n, face.n)
        if data.shape != expected_shape:
            raise ValueError(
                f"Restart {var}_{name} has shape {data.shape}; expected {expected_shape}."
            )
        return data[self._restart_tile_slice(face)]

    def _restart_barrier(self):
        if self.comm is None:
            return
        if hasattr(self.comm, "Barrier"):
            self.comm.Barrier()
        elif hasattr(self.comm, "barrier"):
            self.comm.barrier()

    def save_restart(self, fn_template, directory):
        vars = ['u', 'v', 'w', 'h']
        if self.parallel:
            for var in vars:
                self._save_parallel_restart_var(var, fn_template, directory)
            self._restart_barrier()
            return

        state = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}
        for name in self.active_face_names:
            for i in range(len(vars)):
                fp = self.make_fp(vars[i], name, fn_template, directory)
                data = _to_numpy(state[name][i])
                np.save(fp, data)

    @staticmethod
    def make_fp(var, name, fn_template, directory):
        fn = f"{var}_{name}_{fn_template}"
        fp = os.path.join(directory, fn)
        return fp

    def load_restart(self, fn_template, directory):
        for name in self.active_face_names:
            vars = ['u', 'v', 'w', 'h']
            data = [self._load_restart_data(vars[i], name, fn_template, directory) for i in range(len(vars))]
            b = self.faces[name].b
            self.faces[name].set_initial_condition(*data)
            self.faces[name].b = b

        self.boundaries()

    def save_diagnostics(self, fn_template, directory):
        diagnostics = np.stack([self.time_list, self.energy_list, self.enstrophy_list, self.vorticity_list, self.mass_list])
        fp = os.path.join(directory, f"diagnostics_{fn_template}")
        np.save(fp, diagnostics)

    def plot_diagnostics(self, fn_template, directory, fig_int, label):
        from matplotlib import pyplot as plt

        diagnostics = np.load(os.path.join(directory, f"diagnostics_{fn_template}"))
        times = diagnostics[0] / (24 * 3600)
        entropy = diagnostics[1]
        enstrophy = diagnostics[2]
        vorticity = diagnostics[3]
        mass = diagnostics[4]

        print('vorticity:', vorticity[0])

        plt.figure(fig_int, figsize=(7, 7))

        tunit = ' (days)'
        plt.suptitle("Conservation errors")

        ax = plt.subplot(2, 2, 1)
        ax.set_ylabel("Energy error (normalized)")
        ax.set_xticks([], [])
        ax.plot(times, (entropy - entropy[0]) / entropy[0], label=label)
        ax.set_yscale('symlog', linthresh=1e-15)
        ax.grid(True, which='both')

        ax = plt.subplot(2, 2, 2)
        ax.set_ylabel("Mass error (normalized)")
        ax.set_xticks([], [])
        ax.plot(times, (mass - mass[0]) / mass[0], label=label)
        ax.set_yscale('symlog', linthresh=1e-16)
        ax.grid(True, which='both')

        ax = plt.subplot(2, 2, 3)
        ax.set_ylabel("Enstrophy error (normalized)")
        ax.set_xlabel("Time" + tunit)
        ax.plot(times, (enstrophy - enstrophy[0]) / enstrophy[0], label=label)
        ax.set_yscale('symlog', linthresh=1e-15)
        ax.grid(True, which='both')

        ax = plt.subplot(2, 2, 4)
        plt.ylabel("Vorticity error")
        plt.xlabel("Time" + tunit)
        plt.plot(times, (vorticity - vorticity[0]), label=label)
        ax.set_yscale('symlog', linthresh=1e-16)
        ax.grid(True, which='both')

        plt.legend()
        plt.tight_layout()


class DGCubedSphereFaceNumpy:
    """
    One face of the cubed sphere.
    """

    def __init__(
            self, name, poly_order, nx, ny, g, f, radius, eps, device='cpu',
            solution=None, a=0.0, ah=0.0, dtype=np.float64, bc='wall',
            flux_type="standard",
            x_proc_idx=0, y_proc_idx=0, nprocx=1, nprocy=1,
            global_nx=None, global_ny=None, x_min=None, x_max=None, y_min=None, y_max=None,
            **kwargs
        ):

        _reject_legacy_flux_kwargs(kwargs)
        valid_names = ['zp', 'zn', 'xp', 'xn', 'yp', 'yn']
        if not name in valid_names:
            raise ValueError(f'name: expected one of: {valid_names}. Found {name}.')
        flux_type = _validate_flux_type(flux_type)
        self.name = name
        self.time = 0
        self.poly_order = poly_order
        self.u = None
        self.v = None
        self.h = None
        self.b = None
        self.g = g
        self.eps = eps
        self.a = a
        self.ah = ah
        self.solution = solution
        self.dtype = dtype
        self.xperiodic = self.yperiodic = False
        self.bc = bc
        self.geometry = EquiangularFace(name, radius=radius)
        self.connections = self.geometry.connections
        self.flux_type = flux_type

        [xs_1d, w_x] = gll(poly_order, iterative=True)
        [y_1d, w_y] = gll(poly_order, iterative=True)
        self.gll_nodes = xs_1d

        explicit_domain = any(
            bound is not None for bound in (x_min, x_max, y_min, y_max)
        )
        if explicit_domain and any(
            bound is None for bound in (x_min, x_max, y_min, y_max)
        ):
            raise ValueError(
                "x_min, x_max, y_min, and y_max must all be provided together."
            )

        if explicit_domain:
            local_nx = nx - 1
            local_ny = ny - 1
            if global_nx is None:
                global_nx = local_nx
            if global_ny is None:
                global_ny = local_ny
        else:
            global_nx = nx - 1 if global_nx is None else global_nx
            global_ny = ny - 1 if global_ny is None else global_ny
            if global_nx % nprocx != 0:
                raise ValueError(
                    f"nx - 1 must be divisible by nprocx; got nx={nx}, nprocx={nprocx}."
                )
            if global_ny % nprocy != 0:
                raise ValueError(
                    f"ny - 1 must be divisible by nprocy; got ny={ny}, nprocy={nprocy}."
                )
            local_nx = global_nx // nprocx
            local_ny = global_ny // nprocy
            x_start = x_proc_idx * local_nx
            y_start = y_proc_idx * local_ny
            dx = 1.0 / global_nx
            dy = 1.0 / global_ny
            x_min = -0.5 + x_start * dx
            x_max = -0.5 + (x_start + local_nx) * dx
            y_min = -0.5 + y_start * dy
            y_max = -0.5 + (y_start + local_ny) * dy

        if local_nx <= 0 or local_ny <= 0:
            raise ValueError(
                f"Local face dimensions must be positive; got local_nx={local_nx}, local_ny={local_ny}."
            )

        self.global_nx = global_nx
        self.global_ny = global_ny
        self.x_proc_idx = x_proc_idx
        self.y_proc_idx = y_proc_idx
        self.nprocx = nprocx
        self.nprocy = nprocy

        xs = np.linspace(x_min, x_max, local_nx + 1)
        ys = np.linspace(y_min, y_max, local_ny + 1)
        self.x_min = xs[0]
        self.y_min = ys[0]
        self.x_max = xs[-1]
        self.y_max = ys[-1]

        lx = np.mean(np.diff(xs))
        ly = np.mean(np.diff(ys))
        self.lx = lx
        self.ly = ly

        self.cdt = eps * radius * min(lx, ly) / (2 * poly_order + 1)  # this should be multiplied by pi / (2 * sqrt(2)) = 1.11... but eh a slightly smaller time step can't hurt

        w_x, w_y = np.meshgrid(w_x, w_y)
        self.weights_x = w_x[0][None, None, ...]
        self.weights = w_x * w_y

        x1, y1 = np.meshgrid(xs_1d, y_1d)

        x1 = (1 + x1) * lx / 2
        y1 = (1 + y1) * ly / 2

        # cube face coordinates
        shape = (len(ys) - 1, len(xs) - 1)
        self.x1 = x1[None, None, ...] + xs[:-1][None, :, None, None] * np.ones(shape + (1, 1))
        self.y1 = y1[None, None, ...] + ys[:-1][:, None, None, None] * np.ones(shape + (1, 1))

        # 3D cartesian coordinates on surface of sphere
        self.xs, self.ys, self.zs = self.geometry.to_cartesian(self.x1, self.y1)
        lat, long = self.geometry.lat_long(self.xs, self.ys, self.zs)
        self.f = 2 * f * np.sin(lat)

        self.l1d = lagrange1st(poly_order, xs_1d)
        n = poly_order + 1

        self.n = n
        self.device = device
        self.weights = self.weights.astype(self.dtype, copy=False)
        self.weights_x = self.weights_x.astype(self.dtype, copy=False)
        self.edge_weights = self.weights_x.ravel().astype(self.dtype, copy=False)
        self.endpoint_weight = self.edge_weights[-1]
        self.nx = local_nx
        self.ny = local_ny
        self.D = self.l1d.astype(self.dtype, copy=False)
        self._siac_matrix_cache = {}
        self._siac_extension_coord_cache = {}

        dxdx1, dxdy1, dxdz1, dydx1, dydy1, dydz1, dzdx1, dzdy1, dzdz1 = self.geometry.covariant_basis(self.x1, self.y1)
        self.dxdxi = dxdx1.astype(self.dtype, copy=False) * lx / 2
        self.dxdeta = dxdy1.astype(self.dtype, copy=False) * ly / 2
        self.dxdzeta = self.xs.astype(self.dtype, copy=False) / radius
        #
        self.dydxi = dydx1.astype(self.dtype, copy=False) * lx / 2
        self.dydeta = dydy1.astype(self.dtype, copy=False) * ly / 2
        self.dydzeta = self.ys.astype(self.dtype, copy=False) / radius
        #
        self.dzdxi = dzdx1.astype(self.dtype, copy=False) * lx / 2
        self.dzdeta = dzdy1.astype(self.dtype, copy=False) * ly / 2
        self.dzdzeta = self.zs.astype(self.dtype, copy=False) / radius

        cross = cross_product(
            [self.dxdxi, self.dydxi, self.dzdxi], [self.dxdzeta, self.dydzeta, self.dzdzeta]
        )
        self.J_vertface = _norm_l2(cross)

        cross = cross_product(
            [self.dxdeta, self.dydeta, self.dzdeta], [self.dxdzeta, self.dydzeta, self.dzdzeta]
        )
        self.J_horzface = _norm_l2(cross)

        self.J = self.dxdxi * (self.dydeta * self.dzdzeta - self.dydzeta * self.dzdeta)
        self.J += self.dydxi * (self.dzdeta * self.dxdzeta - self.dzdzeta * self.dxdeta)
        self.J += self.dzdxi * (self.dxdeta * self.dydzeta - self.dxdzeta * self.dydeta)
        self.J = self.J

        self.dxidx = (self.dydeta * self.dzdzeta - self.dydzeta * self.dzdeta) / self.J
        self.dxidy = (self.dzdeta * self.dxdzeta - self.dzdzeta * self.dxdeta) / self.J
        self.dxidz = (self.dxdeta * self.dydzeta - self.dxdzeta * self.dydeta) / self.J

        self.detadx = (self.dydzeta * self.dzdxi - self.dydxi * self.dzdzeta) / self.J
        self.detady = (self.dzdzeta * self.dxdxi - self.dzdxi * self.dxdzeta) / self.J
        self.detadz = (self.dxdzeta * self.dydxi - self.dxdxi * self.dydzeta) / self.J

        self.dzetadx = (self.dydxi * self.dzdeta - self.dydeta * self.dzdxi) / self.J
        self.dzetady = (self.dzdxi * self.dxdeta - self.dzdeta * self.dxdxi) / self.J
        self.dzetadz = (self.dxdxi * self.dydeta - self.dxdeta * self.dydxi) / self.J

        self.dxyzdzeta_norm = _norm_l2([self.dxdzeta, self.dydzeta, self.dzdzeta])
        self.grad_zeta_norm = _norm_l2([self.dzetadx, self.dzetady, self.dzetadz])

        self.kx = self.dzetadx / self.grad_zeta_norm
        self.ky = self.dzetady / self.grad_zeta_norm
        self.kz = self.dzetadz / self.grad_zeta_norm

        self.kx = self.xs.astype(self.dtype, copy=False) / radius
        self.ky = self.ys.astype(self.dtype, copy=False) / radius
        self.kz = self.zs.astype(self.dtype, copy=False) / radius

        self.J_xi = np.sqrt(self.dxidx ** 2 + self.dxidy ** 2 + self.dxidz ** 2)
        self.J_eta = np.sqrt(self.detadx ** 2 + self.detady ** 2 + self.detadz ** 2)

        self.eta_x_up, self.eta_x_down = self.make_up_down_arrays(self.detadx / self.J_eta)
        self.eta_y_up, self.eta_y_down = self.make_up_down_arrays(self.detady / self.J_eta)
        self.eta_z_up, self.eta_z_down = self.make_up_down_arrays(self.detadz / self.J_eta)

        self.xi_x_right, self.xi_x_left = self.make_left_right_arrays(self.dxidx / self.J_xi)
        self.xi_y_right, self.xi_y_left = self.make_left_right_arrays(self.dxidy / self.J_xi)
        self.xi_z_right, self.xi_z_left = self.make_left_right_arrays(self.dxidz / self.J_xi)

        self.dxidx_up, self.dxidx_down = self.make_up_down_arrays(self.dxidx)
        self.dxidy_up, self.dxidy_down = self.make_up_down_arrays(self.dxidy)
        self.dxidz_up, self.dxidz_down = self.make_up_down_arrays(self.dxidz)

        self.dxidx_right, self.dxidx_left = self.make_left_right_arrays(self.dxidx)
        self.dxidy_right, self.dxidy_left = self.make_left_right_arrays(self.dxidy)
        self.dxidz_right, self.dxidz_left = self.make_left_right_arrays(self.dxidz)

        self.detadx_up, self.detadx_down = self.make_up_down_arrays(self.detadx)
        self.detady_up, self.detady_down = self.make_up_down_arrays(self.detady)
        self.detadz_up, self.detadz_down = self.make_up_down_arrays(self.detadz)

        self.detadx_right, self.detadx_left = self.make_left_right_arrays(self.detadx)
        self.detady_right, self.detady_left = self.make_left_right_arrays(self.detady)
        self.detadz_right, self.detadz_left = self.make_left_right_arrays(self.detadz)

        self.dxdxi_up, self.dxdxi_down = self.make_up_down_arrays(self.dxdxi)
        self.dydxi_up, self.dydxi_down = self.make_up_down_arrays(self.dydxi)
        self.dzdxi_up, self.dzdxi_down = self.make_up_down_arrays(self.dzdxi)

        self.dxdxi_right, self.dxdxi_left = self.make_left_right_arrays(self.dxdxi)
        self.dydxi_right, self.dydxi_left = self.make_left_right_arrays(self.dydxi)
        self.dzdxi_right, self.dzdxi_left = self.make_left_right_arrays(self.dzdxi)

        self.dxdeta_up, self.dxdeta_down = self.make_up_down_arrays(self.dxdeta)
        self.dydeta_up, self.dydeta_down = self.make_up_down_arrays(self.dydeta)
        self.dzdeta_up, self.dzdeta_down = self.make_up_down_arrays(self.dzdeta)

        self.dxdeta_right, self.dxdeta_left = self.make_left_right_arrays(self.dxdeta)
        self.dydeta_right, self.dydeta_left = self.make_left_right_arrays(self.dydeta)
        self.dzdeta_right, self.dzdeta_left = self.make_left_right_arrays(self.dzdeta)

        self.k_cov_norm = _norm_l2([self.dxdzeta, self.dydzeta, self.dzdzeta])
        self.k_cov_norm_w = self.k_cov_norm * self.weights
        self.Jw = self.J * self.weights
        self.inv_Jw = 1.0 / self.Jw
        edge_weights = self.edge_weights[None, None, :]
        self.vert_upper_edge_factor = self.J_vertface[:, :, -1, :] * edge_weights * self.inv_Jw[:, :, -1, :]
        self.vert_lower_edge_factor = self.J_vertface[:, :, 0, :] * edge_weights * self.inv_Jw[:, :, 0, :]
        self.horz_right_edge_factor = self.J_horzface[:, :, :, -1] * edge_weights * self.inv_Jw[:, :, :, -1]
        self.horz_left_edge_factor = self.J_horzface[:, :, :, 0] * edge_weights * self.inv_Jw[:, :, :, 0]

    def make_left_right_arrays(self, arr):
        right_arr = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.dtype)
        left_arr = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.dtype)

        right_arr[:, :-1] = arr[:, :, :, 0]
        right_arr[:, -1] = arr[:, -1, :, -1]

        left_arr[:, 1:] = arr[:, :, :, -1]
        left_arr[:, 0] = arr[:, 0, :, 0]

        return right_arr, left_arr

    def make_up_down_arrays(self, arr):
        up_arr = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.dtype)
        down_arr = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.dtype)

        up_arr[:-1] = arr[:, :, 0, :]
        up_arr[-1] = arr[-1, :, -1]

        down_arr[1:] = arr[:, :, -1, :]
        down_arr[0] = arr[0, :, 0]

        return up_arr, down_arr

    def ddxi(self, arr):
        return np.matmul(arr, self.D)

    def ddeta(self, arr):
        return np.einsum('ca,...cb->...ab', self.D, arr)

    def weak_ddxi(self, arr):
        return np.matmul(arr * self.k_cov_norm_w, self.D.T)

    def weak_ddeta(self, arr):
        return np.einsum('ac,...cb->...ab', self.D, arr * self.k_cov_norm_w)

    def boundaries(self, u, v, w, h, t):

        self.u_up[:-1] = u[:, :, 0, :]
        self.u_down[1:] = u[:, :, -1, :]
        self.u_right[:, :-1] = u[:, :, :, 0]
        self.u_left[:, 1:] = u[:, :, :, -1]

        self.v_up[:-1] = v[:, :, 0, :]
        self.v_down[1:] = v[:, :, -1, :]
        self.v_right[:, :-1] = v[:, :, :, 0]
        self.v_left[:, 1:] = v[:, :, :, -1]

        self.w_up[:-1] = w[:, :, 0, :]
        self.w_down[1:] = w[:, :, -1, :]
        self.w_right[:, :-1] = w[:, :, :, 0]
        self.w_left[:, 1:] = w[:, :, :, -1]

        self.h_up[:-1] = h[:, :, 0, :]
        self.h_down[1:] = h[:, :, -1, :]
        self.h_right[:, :-1] = h[:, :, :, 0]
        self.h_left[:, 1:] = h[:, :, :, -1]

        # wall boundary condition
        if self.bc.lower() == 'wall':
            self.h_up[-1] = h[-1, :, -1, :]
            self.h_down[0] = h[0, :, 0, :]
            self.h_right[:, -1] = h[:, -1, :, -1]
            self.h_left[:, 0] = h[:, 0, :, 0]

            u_, v_ = self.phys_to_contra(u, v, w)
            u_, v_, w_, = self.contra_to_phys(u_, 0 * v_)
            self.u_down[0], self.v_down[0], self.w_down[0] = u_[0, :, 0, :], v_[0, :, 0, :], w_[0, :, 0, :]
            self.u_up[-1], self.v_up[-1], self.w_up[-1] = u_[-1, :, -1, :], v_[-1, :, -1, :], w_[-1, :, -1, :]

            u_, v_ = self.phys_to_contra(u, v, w)
            u_, v_, w_, = self.contra_to_phys(0 * u_, v_)
            self.u_left[:, 0], self.v_left[:, 0], self.w_left[:, 0] = u_[:, 0, :, 0], v_[:, 0, :, 0], w_[:, 0, :, 0]
            self.u_right[:, -1], self.v_right[:, -1], self.w_right[:, -1] = u_[:, -1, :, -1], v_[:, -1, :, -1], w_[:, -1, :, -1]

    def apply_forcing(self, uk, vk, hk, t, forcing):
        uk_, vk_, hk_ = forcing(self.xs, self.ys, t)
        uk += _to_numpy(uk_, dtype=self.dtype)
        vk += _to_numpy(vk_, dtype=self.dtype)
        hk += _to_numpy(hk_, dtype=self.dtype)

    def get_dt(self):
        speed = self.wave_speed(self.u, self.v, self.w, self.h)
        dt = self.cdt / np.max(speed)
        return dt

    def time_step(self, dt=None, order=3, forcing=None):
        if dt is None:
            dt = self.get_dt()

        if order == 3:
            uk_1, vk_1, wk_1, hk_1 = self.solve(self.u, self.v, self.w, self.h, self.time, dt)
            if forcing is not None: self.apply_forcing(uk_1, vk_1, hk_1, self.time, forcing)

            # SSPRK3
            u_1 = self.u + dt * uk_1
            v_1 = self.v + dt * vk_1
            w_1 = self.w + dt * wk_1
            h_1 = self.h + dt * hk_1

            uk_2, vk_2, wk_2, hk_2 = self.solve(u_1, v_1, w_1, h_1, self.time + dt, dt)

            u_2 = 0.75 * self.u + 0.25 * (u_1 + uk_2 * dt)
            v_2 = 0.75 * self.v + 0.25 * (v_1 + vk_2 * dt)
            w_2 = 0.75 * self.w + 0.25 * (w_1 + wk_2 * dt)
            h_2 = 0.75 * self.h + 0.25 * (h_1 + hk_2 * dt)
            uk_3, vk_3, wk_3, hk_3 = self.solve(u_2, v_2, w_2, h_2, self.time + 0.5 * dt, dt)

            self.u = (self.u + 2 * (u_2 + dt * uk_3)) / 3
            self.v = (self.v + 2 * (v_2 + dt * vk_3)) / 3
            self.w = (self.w + 2 * (w_2 + dt * wk_3)) / 3
            self.h = (self.h + 2 * (h_2 + dt * hk_3)) / 3
        else:
            raise ValueError(f"order: expected one of [3], found {order}.")

        self.time += dt

    def set_initial_condition(self, u, v, w, h, b=None):

        self.u = _to_numpy(u, dtype=self.dtype, copy=True)
        self.v = _to_numpy(v, dtype=self.dtype, copy=True)
        self.w = _to_numpy(w, dtype=self.dtype, copy=True)

        self.h = _to_numpy(h, dtype=self.dtype, copy=True)

        if b is None:
            self.b = np.zeros_like(self.u)
        else:
            self.b = _to_numpy(b, dtype=self.dtype, copy=True)

        self.vort = self.dg_vort(self.u, self.v, self.w, self.h)

        self.tmp1 = np.zeros_like(self.u)
        self.tmp2 = np.zeros_like(self.u)

        self.u_left = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.u_right = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.u_up = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)
        self.u_down = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)

        self.v_left = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.v_right = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.v_up = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)
        self.v_down = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)

        self.w_left = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.w_right = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.w_up = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)
        self.w_down = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)

        self.h_left = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.h_right = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.h_up = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)
        self.h_down = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)

        self.vort_left = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.vort_right = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.vort_up = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)
        self.vort_down = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)

        self.boundaries(self.u, self.v, self.w, self.h, 0)

    def locate_element(self, x1, y1):
        x1, y1 = np.broadcast_arrays(x1, y1)
        x1 = np.clip(x1, self.x_min, self.x_max)
        y1 = np.clip(y1, self.y_min, self.y_max)

        x_elem = (x1 - self.x_min) / self.lx
        y_elem = (y1 - self.y_min) / self.ly

        ix = np.floor(x_elem).astype(int)
        iy = np.floor(y_elem).astype(int)
        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)

        xi = np.clip(2 * (x_elem - ix) - 1, -1, 1)
        eta = np.clip(2 * (y_elem - iy) - 1, -1, 1)

        return iy, ix, eta, xi

    def evaluate(self, x1, y1, coeffs):
        """
        Evaluate face-local nodal DG coefficients at cubed-sphere coordinates.
        """
        coeffs = _to_numpy(coeffs)
        expected_shape = (self.ny, self.nx, self.n, self.n)
        if coeffs.shape != expected_shape:
            raise ValueError(f"coeffs: expected shape {expected_shape}. Found {coeffs.shape}.")

        x1, y1 = np.broadcast_arrays(x1, y1)
        iy, ix, eta, xi = self.locate_element(x1, y1)

        eta_basis = lagrange_basis_values(eta, self.gll_nodes)
        xi_basis = lagrange_basis_values(xi, self.gll_nodes)
        elem_coeffs = coeffs[iy, ix]

        return np.einsum('...ij,...i,...j->...', elem_coeffs, eta_basis, xi_basis)

    def _siac_widths(self, width, scale):
        if width is None:
            return scale * self.lx, scale * self.ly
        if np.isscalar(width):
            return float(width), float(width)
        if len(width) != 2:
            raise ValueError(f"width: expected a scalar or length-2 tuple. Found {width}.")
        return float(width[0]), float(width[1])

    def _siac_axis_nodes(self, axis):
        if axis == "x":
            return self.x1[0, :, 0, :].reshape(-1), self.x_min, self.lx, self.nx
        if axis == "y":
            return self.y1[:, 0, :, 0].reshape(-1), self.y_min, self.ly, self.ny
        raise ValueError(f"axis: expected one of ['x', 'y']. Found {axis}.")

    def _siac_axis_matrix(
        self,
        target_nodes,
        coord_min,
        cell_width,
        num_cells,
        derivative_order,
        width,
        kernel,
        quadrature_order,
    ):
        if derivative_order < 0:
            raise ValueError(
                f"derivative_order must be non-negative; found {derivative_order}."
            )
        if derivative_order >= kernel.spline_order:
            raise ValueError(
                "derivative_order must be less than the SIAC spline order; "
                f"found derivative_order={derivative_order}, spline_order={kernel.spline_order}."
            )

        num_targets = target_nodes.size
        matrix = np.zeros((num_targets, num_cells * self.n), dtype=self.dtype)

        quad_points, quad_weights = kernel.quadrature(
            quadrature_order, derivative=derivative_order
        )
        quad_weights = quad_weights / width ** derivative_order

        rows = np.arange(num_targets)[:, None]
        local_cols = np.arange(self.n)[None, :]
        coord_max = coord_min + num_cells * cell_width

        for point, weight in zip(quad_points, quad_weights):
            sample = np.clip(
                target_nodes - width * point,
                coord_min,
                coord_max,
            )
            elem_coord = (sample - coord_min) / cell_width
            elem = np.floor(elem_coord).astype(int)
            elem = np.clip(elem, 0, num_cells - 1)
            reference = np.clip(2.0 * (elem_coord - elem) - 1.0, -1.0, 1.0)
            basis = lagrange_basis_values(reference, self.gll_nodes)
            cols = elem[:, None] * self.n + local_cols
            np.add.at(matrix, (rows, cols), weight * basis)

        return matrix

    def _siac_matrix(self, axis, derivative_order, width, kernel, quadrature_order):
        key = (
            "local",
            axis,
            derivative_order,
            float(width),
            int(quadrature_order),
            kernel.cache_key(),
        )
        cached = self._siac_matrix_cache.get(key)
        if cached is not None:
            return cached

        target_nodes, coord_min, cell_width, num_cells = self._siac_axis_nodes(axis)
        matrix = self._siac_axis_matrix(
            target_nodes,
            coord_min,
            cell_width,
            num_cells,
            derivative_order,
            width,
            kernel,
            quadrature_order,
        )

        self._siac_matrix_cache[key] = matrix
        return matrix

    def _siac_padding(self, Hx, Hy, kernel):
        support = max(abs(kernel.breakpoints[0]), abs(kernel.breakpoints[-1]))
        pad_x = int(np.ceil(max(0.0, support * Hx / self.lx - 1.0e-12)))
        pad_y = int(np.ceil(max(0.0, support * Hy / self.ly - 1.0e-12)))
        return pad_x, pad_y

    def _siac_extended_matrix(
        self, axis, derivative_order, width, kernel, quadrature_order, pad
    ):
        key = (
            "extended",
            axis,
            int(pad),
            derivative_order,
            float(width),
            int(quadrature_order),
            kernel.cache_key(),
        )
        cached = self._siac_matrix_cache.get(key)
        if cached is not None:
            return cached

        target_nodes, _, cell_width, _ = self._siac_axis_nodes(axis)
        if axis == "x":
            coord_min = self.x_min - pad * self.lx
            num_cells = self.nx + 2 * pad
        elif axis == "y":
            coord_min = self.y_min - pad * self.ly
            num_cells = self.ny + 2 * pad
        else:
            raise ValueError(f"axis: expected one of ['x', 'y']. Found {axis}.")

        matrix = self._siac_axis_matrix(
            target_nodes,
            coord_min,
            cell_width,
            num_cells,
            derivative_order,
            width,
            kernel,
            quadrature_order,
        )
        self._siac_matrix_cache[key] = matrix
        return matrix

    def _siac_extended_coordinates(self, pad_x, pad_y):
        key = (int(pad_x), int(pad_y))
        cached = self._siac_extension_coord_cache.get(key)
        if cached is not None:
            return cached

        x_cells = np.arange(-pad_x, self.nx + pad_x)
        y_cells = np.arange(-pad_y, self.ny + pad_y)
        local_nodes = 0.5 * (1.0 + self.gll_nodes)
        x_nodes = self.x_min + (x_cells[:, None] + local_nodes[None, :]) * self.lx
        y_nodes = self.y_min + (y_cells[:, None] + local_nodes[None, :]) * self.ly

        shape = (y_cells.size, x_cells.size, self.n, self.n)
        x1 = np.broadcast_to(x_nodes[None, :, None, :], shape).copy()
        y1 = np.broadcast_to(y_nodes[:, None, :, None], shape).copy()
        self._siac_extension_coord_cache[key] = (x1, y1)
        return x1, y1

    def _siac_extension_ghost_mask(self, pad_x, pad_y):
        x_cells = np.arange(-pad_x, self.nx + pad_x)
        y_cells = np.arange(-pad_y, self.ny + pad_y)
        x_ghost = (x_cells < 0) | (x_cells >= self.nx)
        y_ghost = (y_cells < 0) | (y_cells >= self.ny)
        ghost_mask = y_ghost[:, None] | x_ghost[None, :]

        x_nudge = np.zeros((y_cells.size, x_cells.size), dtype=self.dtype)
        y_nudge = np.zeros_like(x_nudge)
        x_nudge[:, x_cells < 0] = -1.0e-12 * self.lx
        x_nudge[:, x_cells >= self.nx] = 1.0e-12 * self.lx
        y_nudge[y_cells < 0, :] = -1.0e-12 * self.ly
        y_nudge[y_cells >= self.ny, :] = 1.0e-12 * self.ly
        return ghost_mask, x_nudge, y_nudge

    def _apply_siac_matrices(self, coeffs, x_matrix, y_matrix):
        grid = coeffs.swapaxes(1, 2).reshape(
            coeffs.shape[0] * self.n, coeffs.shape[1] * self.n
        )
        filtered = y_matrix @ grid @ x_matrix.T
        out_ny = y_matrix.shape[0] // self.n
        out_nx = x_matrix.shape[0] // self.n
        return filtered.reshape(out_ny, self.n, out_nx, self.n).swapaxes(1, 2)

    def local_covariant_velocity(self, u=None, v=None, w=None):
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w

        dx_dxi_scale = 0.5 * self.lx
        dy_deta_scale = 0.5 * self.ly
        u_cov, v_cov, _ = self.phys_to_cov(u, v, w)
        return u_cov / dx_dxi_scale, v_cov / dy_deta_scale

    def local_surface_jacobian(self):
        return self.J / (0.25 * self.lx * self.ly)

    def siac_filter(
        self,
        coeffs=None,
        *,
        derivative=(0, 0),
        width=None,
        scale=1.0,
        kernel=None,
        quadrature_order=None,
    ):
        """
        Fast face-local tensor-product SIAC filter.

        Samples that would leave this face are clipped to the face edge. Use
        ``DGCubedSphereSWENumpy.siac_filter(..., boundary="sphere")`` when a
        serial, seam-aware cubed-sphere filter is needed.
        """
        if coeffs is None:
            coeffs = self.h
        elif isinstance(coeffs, str):
            coeffs = getattr(self, coeffs)
        coeffs = _to_numpy(coeffs, dtype=self.dtype)
        expected_shape = (self.ny, self.nx, self.n, self.n)
        if coeffs.shape != expected_shape:
            raise ValueError(f"coeffs: expected shape {expected_shape}. Found {coeffs.shape}.")

        if kernel is None:
            kernel = siac_kernel(self.poly_order)
        if quadrature_order is None:
            quadrature_order = max(self.poly_order + 3, 6)

        if len(derivative) != 2:
            raise ValueError(f"derivative: expected a length-2 tuple. Found {derivative}.")
        dx_order, dy_order = derivative
        Hx, Hy = self._siac_widths(width, scale)

        x_matrix = self._siac_matrix("x", dx_order, Hx, kernel, quadrature_order)
        y_matrix = self._siac_matrix("y", dy_order, Hy, kernel, quadrature_order)
        return self._apply_siac_matrices(coeffs, x_matrix, y_matrix)

    def siac_vorticity(
        self,
        *,
        width=None,
        scale=1.0,
        kernel=None,
        quadrature_order=None,
        include_coriolis=True,
    ):
        """
        Compute vorticity with SIAC derivative filters on covariant velocity.
        """
        u_cov, v_cov = self.local_covariant_velocity()
        dvy_dx = self.siac_filter(
            v_cov,
            derivative=(1, 0),
            width=width,
            scale=scale,
            kernel=kernel,
            quadrature_order=quadrature_order,
        )
        dux_dy = self.siac_filter(
            u_cov,
            derivative=(0, 1),
            width=width,
            scale=scale,
            kernel=kernel,
            quadrature_order=quadrature_order,
        )
        vort = (dvy_dx - dux_dy) / self.local_surface_jacobian()
        if include_coriolis:
            vort = vort + self.f
        return vort.astype(self.dtype, copy=False)

    def integrate(self, q):
        return (q * abs(self.Jw)).sum()

    def entropy(self, u=None, v=None, w=None, h=None):
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if h is None:
            h = self.h
        return 0.5 * h * (u ** 2 + v ** 2 + w ** 2 + self.g * h)

    def enstrophy(self, u=None, v=None, w=None, h=None):
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if h is None:
            h = self.h

        q = self.q(u, v, w, h)
        return 0.5 * h * q ** 2

    def dEdt(self):
        u, v, w, h = self.u, self.v, self.w, self.h
        dudt, dvdt, dwdt, dhdt = self.solve(u, v, w, h, 0, 0)
        # E = 0.5 * h * |u|^2 + 0.5 * g * h^2
        dEdt = h * (u * dudt + v * dvdt + w * dwdt)
        dEdt += (0.5 * (u ** 2 + v ** 2 + w ** 2) + self.g * (h + self.b)) * dhdt
        return dEdt

    def dg_vort(self, u=None, v=None, w=None, h=None):
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if h is None:
            h = self.h

        u, v, _ = self.phys_to_cov(u, v, w)
        vort = -(self.weak_ddxi(v) - self.weak_ddeta(u))
        vort *= self.grad_zeta_norm
        vort /= self.Jw
        vort += self.f
        return vort

    def vorticity(self, u=None, v=None, w=None, h=None):
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if h is None:
            h = self.h

        vort = self.dg_vort(u, v, w, h)

        vort_sum = vort * self.Jw
        h_sum = self.Jw.copy()

        Jw = self.Jw

        h_sum[0, :, 0] = h_sum[0, :, 0] + Jw[0, :, 0]
        h_sum[-1, :, -1] = h_sum[-1, :, -1] + Jw[-1, :, -1]
        h_sum[:, 0, :, 0] = h_sum[:, 0, :, 0] + Jw[:, 0, :, 0]
        h_sum[:, -1, :, -1] = h_sum[:, -1, :, -1] + Jw[:, -1, :, -1]

        vort_sum[0, :, 0] = vort_sum[0, :, 0] + self.vort_down[0] * Jw[0, :, 0]
        vort_sum[-1, :, -1] = vort_sum[-1, :, -1] + self.vort_up[-1] * Jw[-1, :, -1]
        vort_sum[:, 0, :, 0] = vort_sum[:, 0, :, 0] + self.vort_left[:, 0] * Jw[:, 0, :, 0]
        vort_sum[:, -1, :, -1] = vort_sum[:, -1, :, -1] + self.vort_right[:, -1] * Jw[:, -1, :, -1]

        for tnsr in [vort_sum, h_sum]:
            tnsr[:, 1:, :, 0] = tnsr[:, 1:, :, 0] + tnsr[:, :-1, :, -1]
            tnsr[:, :-1, :, -1] = tnsr[:, 1:, :, 0]

            tnsr[1:, :, 0] = tnsr[1:, :, 0] + tnsr[:-1, :, -1]
            tnsr[:-1, :, -1] = tnsr[1:, :, 0]

            if self.xperiodic:
                tnsr[:, 0, :, 0] = tnsr[:, 0, :, 0] + tnsr[:, -1, :, -1]
                tnsr[:, -1, :, -1] = tnsr[:, 0, :, 0]

            if self.yperiodic:
                tnsr[0, :, 0] = tnsr[0, :, 0] + tnsr[-1, :, -1]
                tnsr[-1, :, -1] = tnsr[0, :, 0]

        vort = vort_sum / h_sum

        return vort

    def q(self, u=None, v=None, w=None, h=None):
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if h is None:
            h = self.h

        vort = self.dg_vort(u, v, w, h)
        vort_sum = vort * self.Jw
        h_sum = h * self.Jw

        Jw = self.Jw

        h_sum[0, :, 0] = h_sum[0, :, 0] + self.h_down[0] * Jw[0, :, 0]
        h_sum[-1, :, -1] = h_sum[-1, :, -1] + self.h_up[-1] * Jw[-1, :, -1]
        h_sum[:, 0, :, 0] = h_sum[:, 0, :, 0] + self.h_left[:, 0] * Jw[:, 0, :, 0]
        h_sum[:, -1, :, -1] = h_sum[:, -1, :, -1] + self.h_right[:, -1] * Jw[:, -1, :, -1]

        vort_sum[0, :, 0] = vort_sum[0, :, 0] + self.vort_down[0] * Jw[0, :, 0]
        vort_sum[-1, :, -1] = vort_sum[-1, :, -1] + self.vort_up[-1] * Jw[-1, :, -1]
        vort_sum[:, 0, :, 0] = vort_sum[:, 0, :, 0] + self.vort_left[:, 0] * Jw[:, 0, :, 0]
        vort_sum[:, -1, :, -1] = vort_sum[:, -1, :, -1] + self.vort_right[:, -1] * Jw[:, -1, :, -1]

        for tnsr in [vort_sum, h_sum]:
            tnsr[:, 1:, :, 0] = tnsr[:, 1:, :, 0] + tnsr[:, :-1, :, -1]
            tnsr[:, :-1, :, -1] = tnsr[:, 1:, :, 0]

            tnsr[1:, :, 0] = tnsr[1:, :, 0] + tnsr[:-1, :, -1]
            tnsr[:-1, :, -1] = tnsr[1:, :, 0]

            if self.xperiodic:
                tnsr[:, 0, :, 0] = tnsr[:, 0, :, 0] + tnsr[:, -1, :, -1]
                tnsr[:, -1, :, -1] = tnsr[:, 0, :, 0]

            if self.yperiodic:
                tnsr[0, :, 0] = tnsr[0, :, 0] + tnsr[-1, :, -1]
                tnsr[-1, :, -1] = tnsr[0, :, 0]

        q = vort_sum / h_sum

        return q

    def plot_solution(self, ax, vmin=None, vmax=None, plot_func=None, dim=3, cmap='nipy_spectral'):
        x_plot = self.xs.swapaxes(1, 2).reshape(self.h.shape[0] * self.h.shape[2], -1)
        y_plot = self.ys.swapaxes(1, 2).reshape(self.h.shape[0] * self.h.shape[2], -1)
        z_plot = self.zs.swapaxes(1, 2).reshape(self.h.shape[0] * self.h.shape[2], -1)

        if plot_func is None:
            data_plot = self.h.swapaxes(1, 2).reshape(self.h.shape[0] * self.h.shape[2], -1)
        else:
            out = plot_func(self)
            data_plot = out.swapaxes(1, 2).reshape(out.shape[0] * out.shape[2], -1)

        if dim == 3:
            from matplotlib import cm
            if vmin is None:
                vmin = data_plot.min()
            if vmax is None:
                vmax = data_plot.max()

            heatmap = data_plot - vmin
            heatmap /= (vmax - vmin)
            return ax.plot_surface(x_plot, y_plot, z_plot, cmap=cmap, vmin=vmin, vmax=vmax, facecolors=cm.jet(heatmap))
        elif dim == 2:
            if self.name in ['xp', 'yn']:
                idx = (z_plot.shape[0] // 2) + 1
                x_plot = x_plot[idx:]
                y_plot = y_plot[idx:]
                z_plot = z_plot[idx:]
                data_plot = data_plot[idx:]
            elif self.name in ['xn', 'yp']:
                idx = (z_plot.shape[1] // 2) + 1
                x_plot = x_plot[:, idx:]
                y_plot = y_plot[:, idx:]
                z_plot = z_plot[:, idx:]
                data_plot = data_plot[:, idx:]
            elif self.name == 'zn':
                raise ValueError('Face zn plotted.')

            # lat, long = self.geometry.lat_long(x_plot, y_plot, z_plot)
            #
            # import scipy
            # func = scipy.interpolate.interp2d(long.ravel(), lat.ravel(), data_plot.ravel())
            # lat = np.linspace(lat.min(), lat.max(), 100)
            # long = np.linspace(long.min(), long.max(), 100)
            # long, lat = np.meshgrid(long, lat)

            # return ax.contourf(long, lat, func(long, lat), cmap=cmap, vmin=vmin, vmax=vmax, levels=1000)
            return ax.contourf(x_plot, y_plot, data_plot, cmap=cmap, vmin=vmin, vmax=vmax, levels=1000)

    def imshow_solution(self, ax):
        u_plot = self.h.swapaxes(1, 2).reshape(self.h.shape[0] * self.h.shape[2], -1)

        return ax.contourf(u_plot)

    def hflux(self, u, v, w, h):
        yflux = v * h
        xflux = u * h
        zflux = w * h

        return xflux, yflux, zflux

    def uv_flux(self, u, v, w, h):
        return 0.5 * (u ** 2 + v ** 2 + w ** 2) + self.g * h

    def wave_speed(self, u, v, w, h):
        return np.sqrt(u ** 2 + v ** 2 + w ** 2) + np.sqrt(self.g * h)

    def solve(self, u, v, w, h, t, dt, *, verbose=False):
        if _solve_numba_kernel is None:
            return self.solve_numpy(u, v, w, h, t, dt, verbose=verbose)

        self.boundaries(u, v, w, h, t)
        if self.flux_type in ("barth", "barth_normal_tangent"):
            normal_only_barth = self.flux_type == "barth_normal_tangent"
            u_k, v_k, w_k, h_k = _solve_numba_kernel(
                u, v, w, h, self.b,
                self.D, self.endpoint_weight, self.J,
                self.vert_upper_edge_factor, self.vert_lower_edge_factor,
                self.horz_right_edge_factor, self.horz_left_edge_factor,
                self.dxidx, self.dxidy, self.dxidz, self.detadx, self.detady, self.detadz,
                self.dxidx_up, self.dxidy_up, self.dxidz_up, self.dxidx_down, self.dxidy_down, self.dxidz_down,
                self.dxidx_right, self.dxidy_right, self.dxidz_right, self.dxidx_left, self.dxidy_left, self.dxidz_left,
                self.detadx_up, self.detady_up, self.detadz_up, self.detadx_down, self.detady_down, self.detadz_down,
                self.detadx_right, self.detady_right, self.detadz_right, self.detadx_left, self.detady_left, self.detadz_left,
                self.dxdxi, self.dydxi, self.dzdxi, self.dxdeta, self.dydeta, self.dzdeta,
                self.f,
                self.u_up, self.v_up, self.w_up, self.h_up, self.u_down, self.v_down, self.w_down, self.h_down,
                self.u_right, self.v_right, self.w_right, self.h_right, self.u_left, self.v_left, self.w_left, self.h_left,
                self.eta_x_up, self.eta_y_up, self.eta_z_up, self.eta_x_down, self.eta_y_down, self.eta_z_down,
                self.xi_x_right, self.xi_y_right, self.xi_z_right, self.xi_x_left, self.xi_y_left, self.xi_z_left,
                self.dxdxi_up, self.dydxi_up, self.dzdxi_up, self.dxdxi_down, self.dydxi_down, self.dzdxi_down,
                self.dxdxi_right, self.dydxi_right, self.dzdxi_right, self.dxdxi_left, self.dydxi_left, self.dzdxi_left,
                self.dxdeta_up, self.dydeta_up, self.dzdeta_up, self.dxdeta_down, self.dydeta_down, self.dzdeta_down,
                self.dxdeta_right, self.dydeta_right, self.dzdeta_right, self.dxdeta_left, self.dydeta_left, self.dzdeta_left,
                self.g, 0.0, 0.0, normal_only_barth,
            )
            _apply_barth_diss_numba(
                u_k, v_k, w_k, h_k, self.g,
                self.vert_upper_edge_factor, self.vert_lower_edge_factor,
                self.horz_right_edge_factor, self.horz_left_edge_factor,
                self.u_up, self.v_up, self.w_up, self.h_up, self.u_down, self.v_down, self.w_down, self.h_down,
                self.u_right, self.v_right, self.w_right, self.h_right, self.u_left, self.v_left, self.w_left, self.h_left,
                self.eta_x_up, self.eta_y_up, self.eta_z_up, self.eta_x_down, self.eta_y_down, self.eta_z_down,
                self.xi_x_right, self.xi_y_right, self.xi_z_right, self.xi_x_left, self.xi_y_left, self.xi_z_left,
                self.dxdxi_up, self.dydxi_up, self.dzdxi_up, self.dxdxi_down, self.dydxi_down, self.dzdxi_down,
                self.dxdxi_right, self.dydxi_right, self.dzdxi_right, self.dxdxi_left, self.dydxi_left, self.dzdxi_left,
                self.dxdeta_up, self.dydeta_up, self.dzdeta_up, self.dxdeta_down, self.dydeta_down, self.dzdeta_down,
                self.dxdeta_right, self.dydeta_right, self.dzdeta_right, self.dxdeta_left, self.dydeta_left, self.dzdeta_left,
                normal_only_barth,
            )
            return u_k, v_k, w_k, h_k

        if self.flux_type == "old_tangent":
            return _solve_numba_old_tangent_kernel(
                u, v, w, h, self.b,
                self.D, self.endpoint_weight, self.J,
                self.vert_upper_edge_factor, self.vert_lower_edge_factor,
                self.horz_right_edge_factor, self.horz_left_edge_factor,
                self.dxidx, self.dxidy, self.dxidz, self.detadx, self.detady, self.detadz,
                self.dxidx_up, self.dxidy_up, self.dxidz_up, self.dxidx_down, self.dxidy_down, self.dxidz_down,
                self.dxidx_right, self.dxidy_right, self.dxidz_right, self.dxidx_left, self.dxidy_left, self.dxidz_left,
                self.detadx_up, self.detady_up, self.detadz_up, self.detadx_down, self.detady_down, self.detadz_down,
                self.detadx_right, self.detady_right, self.detadz_right, self.detadx_left, self.detady_left, self.detadz_left,
                self.dxdxi, self.dydxi, self.dzdxi, self.dxdeta, self.dydeta, self.dzdeta,
                self.f,
                self.u_up, self.v_up, self.w_up, self.h_up, self.u_down, self.v_down, self.w_down, self.h_down,
                self.u_right, self.v_right, self.w_right, self.h_right, self.u_left, self.v_left, self.w_left, self.h_left,
                self.eta_x_up, self.eta_y_up, self.eta_z_up, self.eta_x_down, self.eta_y_down, self.eta_z_down,
                self.xi_x_right, self.xi_y_right, self.xi_z_right, self.xi_x_left, self.xi_y_left, self.xi_z_left,
                self.J_eta, self.J_xi,
                self.dxdxi_up, self.dydxi_up, self.dzdxi_up, self.dxdxi_down, self.dydxi_down, self.dzdxi_down,
                self.dxdxi_right, self.dydxi_right, self.dzdxi_right, self.dxdxi_left, self.dydxi_left, self.dzdxi_left,
                self.dxdeta_up, self.dydeta_up, self.dzdeta_up, self.dxdeta_down, self.dydeta_down, self.dzdeta_down,
                self.dxdeta_right, self.dydeta_right, self.dzdeta_right, self.dxdeta_left, self.dydeta_left, self.dzdeta_left,
                self.g, self.a, self.ah,
            )

        kernel = _solve_numba_lmars_kernel if self.flux_type == "lmars" else _solve_numba_kernel
        return kernel(
            u, v, w, h, self.b,
            self.D, self.endpoint_weight, self.J,
            self.vert_upper_edge_factor, self.vert_lower_edge_factor,
            self.horz_right_edge_factor, self.horz_left_edge_factor,
            self.dxidx, self.dxidy, self.dxidz, self.detadx, self.detady, self.detadz,
            self.dxidx_up, self.dxidy_up, self.dxidz_up, self.dxidx_down, self.dxidy_down, self.dxidz_down,
            self.dxidx_right, self.dxidy_right, self.dxidz_right, self.dxidx_left, self.dxidy_left, self.dxidz_left,
            self.detadx_up, self.detady_up, self.detadz_up, self.detadx_down, self.detady_down, self.detadz_down,
            self.detadx_right, self.detady_right, self.detadz_right, self.detadx_left, self.detady_left, self.detadz_left,
            self.dxdxi, self.dydxi, self.dzdxi, self.dxdeta, self.dydeta, self.dzdeta,
            self.f,
            self.u_up, self.v_up, self.w_up, self.h_up, self.u_down, self.v_down, self.w_down, self.h_down,
            self.u_right, self.v_right, self.w_right, self.h_right, self.u_left, self.v_left, self.w_left, self.h_left,
            self.eta_x_up, self.eta_y_up, self.eta_z_up, self.eta_x_down, self.eta_y_down, self.eta_z_down,
            self.xi_x_right, self.xi_y_right, self.xi_z_right, self.xi_x_left, self.xi_y_left, self.xi_z_left,
            self.dxdxi_up, self.dydxi_up, self.dzdxi_up, self.dxdxi_down, self.dydxi_down, self.dzdxi_down,
            self.dxdxi_right, self.dydxi_right, self.dzdxi_right, self.dxdxi_left, self.dydxi_left, self.dzdxi_left,
            self.dxdeta_up, self.dydeta_up, self.dzdeta_up, self.dxdeta_down, self.dydeta_down, self.dzdeta_down,
            self.dxdeta_right, self.dydeta_right, self.dzdeta_right, self.dxdeta_left, self.dydeta_left, self.dzdeta_left,
            self.g, self.a, self.ah, self.flux_type == "standard_tangent",
        )

    def solve_numpy(self, u, v, w, h, t, dt, *, verbose=False):
        if self.flux_type == "barth":
            return self.solve_numpy_barth_diss(
                u, v, w, h, t, dt, tangent_diss=False, normal_only=False, verbose=verbose
            )
        if self.flux_type == "barth_normal_tangent":
            return self.solve_numpy_barth_diss(
                u, v, w, h, t, dt, tangent_diss=True, normal_only=True, verbose=verbose
            )
        if self.flux_type == "lmars":
            return self.solve_numpy_lmars(u, v, w, h, t, dt, verbose=verbose)
        if self.flux_type == "old_tangent":
            return self.solve_numpy_old_tangent(u, v, w, h, t, dt, verbose=verbose)
        return self._solve_numpy_standard(
            u, v, w, h, t, dt, tangent_diss=self.flux_type == "standard_tangent", verbose=verbose
        )

    def solve_numpy_lmars(self, u, v, w, h, t, dt, *, verbose=False):
        return self._solve_numpy_lmars_kernel(
            u, v, w, h, t, dt, tangent_diss=False, verbose=verbose
        )

    def solve_numpy_old_tangent(self, u, v, w, h, t, dt, *, verbose=False):
        u_k, v_k, w_k, h_k = self._solve_numpy_standard(
            u, v, w, h, t, dt, tangent_diss=False, verbose=verbose
        )
        self._apply_old_tangent_diss_numpy(u_k, v_k, w_k)
        return u_k, v_k, w_k, h_k

    def solve_numpy_barth_diss(self, u, v, w, h, t, dt, *, tangent_diss, normal_only, verbose=False):
        u_k, v_k, w_k, h_k = self._solve_numpy_standard(
            u, v, w, h, t, dt, tangent_diss=tangent_diss, a=0.0, ah=0.0, verbose=verbose
        )
        self._apply_barth_diss_numpy(u_k, v_k, w_k, h_k, normal_only=normal_only)
        return u_k, v_k, w_k, h_k

    def _solve_numpy_standard(self, u, v, w, h, t, dt, *, tangent_diss, a=None, ah=None, verbose=False):
        if a is None:
            a = self.a
        if ah is None:
            ah = self.ah

        # copy the boundaries across
        self.boundaries(u, v, w, h, t)

        # handle h
        h_xflux, h_yflux, h_zflux = self.hflux(u, v, w, h)
        h_xflux, h_yflux = self.phys_to_contra(h_xflux, h_yflux,
                                               h_zflux)  # flux is in contravariant form
        div = self.ddxi(h_xflux * self.J)
        div += self.ddeta(h_yflux * self.J)
        div /= self.J
        verbose = False

        out = -self.Jw * div

        h_up_flux_x, h_up_flux_y, h_up_flux_z = self.hflux(self.u_up, self.v_up, self.w_up, self.h_up)
        h_down_flux_x, h_down_flux_y, h_down_flux_z = self.hflux(self.u_down, self.v_down, self.w_down, self.h_down)
        h_right_flux_x, h_right_flux_y, h_right_flux_z = self.hflux(self.u_right, self.v_right, self.w_right, self.h_right)
        h_left_flux_x, h_left_flux_y, h_left_flux_z = self.hflux(self.u_left, self.v_left, self.w_left, self.h_left)

        # fluxes through boundary
        h_up_flux = h_up_flux_y * self.eta_y_up + h_up_flux_x * self.eta_x_up + h_up_flux_z * self.eta_z_up
        h_down_flux = h_down_flux_y * self.eta_y_down + h_down_flux_x * self.eta_x_down + h_down_flux_z * self.eta_z_down
        h_right_flux = h_right_flux_y * self.xi_y_right + h_right_flux_x * self.xi_x_right + h_right_flux_z * self.xi_z_right
        h_left_flux = h_left_flux_y * self.xi_y_left + h_left_flux_x * self.xi_x_left + h_left_flux_z * self.xi_z_left

        uv_up_flux = self.uv_flux(self.u_up, self.v_up, self.w_up, self.h_up)
        uv_down_flux = self.uv_flux(self.u_down, self.v_down, self.w_down, self.h_down)
        uv_right_flux = self.uv_flux(self.u_right, self.v_right, self.w_right, self.h_right)
        uv_left_flux = self.uv_flux(self.u_left, self.v_left, self.w_left, self.h_left)

        # upper boundary
        # c_up = self.wave_speed(self.u_up, self.v_up, self.w_up, self.h_up)
        # c_down = self.wave_speed(self.u_down, self.v_down, self.w_down, self.h_down)
        # c_ve = 0.5 * (c_up + c_down)
        # c_right = self.wave_speed(self.u_right, self.v_right, self.w_right, self.h_right)
        # c_left = self.wave_speed(self.u_left, self.v_left, self.w_left, self.h_left)
        # c_ho = 0.5 * (c_right + c_left)

        vel_up = h_up_flux / self.h_up
        vel_down = h_down_flux / self.h_down
        vel_right = h_right_flux / self.h_right
        vel_left = h_left_flux / self.h_left

        c_snd_ho = 0.5 * (np.sqrt(self.g * self.h_right) + np.sqrt(self.g * self.h_left))
        c_snd_ve = 0.5 * (np.sqrt(self.g * self.h_up) + np.sqrt(self.g * self.h_down))
        # c_ho = c_adv_horz + c_snd_ho
        # c_ve = c_adv_vert + c_snd_ve
        h_ve = 0.5 * (self.h_up + self.h_down)
        h_ho = 0.5 * (self.h_right + self.h_left)

        c_adv_vert = 0.5 * (self.h_up * vel_up + self.h_down * vel_down) / h_ve #- self.g * self.ah * (self.h_up - self.h_down) / (c_snd_ve * h_ve)
        c_adv_horz = 0.5 * (self.h_right * vel_right + self.h_left * vel_left) / h_ho #- self.g * self.ah * (self.h_right - self.h_left) / (c_snd_ho * h_ho)

        u_cov_up = self.u_up * self.dxdxi_up + self.v_up * self.dydxi_up + self.w_up * self.dzdxi_up
        u_cov_down = self.u_down * self.dxdxi_down + self.v_down * self.dydxi_down + self.w_down * self.dzdxi_down
        u_cov_right = self.u_right * self.dxdxi_right + self.v_right * self.dydxi_right + self.w_right * self.dzdxi_right
        u_cov_left = self.u_left * self.dxdxi_left + self.v_left * self.dydxi_left + self.w_left * self.dzdxi_left

        v_cov_up = self.u_up * self.dxdeta_up + self.v_up * self.dydeta_up + self.w_up * self.dzdeta_up
        v_cov_down = self.u_down * self.dxdeta_down + self.v_down * self.dydeta_down + self.w_down * self.dzdeta_down
        v_cov_right = self.u_right * self.dxdeta_right + self.v_right * self.dydeta_right + self.w_right * self.dzdeta_right
        v_cov_left = self.u_left * self.dxdeta_left + self.v_left * self.dydeta_left + self.w_left * self.dzdeta_left

        u_contra_up = self.u_up * self.dxidx_up + self.v_up * self.dxidy_up + self.w_up * self.dxidz_up
        u_contra_down = self.u_down * self.dxidx_down + self.v_down * self.dxidy_down + self.w_down * self.dxidz_down
        u_contra_right = self.u_right * self.dxidx_right + self.v_right * self.dxidy_right + self.w_right * self.dxidz_right
        u_contra_left = self.u_left * self.dxidx_left + self.v_left * self.dxidy_left + self.w_left * self.dxidz_left

        v_contra_up = self.u_up * self.detadx_up + self.v_up * self.detady_up + self.w_up * self.detadz_up
        v_contra_down = self.u_down * self.detadx_down + self.v_down * self.detady_down + self.w_down * self.detadz_down
        v_contra_right = self.u_right * self.detadx_right + self.v_right * self.detady_right + self.w_right * self.detadz_right
        v_contra_left = self.u_left * self.detadx_left + self.v_left * self.detady_left + self.w_left * self.detadz_left

        h_flux_vert = c_adv_vert * h_ve - ah * abs(c_adv_vert) * (self.h_up - self.h_down)
        h_flux_horz = c_adv_horz * h_ho - ah * abs(c_adv_horz) * (self.h_right - self.h_left)

        uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux) - a * (c_snd_ho + abs(c_adv_horz)) * (h_right_flux - h_left_flux) / h_ho
        uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux) - a * (c_snd_ve + abs(c_adv_vert)) * (h_up_flux - h_down_flux) / h_ve

        if tangent_diss:
            u_cov_vert_avg = (c_adv_vert < 0) * u_cov_up + (c_adv_vert >= 0) * u_cov_down
            v_cov_horz_avg = (c_adv_horz < 0) * v_cov_right + (c_adv_horz >= 0) * v_cov_left
        else:
            u_cov_vert_avg = 0.5 * (u_cov_up + u_cov_down)
            v_cov_horz_avg = 0.5 * (v_cov_right + v_cov_left)

        u_flux_vert_up = v_contra_up * u_cov_vert_avg
        u_flux_vert_down = v_contra_down * u_cov_vert_avg
        v_flux_vert_up = uv_flux_vert - u_contra_up * u_cov_vert_avg
        v_flux_vert_down = uv_flux_vert - u_contra_down * u_cov_vert_avg

        u_flux_horz_right = uv_flux_horz - v_contra_right * v_cov_horz_avg
        u_flux_horz_left = uv_flux_horz - v_contra_left * v_cov_horz_avg
        v_flux_horz_right = u_contra_right * v_cov_horz_avg
        v_flux_horz_left = u_contra_left * v_cov_horz_avg

        # h_flux_vert = 0.5 * (h_up_flux + h_down_flux) - self.a * c_snd_ve * (uv_up_flux - uv_down_flux) / self.g
        # h_flux_horz = 0.5 * (h_right_flux + h_left_flux) - self.a * c_snd_ho * (uv_right_flux - uv_left_flux) / self.g

        self.tmp1[:, :, -1] = (h_flux_vert[1:] - h_down_flux[1:]) * (self.weights_x * self.J_vertface[:, :, -1])
        self.tmp1[:, :, 0] = -(h_flux_vert[:-1] - h_up_flux[:-1]) * (self.weights_x * self.J_vertface[:, :, 0])
        self.tmp2[:, :, :, -1] = (h_flux_horz[:, 1:] - h_left_flux[:, 1:]) * (self.weights_x * self.J_horzface[..., -1])
        self.tmp2[:, :, :, 0] = -(h_flux_horz[:, :-1] - h_right_flux[:, :-1]) * (
                self.weights_x * self.J_horzface[..., 0])
        out -= (self.tmp1 + self.tmp2)

        h_k = out / self.Jw

        # u and v fluxes
        ########
        #######

        uv_flux = self.uv_flux(u, v, w, h)

        # alpha = np.maximum(c_ho / self.h_right, c_ho / self.h_left)
        # uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux) - self.a * (h_right_flux - h_left_flux) * alpha #(self.g / c_ho)
        # alpha = np.maximum(c_ve / self.h_up, c_ve / self.h_down)
        # uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux) - self.a * (h_up_flux - h_down_flux) * alpha #(self.g / c_ve) * (h_up_flux - h_down_flux)

        if self.b is not None:
            uv_flux += self.g * self.b

        u_contra, v_contra = self.phys_to_contra(u, v, w)
        u_cov, v_cov, _ = self.phys_to_cov(u, v, w)
        abs_vort_cov = self.ddxi(v_cov)
        abs_vort_cov += -self.ddeta(u_cov)
        abs_vort_cov += self.f * self.J

        #
        #
        u_cov_up = self.u_up * self.dxdxi_up + self.v_up * self.dydxi_up + self.w_up * self.dzdxi_up
        u_cov_down = self.u_down * self.dxdxi_down + self.v_down * self.dydxi_down + self.w_down * self.dzdxi_down
        u_cov_right = self.u_right * self.dxdxi_right + self.v_right * self.dydxi_right + self.w_right * self.dzdxi_right
        u_cov_left = self.u_left * self.dxdxi_left + self.v_left * self.dydxi_left + self.w_left * self.dzdxi_left

        v_cov_up = self.u_up * self.dxdeta_up + self.v_up * self.dydeta_up + self.w_up * self.dzdeta_up
        v_cov_down = self.u_down * self.dxdeta_down + self.v_down * self.dydeta_down + self.w_down * self.dzdeta_down
        v_cov_right = self.u_right * self.dxdeta_right + self.v_right * self.dydeta_right + self.w_right * self.dzdeta_right
        v_cov_left = self.u_left * self.dxdeta_left + self.v_left * self.dydeta_left + self.w_left * self.dzdeta_left

        u_contra_up = self.u_up * self.dxidx_up + self.v_up * self.dxidy_up + self.w_up * self.dxidz_up
        u_contra_down = self.u_down * self.dxidx_down + self.v_down * self.dxidy_down + self.w_down * self.dxidz_down
        u_contra_right = self.u_right * self.dxidx_right + self.v_right * self.dxidy_right + self.w_right * self.dxidz_right
        u_contra_left = self.u_left * self.dxidx_left + self.v_left * self.dxidy_left + self.w_left * self.dxidz_left

        v_contra_up = self.u_up * self.detadx_up + self.v_up * self.detady_up + self.w_up * self.detadz_up
        v_contra_down = self.u_down * self.detadx_down + self.v_down * self.detady_down + self.w_down * self.detadz_down
        v_contra_right = self.u_right * self.detadx_right + self.v_right * self.detady_right + self.w_right * self.detadz_right
        v_contra_left = self.u_left * self.detadx_left + self.v_left * self.detady_left + self.w_left * self.detadz_left

        # handle u
        #######
        ###

        u_k = -self.ddxi(uv_flux)
        u_k += v_contra * abs_vort_cov

        wx = self.weights_x.ravel()[-1]

        u_k[:, :, -1] -= (u_flux_vert_down[1:] - (v_contra_down * u_cov_down)[1:]) / wx
        u_k[:, :, 0] += (u_flux_vert_up[:-1] - (v_contra_up * u_cov_up)[:-1]) / wx
        u_k[:, :, :, -1] -= (
            u_flux_horz_left[:, 1:] - (uv_left_flux - v_contra_left * v_cov_left)[:, 1:]
        ) / wx
        u_k[:, :, :, 0] += (
            u_flux_horz_right[:, :-1] - (uv_right_flux - v_contra_right * v_cov_right)[:, :-1]
        ) / wx

        # handle v
        #######
        ###

        v_k = -self.ddeta(uv_flux)
        v_k += -u_contra * abs_vort_cov

        v_k[:, :, -1] -= (
            v_flux_vert_down[1:] - (uv_down_flux - u_contra_down * u_cov_down)[1:]
        ) / wx
        v_k[:, :, 0] += (
            v_flux_vert_up[:-1] - (uv_up_flux - u_contra_up * u_cov_up)[:-1]
        ) / wx
        v_k[:, :, :, -1] -= (
            v_flux_horz_left[:, 1:] - (u_contra_left * v_cov_left)[:, 1:]
        ) / wx
        v_k[:, :, :, 0] += (
            v_flux_horz_right[:, :-1] - (u_contra_right * v_cov_right)[:, :-1]
        ) / wx

        u_k, v_k, w_k = self.cov_to_phys(u_k, v_k, 0)

        return u_k, v_k, w_k, h_k

    def _barth_dissipation_numpy(
            self,
            u_l, v_l, w_l, h_l,
            u_r, v_r, w_r, h_r,
            n_x, n_y, n_z,
            t_x, t_y, t_z,
            normal_only=False,
    ):
        n_norm = np.sqrt(n_x ** 2 + n_y ** 2 + n_z ** 2)
        n_x = n_x / n_norm
        n_y = n_y / n_norm
        n_z = n_z / n_norm

        t_norm = np.sqrt(t_x ** 2 + t_y ** 2 + t_z ** 2)
        t_x = t_x / t_norm
        t_y = t_y / t_norm
        t_z = t_z / t_norm

        h_avg = 0.5 * (h_l + h_r)
        c = np.sqrt(self.g * h_avg)
        u_avg = 0.5 * (u_l + u_r)
        v_avg = 0.5 * (v_l + v_r)
        w_avg = 0.5 * (w_l + w_r)
        du = u_r - u_l
        dv = v_r - v_l
        dw = w_r - w_l
        dh = h_r - h_l

        u_n = u_avg * n_x + v_avg * n_y + w_avg * n_z
        du_n = du * n_x + dv * n_y + dw * n_z
        du_t = du * t_x + dv * t_y + dw * t_z

        mu_m = np.abs(u_n - c)
        mu_0 = np.abs(u_n)
        mu_p = np.abs(u_n + c)
        amp_m = (self.g * dh - c * du_n) / (2.0 * self.g)
        amp_p = (self.g * dh + c * du_n) / (2.0 * self.g)

        diss_h = mu_m * amp_m + mu_p * amp_p
        if normal_only:
            diss_un = mu_m * amp_m * (u_n - c) + mu_p * amp_p * (u_n + c)
            return diss_un, diss_h, n_x, n_y, n_z

        amp_0 = h_avg * du_t
        diss_u = (
            mu_m * amp_m * (u_avg - c * n_x)
            + mu_p * amp_p * (u_avg + c * n_x)
            + mu_0 * amp_0 * t_x
        )
        diss_v = (
            mu_m * amp_m * (v_avg - c * n_y)
            + mu_p * amp_p * (v_avg + c * n_y)
            + mu_0 * amp_0 * t_y
        )
        diss_w = (
            mu_m * amp_m * (w_avg - c * n_z)
            + mu_p * amp_p * (w_avg + c * n_z)
            + mu_0 * amp_0 * t_z
        )
        return diss_u, diss_v, diss_w, diss_h

    def _apply_barth_diss_numpy(self, u_k, v_k, w_k, h_k, *, normal_only):
        if normal_only:
            diss_un, diss_h, n_x, n_y, n_z = self._barth_dissipation_numpy(
                self.u_down, self.v_down, self.w_down, self.h_down,
                self.u_up, self.v_up, self.w_up, self.h_up,
                0.5 * (self.eta_x_down + self.eta_x_up),
                0.5 * (self.eta_y_down + self.eta_y_up),
                0.5 * (self.eta_z_down + self.eta_z_up),
                0.5 * (self.dxdxi_down + self.dxdxi_up),
                0.5 * (self.dydxi_down + self.dydxi_up),
                0.5 * (self.dzdxi_down + self.dzdxi_up),
                normal_only=True,
            )
        else:
            diss_u, diss_v, diss_w, diss_h = self._barth_dissipation_numpy(
                self.u_down, self.v_down, self.w_down, self.h_down,
                self.u_up, self.v_up, self.w_up, self.h_up,
                0.5 * (self.eta_x_down + self.eta_x_up),
                0.5 * (self.eta_y_down + self.eta_y_up),
                0.5 * (self.eta_z_down + self.eta_z_up),
                0.5 * (self.dxdxi_down + self.dxdxi_up),
                0.5 * (self.dydxi_down + self.dydxi_up),
                0.5 * (self.dzdxi_down + self.dzdxi_up),
            )

        scale = 0.5 * self.vert_upper_edge_factor
        dh_k = scale * diss_h[1:]
        h_l = self.h_down[1:]
        h_k[:, :, -1] += dh_k
        if normal_only:
            un_l = (
                self.u_down[1:] * n_x[1:]
                + self.v_down[1:] * n_y[1:]
                + self.w_down[1:] * n_z[1:]
            )
            dun = (scale * diss_un[1:] - un_l * dh_k) / h_l
            u_k[:, :, -1] += dun * n_x[1:]
            v_k[:, :, -1] += dun * n_y[1:]
            w_k[:, :, -1] += dun * n_z[1:]
        else:
            dm_u = scale * diss_u[1:]
            dm_v = scale * diss_v[1:]
            dm_w = scale * diss_w[1:]
            u_k[:, :, -1] += (dm_u - self.u_down[1:] * dh_k) / h_l
            v_k[:, :, -1] += (dm_v - self.v_down[1:] * dh_k) / h_l
            w_k[:, :, -1] += (dm_w - self.w_down[1:] * dh_k) / h_l

        scale = -0.5 * self.vert_lower_edge_factor
        dh_k = scale * diss_h[:-1]
        h_r = self.h_up[:-1]
        h_k[:, :, 0] += dh_k
        if normal_only:
            un_r = (
                self.u_up[:-1] * n_x[:-1]
                + self.v_up[:-1] * n_y[:-1]
                + self.w_up[:-1] * n_z[:-1]
            )
            dun = (scale * diss_un[:-1] - un_r * dh_k) / h_r
            u_k[:, :, 0] += dun * n_x[:-1]
            v_k[:, :, 0] += dun * n_y[:-1]
            w_k[:, :, 0] += dun * n_z[:-1]
        else:
            dm_u = scale * diss_u[:-1]
            dm_v = scale * diss_v[:-1]
            dm_w = scale * diss_w[:-1]
            u_k[:, :, 0] += (dm_u - self.u_up[:-1] * dh_k) / h_r
            v_k[:, :, 0] += (dm_v - self.v_up[:-1] * dh_k) / h_r
            w_k[:, :, 0] += (dm_w - self.w_up[:-1] * dh_k) / h_r

        if normal_only:
            diss_un, diss_h, n_x, n_y, n_z = self._barth_dissipation_numpy(
                self.u_left, self.v_left, self.w_left, self.h_left,
                self.u_right, self.v_right, self.w_right, self.h_right,
                0.5 * (self.xi_x_left + self.xi_x_right),
                0.5 * (self.xi_y_left + self.xi_y_right),
                0.5 * (self.xi_z_left + self.xi_z_right),
                0.5 * (self.dxdeta_left + self.dxdeta_right),
                0.5 * (self.dydeta_left + self.dydeta_right),
                0.5 * (self.dzdeta_left + self.dzdeta_right),
                normal_only=True,
            )
        else:
            diss_u, diss_v, diss_w, diss_h = self._barth_dissipation_numpy(
                self.u_left, self.v_left, self.w_left, self.h_left,
                self.u_right, self.v_right, self.w_right, self.h_right,
                0.5 * (self.xi_x_left + self.xi_x_right),
                0.5 * (self.xi_y_left + self.xi_y_right),
                0.5 * (self.xi_z_left + self.xi_z_right),
                0.5 * (self.dxdeta_left + self.dxdeta_right),
                0.5 * (self.dydeta_left + self.dydeta_right),
                0.5 * (self.dzdeta_left + self.dzdeta_right),
            )

        scale = 0.5 * self.horz_right_edge_factor
        dh_k = scale * diss_h[:, 1:]
        h_l = self.h_left[:, 1:]
        h_k[:, :, :, -1] += dh_k
        if normal_only:
            un_l = (
                self.u_left[:, 1:] * n_x[:, 1:]
                + self.v_left[:, 1:] * n_y[:, 1:]
                + self.w_left[:, 1:] * n_z[:, 1:]
            )
            dun = (scale * diss_un[:, 1:] - un_l * dh_k) / h_l
            u_k[:, :, :, -1] += dun * n_x[:, 1:]
            v_k[:, :, :, -1] += dun * n_y[:, 1:]
            w_k[:, :, :, -1] += dun * n_z[:, 1:]
        else:
            dm_u = scale * diss_u[:, 1:]
            dm_v = scale * diss_v[:, 1:]
            dm_w = scale * diss_w[:, 1:]
            u_k[:, :, :, -1] += (dm_u - self.u_left[:, 1:] * dh_k) / h_l
            v_k[:, :, :, -1] += (dm_v - self.v_left[:, 1:] * dh_k) / h_l
            w_k[:, :, :, -1] += (dm_w - self.w_left[:, 1:] * dh_k) / h_l

        scale = -0.5 * self.horz_left_edge_factor
        dh_k = scale * diss_h[:, :-1]
        h_r = self.h_right[:, :-1]
        h_k[:, :, :, 0] += dh_k
        if normal_only:
            un_r = (
                self.u_right[:, :-1] * n_x[:, :-1]
                + self.v_right[:, :-1] * n_y[:, :-1]
                + self.w_right[:, :-1] * n_z[:, :-1]
            )
            dun = (scale * diss_un[:, :-1] - un_r * dh_k) / h_r
            u_k[:, :, :, 0] += dun * n_x[:, :-1]
            v_k[:, :, :, 0] += dun * n_y[:, :-1]
            w_k[:, :, :, 0] += dun * n_z[:, :-1]
        else:
            dm_u = scale * diss_u[:, :-1]
            dm_v = scale * diss_v[:, :-1]
            dm_w = scale * diss_w[:, :-1]
            u_k[:, :, :, 0] += (dm_u - self.u_right[:, :-1] * dh_k) / h_r
            v_k[:, :, :, 0] += (dm_v - self.v_right[:, :-1] * dh_k) / h_r
            w_k[:, :, :, 0] += (dm_w - self.w_right[:, :-1] * dh_k) / h_r

    def _solve_numpy_lmars_kernel(self, u, v, w, h, t, dt, *, tangent_diss, verbose=False):

        # copy the boundaries across
        self.boundaries(u, v, w, h, t)

        # handle h
        h_xflux, h_yflux, h_zflux = self.hflux(u, v, w, h)
        h_xflux, h_yflux = self.phys_to_contra(h_xflux, h_yflux,
                                               h_zflux)  # flux is in contravariant form
        div = self.ddxi(h_xflux * self.J)
        div += self.ddeta(h_yflux * self.J)
        div /= self.J
        verbose = False

        out = -self.Jw * div

        h_up_flux_x, h_up_flux_y, h_up_flux_z = self.hflux(self.u_up, self.v_up, self.w_up, self.h_up)
        h_down_flux_x, h_down_flux_y, h_down_flux_z = self.hflux(self.u_down, self.v_down, self.w_down, self.h_down)
        h_right_flux_x, h_right_flux_y, h_right_flux_z = self.hflux(self.u_right, self.v_right, self.w_right, self.h_right)
        h_left_flux_x, h_left_flux_y, h_left_flux_z = self.hflux(self.u_left, self.v_left, self.w_left, self.h_left)

        # fluxes through boundary
        h_up_flux = h_up_flux_y * self.eta_y_up + h_up_flux_x * self.eta_x_up + h_up_flux_z * self.eta_z_up
        h_down_flux = h_down_flux_y * self.eta_y_down + h_down_flux_x * self.eta_x_down + h_down_flux_z * self.eta_z_down
        h_right_flux = h_right_flux_y * self.xi_y_right + h_right_flux_x * self.xi_x_right + h_right_flux_z * self.xi_z_right
        h_left_flux = h_left_flux_y * self.xi_y_left + h_left_flux_x * self.xi_x_left + h_left_flux_z * self.xi_z_left

        uv_up_flux = self.uv_flux(self.u_up, self.v_up, self.w_up, self.h_up)
        uv_down_flux = self.uv_flux(self.u_down, self.v_down, self.w_down, self.h_down)
        uv_right_flux = self.uv_flux(self.u_right, self.v_right, self.w_right, self.h_right)
        uv_left_flux = self.uv_flux(self.u_left, self.v_left, self.w_left, self.h_left)

        # upper boundary
        # c_up = self.wave_speed(self.u_up, self.v_up, self.w_up, self.h_up)
        # c_down = self.wave_speed(self.u_down, self.v_down, self.w_down, self.h_down)
        # c_ve = 0.5 * (c_up + c_down)
        # c_right = self.wave_speed(self.u_right, self.v_right, self.w_right, self.h_right)
        # c_left = self.wave_speed(self.u_left, self.v_left, self.w_left, self.h_left)
        # c_ho = 0.5 * (c_right + c_left)

        vel_up = h_up_flux / self.h_up
        vel_down = h_down_flux / self.h_down
        vel_right = h_right_flux / self.h_right
        vel_left = h_left_flux / self.h_left

        c_snd_ho = 0.5 * (np.sqrt(self.g * self.h_right) + np.sqrt(self.g * self.h_left)) - 0.25 * (vel_right - vel_left)
        c_snd_ve = 0.5 * (np.sqrt(self.g * self.h_up) + np.sqrt(self.g * self.h_down)) - 0.25 * (vel_up - vel_down)
        # c_ho = c_adv_horz + c_snd_ho
        # c_ve = c_adv_vert + c_snd_ve
        h_ve = 0.5 * (self.h_up + self.h_down)
        h_ho = 0.5 * (self.h_right + self.h_left)
        h_star_ve = c_snd_ve**2 / self.g
        h_star_ho = c_snd_ho**2 / self.g

        c_adv_vert = 0.5 * (vel_up + vel_down) + np.sqrt(self.g * self.h_down) - np.sqrt(self.g * self.h_up)
        c_adv_horz = 0.5 * (vel_right + vel_left) + np.sqrt(self.g * self.h_left) - np.sqrt(self.g * self.h_right)

        u_cov_up = self.u_up * self.dxdxi_up + self.v_up * self.dydxi_up + self.w_up * self.dzdxi_up
        u_cov_down = self.u_down * self.dxdxi_down + self.v_down * self.dydxi_down + self.w_down * self.dzdxi_down
        u_cov_right = self.u_right * self.dxdxi_right + self.v_right * self.dydxi_right + self.w_right * self.dzdxi_right
        u_cov_left = self.u_left * self.dxdxi_left + self.v_left * self.dydxi_left + self.w_left * self.dzdxi_left

        v_cov_up = self.u_up * self.dxdeta_up + self.v_up * self.dydeta_up + self.w_up * self.dzdeta_up
        v_cov_down = self.u_down * self.dxdeta_down + self.v_down * self.dydeta_down + self.w_down * self.dzdeta_down
        v_cov_right = self.u_right * self.dxdeta_right + self.v_right * self.dydeta_right + self.w_right * self.dzdeta_right
        v_cov_left = self.u_left * self.dxdeta_left + self.v_left * self.dydeta_left + self.w_left * self.dzdeta_left

        u_contra_up = self.u_up * self.dxidx_up + self.v_up * self.dxidy_up + self.w_up * self.dxidz_up
        u_contra_down = self.u_down * self.dxidx_down + self.v_down * self.dxidy_down + self.w_down * self.dxidz_down
        u_contra_right = self.u_right * self.dxidx_right + self.v_right * self.dxidy_right + self.w_right * self.dxidz_right
        u_contra_left = self.u_left * self.dxidx_left + self.v_left * self.dxidy_left + self.w_left * self.dxidz_left

        v_contra_up = self.u_up * self.detadx_up + self.v_up * self.detady_up + self.w_up * self.detadz_up
        v_contra_down = self.u_down * self.detadx_down + self.v_down * self.detady_down + self.w_down * self.detadz_down
        v_contra_right = self.u_right * self.detadx_right + self.v_right * self.detady_right + self.w_right * self.detadz_right
        v_contra_left = self.u_left * self.detadx_left + self.v_left * self.detady_left + self.w_left * self.detadz_left

        # h_flux_vert = c_adv_vert * h_ve - 0.5 * abs(c_adv_vert) * (self.h_up - self.h_down)
        # h_flux_horz = c_adv_horz * h_ho - 0.5 * abs(c_adv_horz) * (self.h_right - self.h_left)
        h_flux_vert = c_adv_vert * h_star_ve
        h_flux_horz = c_adv_horz * h_star_ho

        # uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux) - 0.5 * c_snd_ho * (h_right_flux - h_left_flux) / h_ho
        # uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux) - 0.5 * c_snd_ve * (h_up_flux - h_down_flux) / h_ve

        uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux) - 0.5 * (0.5 * vel_right**2 + 0.5 * vel_left**2 + self.g * self.h_right + self.g * self.h_left)
        uv_flux_horz += 0.5 * c_adv_horz**2 + self.g * h_star_ho
        uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux) - 0.5 * (0.5 * vel_up**2 + 0.5 * vel_down**2 + self.g * self.h_up + self.g * self.h_down)
        uv_flux_vert += 0.5 * c_adv_vert**2 + self.g * h_star_ve
        
        u_cov_vert_avg = (c_adv_vert < 0) * u_cov_up + (c_adv_vert >= 0) * u_cov_down
        v_cov_horz_avg = (c_adv_horz < 0) * v_cov_right + (c_adv_horz >= 0) * v_cov_left

        u_flux_vert_up = v_contra_up * u_cov_vert_avg
        u_flux_vert_down = v_contra_down * u_cov_vert_avg
        v_flux_vert_up = uv_flux_vert - u_contra_up * u_cov_vert_avg
        v_flux_vert_down = uv_flux_vert - u_contra_down * u_cov_vert_avg

        u_flux_horz_right = uv_flux_horz - v_contra_right * v_cov_horz_avg
        u_flux_horz_left = uv_flux_horz - v_contra_left * v_cov_horz_avg
        v_flux_horz_right = u_contra_right * v_cov_horz_avg
        v_flux_horz_left = u_contra_left * v_cov_horz_avg

        # h_flux_vert = 0.5 * (h_up_flux + h_down_flux) - self.a * c_snd_ve * (uv_up_flux - uv_down_flux) / self.g
        # h_flux_horz = 0.5 * (h_right_flux + h_left_flux) - self.a * c_snd_ho * (uv_right_flux - uv_left_flux) / self.g

        self.tmp1[:, :, -1] = (h_flux_vert[1:] - h_down_flux[1:]) * (self.weights_x * self.J_vertface[:, :, -1])
        self.tmp1[:, :, 0] = -(h_flux_vert[:-1] - h_up_flux[:-1]) * (self.weights_x * self.J_vertface[:, :, 0])
        self.tmp2[:, :, :, -1] = (h_flux_horz[:, 1:] - h_left_flux[:, 1:]) * (self.weights_x * self.J_horzface[..., -1])
        self.tmp2[:, :, :, 0] = -(h_flux_horz[:, :-1] - h_right_flux[:, :-1]) * (
                self.weights_x * self.J_horzface[..., 0])
        out -= (self.tmp1 + self.tmp2)

        h_k = out / self.Jw

        # u and v fluxes
        ########
        #######

        uv_flux = self.uv_flux(u, v, w, h)

        # alpha = np.maximum(c_ho / self.h_right, c_ho / self.h_left)
        # uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux) - self.a * (h_right_flux - h_left_flux) * alpha #(self.g / c_ho)
        # alpha = np.maximum(c_ve / self.h_up, c_ve / self.h_down)
        # uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux) - self.a * (h_up_flux - h_down_flux) * alpha #(self.g / c_ve) * (h_up_flux - h_down_flux)

        if self.b is not None:
            uv_flux += self.g * self.b

        u_contra, v_contra = self.phys_to_contra(u, v, w)
        u_cov, v_cov, _ = self.phys_to_cov(u, v, w)
        abs_vort_cov = self.ddxi(v_cov)
        abs_vort_cov += -self.ddeta(u_cov)
        abs_vort_cov += self.f * self.J

        #
        #
        u_cov_up = self.u_up * self.dxdxi_up + self.v_up * self.dydxi_up + self.w_up * self.dzdxi_up
        u_cov_down = self.u_down * self.dxdxi_down + self.v_down * self.dydxi_down + self.w_down * self.dzdxi_down
        u_cov_right = self.u_right * self.dxdxi_right + self.v_right * self.dydxi_right + self.w_right * self.dzdxi_right
        u_cov_left = self.u_left * self.dxdxi_left + self.v_left * self.dydxi_left + self.w_left * self.dzdxi_left

        v_cov_up = self.u_up * self.dxdeta_up + self.v_up * self.dydeta_up + self.w_up * self.dzdeta_up
        v_cov_down = self.u_down * self.dxdeta_down + self.v_down * self.dydeta_down + self.w_down * self.dzdeta_down
        v_cov_right = self.u_right * self.dxdeta_right + self.v_right * self.dydeta_right + self.w_right * self.dzdeta_right
        v_cov_left = self.u_left * self.dxdeta_left + self.v_left * self.dydeta_left + self.w_left * self.dzdeta_left

        u_contra_up = self.u_up * self.dxidx_up + self.v_up * self.dxidy_up + self.w_up * self.dxidz_up
        u_contra_down = self.u_down * self.dxidx_down + self.v_down * self.dxidy_down + self.w_down * self.dxidz_down
        u_contra_right = self.u_right * self.dxidx_right + self.v_right * self.dxidy_right + self.w_right * self.dxidz_right
        u_contra_left = self.u_left * self.dxidx_left + self.v_left * self.dxidy_left + self.w_left * self.dxidz_left

        v_contra_up = self.u_up * self.detadx_up + self.v_up * self.detady_up + self.w_up * self.detadz_up
        v_contra_down = self.u_down * self.detadx_down + self.v_down * self.detady_down + self.w_down * self.detadz_down
        v_contra_right = self.u_right * self.detadx_right + self.v_right * self.detady_right + self.w_right * self.detadz_right
        v_contra_left = self.u_left * self.detadx_left + self.v_left * self.detady_left + self.w_left * self.detadz_left

        # handle u
        #######
        ###

        u_k = -self.ddxi(uv_flux)
        u_k += v_contra * abs_vort_cov

        wx = self.weights_x.ravel()[-1]

        u_k[:, :, -1] -= (u_flux_vert_down[1:] - (v_contra_down * u_cov_down)[1:]) / wx
        u_k[:, :, 0] += (u_flux_vert_up[:-1] - (v_contra_up * u_cov_up)[:-1]) / wx
        u_k[:, :, :, -1] -= (
            u_flux_horz_left[:, 1:] - (uv_left_flux - v_contra_left * v_cov_left)[:, 1:]
        ) / wx
        u_k[:, :, :, 0] += (
            u_flux_horz_right[:, :-1] - (uv_right_flux - v_contra_right * v_cov_right)[:, :-1]
        ) / wx

        # handle v
        #######
        ###

        v_k = -self.ddeta(uv_flux)
        v_k += -u_contra * abs_vort_cov

        v_k[:, :, -1] -= (
            v_flux_vert_down[1:] - (uv_down_flux - u_contra_down * u_cov_down)[1:]
        ) / wx
        v_k[:, :, 0] += (
            v_flux_vert_up[:-1] - (uv_up_flux - u_contra_up * u_cov_up)[:-1]
        ) / wx
        v_k[:, :, :, -1] -= (
            v_flux_horz_left[:, 1:] - (u_contra_left * v_cov_left)[:, 1:]
        ) / wx
        v_k[:, :, :, 0] += (
            v_flux_horz_right[:, :-1] - (u_contra_right * v_cov_right)[:, :-1]
        ) / wx

        u_k, v_k, w_k = self.cov_to_phys(u_k, v_k, 0)

        return u_k, v_k, w_k, h_k

    def _apply_old_tangent_diss_numpy(self, u_k, v_k, w_k):
        h_up_flux = self.h_up * (
            self.u_up * self.eta_x_up
            + self.v_up * self.eta_y_up
            + self.w_up * self.eta_z_up
        )
        h_down_flux = self.h_down * (
            self.u_down * self.eta_x_down
            + self.v_down * self.eta_y_down
            + self.w_down * self.eta_z_down
        )
        h_right_flux = self.h_right * (
            self.u_right * self.xi_x_right
            + self.v_right * self.xi_y_right
            + self.w_right * self.xi_z_right
        )
        h_left_flux = self.h_left * (
            self.u_left * self.xi_x_left
            + self.v_left * self.xi_y_left
            + self.w_left * self.xi_z_left
        )

        vel_up = h_up_flux / self.h_up
        vel_down = h_down_flux / self.h_down
        vel_right = h_right_flux / self.h_right
        vel_left = h_left_flux / self.h_left
        h_ve = 0.5 * (self.h_up + self.h_down)
        h_ho = 0.5 * (self.h_right + self.h_left)
        c_adv_vert = 0.5 * abs(vel_up + vel_down)
        c_adv_horz = 0.5 * abs(vel_right + vel_left)

        u_cov_up = self.u_up * self.dxdxi_up + self.v_up * self.dydxi_up + self.w_up * self.dzdxi_up
        u_cov_down = self.u_down * self.dxdxi_down + self.v_down * self.dydxi_down + self.w_down * self.dzdxi_down
        u_cov_right = self.u_right * self.dxdxi_right + self.v_right * self.dydxi_right + self.w_right * self.dzdxi_right
        u_cov_left = self.u_left * self.dxdxi_left + self.v_left * self.dydxi_left + self.w_left * self.dzdxi_left

        v_cov_up = self.u_up * self.dxdeta_up + self.v_up * self.dydeta_up + self.w_up * self.dzdeta_up
        v_cov_down = self.u_down * self.dxdeta_down + self.v_down * self.dydeta_down + self.w_down * self.dzdeta_down
        v_cov_right = self.u_right * self.dxdeta_right + self.v_right * self.dydeta_right + self.w_right * self.dzdeta_right
        v_cov_left = self.u_left * self.dxdeta_left + self.v_left * self.dydeta_left + self.w_left * self.dzdeta_left

        wx = self.endpoint_weight

        diss_u = -0.5 * c_adv_vert * (self.h_up * u_cov_up - self.h_down * u_cov_down) / h_ve
        diss_v = -0.5 * c_adv_vert * (self.h_up * v_cov_up - self.h_down * v_cov_down) / h_ve
        diss_vel = 0.5 * c_adv_vert * (h_up_flux - h_down_flux) / h_ve

        du_cov = -diss_u[1:] * self.J_eta[:, :, -1] / wx
        dv_cov = -(diss_v[1:] * self.J_eta[:, :, -1] + diss_vel[1:]) / wx
        u_k[:, :, -1] += du_cov * self.dxidx[:, :, -1] + dv_cov * self.detadx[:, :, -1]
        v_k[:, :, -1] += du_cov * self.dxidy[:, :, -1] + dv_cov * self.detady[:, :, -1]
        w_k[:, :, -1] += du_cov * self.dxidz[:, :, -1] + dv_cov * self.detadz[:, :, -1]

        du_cov = diss_u[:-1] * self.J_eta[:, :, 0] / wx
        dv_cov = (diss_v[:-1] * self.J_eta[:, :, 0] + diss_vel[:-1]) / wx
        u_k[:, :, 0] += du_cov * self.dxidx[:, :, 0] + dv_cov * self.detadx[:, :, 0]
        v_k[:, :, 0] += du_cov * self.dxidy[:, :, 0] + dv_cov * self.detady[:, :, 0]
        w_k[:, :, 0] += du_cov * self.dxidz[:, :, 0] + dv_cov * self.detadz[:, :, 0]

        diss_u = -0.5 * c_adv_horz * (self.h_right * u_cov_right - self.h_left * u_cov_left) / h_ho
        diss_v = -0.5 * c_adv_horz * (self.h_right * v_cov_right - self.h_left * v_cov_left) / h_ho
        diss_vel = 0.5 * c_adv_horz * (h_right_flux - h_left_flux) / h_ho

        du_cov = -(diss_u[:, 1:] * self.J_xi[:, :, :, -1] + diss_vel[:, 1:]) / wx
        dv_cov = -diss_v[:, 1:] * self.J_xi[:, :, :, -1] / wx
        u_k[:, :, :, -1] += du_cov * self.dxidx[:, :, :, -1] + dv_cov * self.detadx[:, :, :, -1]
        v_k[:, :, :, -1] += du_cov * self.dxidy[:, :, :, -1] + dv_cov * self.detady[:, :, :, -1]
        w_k[:, :, :, -1] += du_cov * self.dxidz[:, :, :, -1] + dv_cov * self.detadz[:, :, :, -1]

        du_cov = (diss_u[:, :-1] * self.J_xi[:, :, :, 0] + diss_vel[:, :-1]) / wx
        dv_cov = diss_v[:, :-1] * self.J_xi[:, :, :, 0] / wx
        u_k[:, :, :, 0] += du_cov * self.dxidx[:, :, :, 0] + dv_cov * self.detadx[:, :, :, 0]
        v_k[:, :, :, 0] += du_cov * self.dxidy[:, :, :, 0] + dv_cov * self.detady[:, :, :, 0]
        w_k[:, :, :, 0] += du_cov * self.dxidz[:, :, :, 0] + dv_cov * self.detadz[:, :, :, 0]

    def phys_to_contra(self, u, v, w):
        u_contra = u * self.dxidx + v * self.dxidy + w * self.dxidz
        v_contra = u * self.detadx + v * self.detady + w * self.detadz
        return u_contra, v_contra

    def phys_to_cov(self, u, v, w):
        u_cov = u * self.dxdxi + v * self.dydxi + w * self.dzdxi
        v_cov = u * self.dxdeta + v * self.dydeta + w * self.dzdeta
        w_cov = u * self.dxdzeta + v * self.dydzeta + w * self.dzdzeta
        return u_cov, v_cov, w_cov

    def cov_to_phys(self, u_cov, v_cov, w_cov):
        u = u_cov * self.dxidx + v_cov * self.detadx + w_cov * self.dzetadx
        v = u_cov * self.dxidy + v_cov * self.detady + w_cov * self.dzetady
        w = u_cov * self.dxidz + v_cov * self.detadz + w_cov * self.dzetadz
        return u, v, w

    def contra_to_phys(self, u_contra, v_contra):
        u = u_contra * self.dxdxi + v_contra * self.dxdeta
        v = u_contra * self.dydxi + v_contra * self.dydeta
        w = u_contra * self.dzdxi + v_contra * self.dzdeta
        return u, v, w

    def k_dot_curl(self, u, v, w):
        u_cov, v_cov, w_cov = self.phys_to_cov(u, v, w)
        out = self.ddxi(v_cov) - self.ddeta(u_cov)
        return out / self.J

    def curl_k(self, psi):
        cov_psi_k = psi * self.k_cov_norm
        u_contra = -self.ddeta(cov_psi_k) / self.J
        v_contra = self.ddxi(cov_psi_k) / self.J

        u, v, w = self.contra_to_phys(u_contra, v_contra)
        return u, v, w

    def k_cross_grad(self, psi):
        u_cov = self.ddxi(psi)
        v_cov = self.ddeta(psi)

        u, v, w = self.cov_to_phys(u_cov, v_cov, 0)
        u, v, w = cross_product([self.kx, self.ky, self.kz], [u, v, w])
        return u, v, w


__all__ = [
    "DGCubedSphereSWENumpy",
    "DGCubedSphereFaceNumpy",
    "SIACKernel",
    "centered_cardinal_bspline",
    "siac_kernel",
]
