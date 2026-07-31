import numpy as np
from dg_swe.geometry import EquiangularFace, SadournyFace, face_name_from_cartesian, lat_long_to_cartesian
import os

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
    @njit(cache=True)
    def _solve_numba_kernel(
        u, v, w, h, b,
        D, edge_weights, endpoint_weight, J, Jw,
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
        J_vertface, J_horzface, J_eta, J_xi,
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

        h_up_flux = np.empty((ny + 1, nx, n), dtype=u.dtype)
        h_down_flux = np.empty((ny + 1, nx, n), dtype=u.dtype)
        h_right_flux = np.empty((ny, nx + 1, n), dtype=u.dtype)
        h_left_flux = np.empty((ny, nx + 1, n), dtype=u.dtype)

        uv_up_flux = np.empty((ny + 1, nx, n), dtype=u.dtype)
        uv_down_flux = np.empty((ny + 1, nx, n), dtype=u.dtype)
        uv_right_flux = np.empty((ny, nx + 1, n), dtype=u.dtype)
        uv_left_flux = np.empty((ny, nx + 1, n), dtype=u.dtype)

        h_flux_vert = np.empty((ny + 1, nx, n), dtype=u.dtype)
        h_flux_horz = np.empty((ny, nx + 1, n), dtype=u.dtype)
        uv_flux_vert = np.empty((ny + 1, nx, n), dtype=u.dtype)
        uv_flux_horz = np.empty((ny, nx + 1, n), dtype=u.dtype)
        vel_up = np.empty((ny + 1, nx, n), dtype=u.dtype)
        vel_down = np.empty((ny + 1, nx, n), dtype=u.dtype)
        vel_right = np.empty((ny, nx + 1, n), dtype=u.dtype)
        vel_left = np.empty((ny, nx + 1, n), dtype=u.dtype)
        c_adv_vert = np.empty((ny + 1, nx, n), dtype=u.dtype)
        c_adv_horz = np.empty((ny, nx + 1, n), dtype=u.dtype)
        h_ve = np.empty((ny + 1, nx, n), dtype=u.dtype)
        h_ho = np.empty((ny, nx + 1, n), dtype=u.dtype)

        u_cov_up = np.empty((ny + 1, nx, n), dtype=u.dtype)
        u_cov_down = np.empty((ny + 1, nx, n), dtype=u.dtype)
        u_cov_right = np.empty((ny, nx + 1, n), dtype=u.dtype)
        u_cov_left = np.empty((ny, nx + 1, n), dtype=u.dtype)
        v_cov_up = np.empty((ny + 1, nx, n), dtype=u.dtype)
        v_cov_down = np.empty((ny + 1, nx, n), dtype=u.dtype)
        v_cov_right = np.empty((ny, nx + 1, n), dtype=u.dtype)
        v_cov_left = np.empty((ny, nx + 1, n), dtype=u.dtype)

        u_contra_up = np.empty((ny + 1, nx, n), dtype=u.dtype)
        u_contra_down = np.empty((ny + 1, nx, n), dtype=u.dtype)
        u_contra_right = np.empty((ny, nx + 1, n), dtype=u.dtype)
        u_contra_left = np.empty((ny, nx + 1, n), dtype=u.dtype)
        v_contra_up = np.empty((ny + 1, nx, n), dtype=u.dtype)
        v_contra_down = np.empty((ny + 1, nx, n), dtype=u.dtype)
        v_contra_right = np.empty((ny, nx + 1, n), dtype=u.dtype)
        v_contra_left = np.empty((ny, nx + 1, n), dtype=u.dtype)

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
                        u_contra[ey, ex, eta, xi] = ux
                        v_contra[ey, ex, eta, xi] = uy
                        hx = hh * ux
                        hy = hh * uy
                        h_xcontra_J[ey, ex, eta, xi] = hx * J[ey, ex, eta, xi]
                        h_ycontra_J[ey, ex, eta, xi] = hy * J[ey, ex, eta, xi]
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
                            ddxi_h += h_xcontra_J[ey, ex, eta, l] * D[l, xi]
                            ddeta_h += D[l, eta] * h_ycontra_J[ey, ex, l, xi]
                            ddxi_uv += uv_flux[ey, ex, eta, l] * D[l, xi]
                            ddeta_uv += D[l, eta] * uv_flux[ey, ex, l, xi]
                            ddxi_vcov += v_cov[ey, ex, eta, l] * D[l, xi]
                            ddeta_ucov += D[l, eta] * u_cov[ey, ex, l, xi]
                        h_k[ey, ex, eta, xi] = -(ddxi_h + ddeta_h) / J[ey, ex, eta, xi]
                        abs_vort_cov = ddxi_vcov - ddeta_ucov + f[ey, ex, eta, xi] * J[ey, ex, eta, xi]
                        u_k[ey, ex, eta, xi] = -ddxi_uv + v_contra[ey, ex, eta, xi] * abs_vort_cov
                        v_k[ey, ex, eta, xi] = -ddeta_uv - u_contra[ey, ex, eta, xi] * abs_vort_cov

        for ey in range(ny + 1):
            for ex in range(nx):
                for xi in range(n):
                    h_up_flux[ey, ex, xi] = h_up[ey, ex, xi] * (
                        u_up[ey, ex, xi] * eta_x_up[ey, ex, xi]
                        + v_up[ey, ex, xi] * eta_y_up[ey, ex, xi]
                        + w_up[ey, ex, xi] * eta_z_up[ey, ex, xi]
                    )
                    h_down_flux[ey, ex, xi] = h_down[ey, ex, xi] * (
                        u_down[ey, ex, xi] * eta_x_down[ey, ex, xi]
                        + v_down[ey, ex, xi] * eta_y_down[ey, ex, xi]
                        + w_down[ey, ex, xi] * eta_z_down[ey, ex, xi]
                    )
                    uv_up_flux[ey, ex, xi] = 0.5 * (
                        u_up[ey, ex, xi] ** 2 + v_up[ey, ex, xi] ** 2 + w_up[ey, ex, xi] ** 2
                    ) + g * h_up[ey, ex, xi]
                    uv_down_flux[ey, ex, xi] = 0.5 * (
                        u_down[ey, ex, xi] ** 2 + v_down[ey, ex, xi] ** 2 + w_down[ey, ex, xi] ** 2
                    ) + g * h_down[ey, ex, xi]
                    vel_up[ey, ex, xi] = h_up_flux[ey, ex, xi] / h_up[ey, ex, xi]
                    vel_down[ey, ex, xi] = h_down_flux[ey, ex, xi] / h_down[ey, ex, xi]
                    c_adv_vert[ey, ex, xi] = 0.5 * abs(vel_up[ey, ex, xi] + vel_down[ey, ex, xi])
                    c_snd = 0.5 * (np.sqrt(g * h_up[ey, ex, xi]) + np.sqrt(g * h_down[ey, ex, xi]))
                    h_ve[ey, ex, xi] = 0.5 * (h_up[ey, ex, xi] + h_down[ey, ex, xi])
                    h_flux_vert[ey, ex, xi] = 0.5 * (h_up_flux[ey, ex, xi] + h_down_flux[ey, ex, xi]) - ah * c_snd * (h_up[ey, ex, xi] - h_down[ey, ex, xi])
                    uv_flux_vert[ey, ex, xi] = 0.5 * (uv_up_flux[ey, ex, xi] + uv_down_flux[ey, ex, xi]) - a * c_snd * (h_up_flux[ey, ex, xi] - h_down_flux[ey, ex, xi]) / h_ve[ey, ex, xi]

                    u_cov_up[ey, ex, xi] = u_up[ey, ex, xi] * dxdxi_up[ey, ex, xi] + v_up[ey, ex, xi] * dydxi_up[ey, ex, xi] + w_up[ey, ex, xi] * dzdxi_up[ey, ex, xi]
                    u_cov_down[ey, ex, xi] = u_down[ey, ex, xi] * dxdxi_down[ey, ex, xi] + v_down[ey, ex, xi] * dydxi_down[ey, ex, xi] + w_down[ey, ex, xi] * dzdxi_down[ey, ex, xi]
                    v_cov_up[ey, ex, xi] = u_up[ey, ex, xi] * dxdeta_up[ey, ex, xi] + v_up[ey, ex, xi] * dydeta_up[ey, ex, xi] + w_up[ey, ex, xi] * dzdeta_up[ey, ex, xi]
                    v_cov_down[ey, ex, xi] = u_down[ey, ex, xi] * dxdeta_down[ey, ex, xi] + v_down[ey, ex, xi] * dydeta_down[ey, ex, xi] + w_down[ey, ex, xi] * dzdeta_down[ey, ex, xi]
                    u_contra_up[ey, ex, xi] = u_up[ey, ex, xi] * dxidx_up[ey, ex, xi] + v_up[ey, ex, xi] * dxidy_up[ey, ex, xi] + w_up[ey, ex, xi] * dxidz_up[ey, ex, xi]
                    u_contra_down[ey, ex, xi] = u_down[ey, ex, xi] * dxidx_down[ey, ex, xi] + v_down[ey, ex, xi] * dxidy_down[ey, ex, xi] + w_down[ey, ex, xi] * dxidz_down[ey, ex, xi]
                    v_contra_up[ey, ex, xi] = u_up[ey, ex, xi] * detadx_up[ey, ex, xi] + v_up[ey, ex, xi] * detady_up[ey, ex, xi] + w_up[ey, ex, xi] * detadz_up[ey, ex, xi]
                    v_contra_down[ey, ex, xi] = u_down[ey, ex, xi] * detadx_down[ey, ex, xi] + v_down[ey, ex, xi] * detady_down[ey, ex, xi] + w_down[ey, ex, xi] * detadz_down[ey, ex, xi]

        for ey in range(ny):
            for ex in range(nx + 1):
                for eta in range(n):
                    h_right_flux[ey, ex, eta] = h_right[ey, ex, eta] * (
                        u_right[ey, ex, eta] * xi_x_right[ey, ex, eta]
                        + v_right[ey, ex, eta] * xi_y_right[ey, ex, eta]
                        + w_right[ey, ex, eta] * xi_z_right[ey, ex, eta]
                    )
                    h_left_flux[ey, ex, eta] = h_left[ey, ex, eta] * (
                        u_left[ey, ex, eta] * xi_x_left[ey, ex, eta]
                        + v_left[ey, ex, eta] * xi_y_left[ey, ex, eta]
                        + w_left[ey, ex, eta] * xi_z_left[ey, ex, eta]
                    )
                    uv_right_flux[ey, ex, eta] = 0.5 * (
                        u_right[ey, ex, eta] ** 2 + v_right[ey, ex, eta] ** 2 + w_right[ey, ex, eta] ** 2
                    ) + g * h_right[ey, ex, eta]
                    uv_left_flux[ey, ex, eta] = 0.5 * (
                        u_left[ey, ex, eta] ** 2 + v_left[ey, ex, eta] ** 2 + w_left[ey, ex, eta] ** 2
                    ) + g * h_left[ey, ex, eta]
                    vel_right[ey, ex, eta] = h_right_flux[ey, ex, eta] / h_right[ey, ex, eta]
                    vel_left[ey, ex, eta] = h_left_flux[ey, ex, eta] / h_left[ey, ex, eta]
                    c_adv_horz[ey, ex, eta] = 0.5 * abs(vel_right[ey, ex, eta] + vel_left[ey, ex, eta])
                    c_snd = 0.5 * (np.sqrt(g * h_right[ey, ex, eta]) + np.sqrt(g * h_left[ey, ex, eta]))
                    h_ho[ey, ex, eta] = 0.5 * (h_right[ey, ex, eta] + h_left[ey, ex, eta])
                    h_flux_horz[ey, ex, eta] = 0.5 * (h_right_flux[ey, ex, eta] + h_left_flux[ey, ex, eta]) - ah * c_snd * (h_right[ey, ex, eta] - h_left[ey, ex, eta])
                    uv_flux_horz[ey, ex, eta] = 0.5 * (uv_right_flux[ey, ex, eta] + uv_left_flux[ey, ex, eta]) - a * c_snd * (h_right_flux[ey, ex, eta] - h_left_flux[ey, ex, eta]) / h_ho[ey, ex, eta]

                    u_cov_right[ey, ex, eta] = u_right[ey, ex, eta] * dxdxi_right[ey, ex, eta] + v_right[ey, ex, eta] * dydxi_right[ey, ex, eta] + w_right[ey, ex, eta] * dzdxi_right[ey, ex, eta]
                    u_cov_left[ey, ex, eta] = u_left[ey, ex, eta] * dxdxi_left[ey, ex, eta] + v_left[ey, ex, eta] * dydxi_left[ey, ex, eta] + w_left[ey, ex, eta] * dzdxi_left[ey, ex, eta]
                    v_cov_right[ey, ex, eta] = u_right[ey, ex, eta] * dxdeta_right[ey, ex, eta] + v_right[ey, ex, eta] * dydeta_right[ey, ex, eta] + w_right[ey, ex, eta] * dzdeta_right[ey, ex, eta]
                    v_cov_left[ey, ex, eta] = u_left[ey, ex, eta] * dxdeta_left[ey, ex, eta] + v_left[ey, ex, eta] * dydeta_left[ey, ex, eta] + w_left[ey, ex, eta] * dzdeta_left[ey, ex, eta]
                    u_contra_right[ey, ex, eta] = u_right[ey, ex, eta] * dxidx_right[ey, ex, eta] + v_right[ey, ex, eta] * dxidy_right[ey, ex, eta] + w_right[ey, ex, eta] * dxidz_right[ey, ex, eta]
                    u_contra_left[ey, ex, eta] = u_left[ey, ex, eta] * dxidx_left[ey, ex, eta] + v_left[ey, ex, eta] * dxidy_left[ey, ex, eta] + w_left[ey, ex, eta] * dxidz_left[ey, ex, eta]
                    v_contra_right[ey, ex, eta] = u_right[ey, ex, eta] * detadx_right[ey, ex, eta] + v_right[ey, ex, eta] * detady_right[ey, ex, eta] + w_right[ey, ex, eta] * detadz_right[ey, ex, eta]
                    v_contra_left[ey, ex, eta] = u_left[ey, ex, eta] * detadx_left[ey, ex, eta] + v_left[ey, ex, eta] * detady_left[ey, ex, eta] + w_left[ey, ex, eta] * detadz_left[ey, ex, eta]

        for ey in range(ny):
            for ex in range(nx):
                for xi in range(n):
                    edge_w = edge_weights[xi]
                    wx = endpoint_weight
                    h_k[ey, ex, n - 1, xi] -= (h_flux_vert[ey + 1, ex, xi] - h_down_flux[ey + 1, ex, xi]) * J_vertface[ey, ex, n - 1, xi] * edge_w / Jw[ey, ex, n - 1, xi]
                    h_k[ey, ex, 0, xi] -= -(h_flux_vert[ey, ex, xi] - h_up_flux[ey, ex, xi]) * J_vertface[ey, ex, 0, xi] * edge_w / Jw[ey, ex, 0, xi]

                    avg_tan_cov = 0.5 * (u_cov_up[ey + 1, ex, xi] + u_cov_down[ey + 1, ex, xi])
                    tan_flux_delta = avg_tan_cov - u_cov_down[ey + 1, ex, xi]
                    u_k[ey, ex, n - 1, xi] += -v_contra_down[ey + 1, ex, xi] * tan_flux_delta / wx
                    v_k[ey, ex, n - 1, xi] += u_contra_down[ey + 1, ex, xi] * tan_flux_delta / wx

                    avg_tan_cov = 0.5 * (u_cov_up[ey, ex, xi] + u_cov_down[ey, ex, xi])
                    tan_flux_delta = avg_tan_cov - u_cov_up[ey, ex, xi]
                    u_k[ey, ex, 0, xi] += v_contra_up[ey, ex, xi] * tan_flux_delta / wx
                    v_k[ey, ex, n - 1, xi] -= (uv_flux_vert[ey + 1, ex, xi] - uv_down_flux[ey + 1, ex, xi]) / wx
                    v_k[ey, ex, 0, xi] -= -(uv_flux_vert[ey, ex, xi] - uv_up_flux[ey, ex, xi]) / wx
                    v_k[ey, ex, 0, xi] += -u_contra_up[ey, ex, xi] * tan_flux_delta / wx

                    if tangent_diss:
                        diss = -0.5 * c_adv_vert[ey + 1, ex, xi] * (h_up[ey + 1, ex, xi] * u_cov_up[ey + 1, ex, xi] - h_down[ey + 1, ex, xi] * u_cov_down[ey + 1, ex, xi]) / h_ve[ey + 1, ex, xi]
                        u_k[ey, ex, n - 1, xi] -= diss * J_eta[ey, ex, n - 1, xi] / wx
                        diss = -0.5 * c_adv_vert[ey, ex, xi] * (h_up[ey, ex, xi] * u_cov_up[ey, ex, xi] - h_down[ey, ex, xi] * u_cov_down[ey, ex, xi]) / h_ve[ey, ex, xi]
                        u_k[ey, ex, 0, xi] += diss * J_eta[ey, ex, 0, xi] / wx

                        diss = -0.5 * c_adv_vert[ey + 1, ex, xi] * (h_up[ey + 1, ex, xi] * v_cov_up[ey + 1, ex, xi] - h_down[ey + 1, ex, xi] * v_cov_down[ey + 1, ex, xi]) / h_ve[ey + 1, ex, xi]
                        v_k[ey, ex, n - 1, xi] -= diss * J_eta[ey, ex, n - 1, xi] / wx
                        diss = -0.5 * c_adv_vert[ey, ex, xi] * (h_up[ey, ex, xi] * v_cov_up[ey, ex, xi] - h_down[ey, ex, xi] * v_cov_down[ey, ex, xi]) / h_ve[ey, ex, xi]
                        v_k[ey, ex, 0, xi] += diss * J_eta[ey, ex, 0, xi] / wx

                        diss = 0.5 * c_adv_vert[ey + 1, ex, xi] * (h_up[ey + 1, ex, xi] * vel_up[ey + 1, ex, xi] - h_down[ey + 1, ex, xi] * vel_down[ey + 1, ex, xi]) / h_ve[ey + 1, ex, xi]
                        v_k[ey, ex, n - 1, xi] -= diss / wx
                        diss = 0.5 * c_adv_vert[ey, ex, xi] * (h_up[ey, ex, xi] * vel_up[ey, ex, xi] - h_down[ey, ex, xi] * vel_down[ey, ex, xi]) / h_ve[ey, ex, xi]
                        v_k[ey, ex, 0, xi] += diss / wx

                for eta in range(n):
                    edge_w = edge_weights[eta]
                    wx = endpoint_weight
                    h_k[ey, ex, eta, n - 1] -= (h_flux_horz[ey, ex + 1, eta] - h_left_flux[ey, ex + 1, eta]) * J_horzface[ey, ex, eta, n - 1] * edge_w / Jw[ey, ex, eta, n - 1]
                    h_k[ey, ex, eta, 0] -= -(h_flux_horz[ey, ex, eta] - h_right_flux[ey, ex, eta]) * J_horzface[ey, ex, eta, 0] * edge_w / Jw[ey, ex, eta, 0]

                    u_k[ey, ex, eta, n - 1] -= (uv_flux_horz[ey, ex + 1, eta] - uv_left_flux[ey, ex + 1, eta]) / wx
                    u_k[ey, ex, eta, 0] -= -(uv_flux_horz[ey, ex, eta] - uv_right_flux[ey, ex, eta]) / wx

                    avg_tan_cov = 0.5 * (v_cov_right[ey, ex + 1, eta] + v_cov_left[ey, ex + 1, eta])
                    tan_flux_delta = avg_tan_cov - v_cov_left[ey, ex + 1, eta]
                    u_k[ey, ex, eta, n - 1] += v_contra_left[ey, ex + 1, eta] * tan_flux_delta / wx
                    v_k[ey, ex, eta, n - 1] += -u_contra_left[ey, ex + 1, eta] * tan_flux_delta / wx

                    avg_tan_cov = 0.5 * (v_cov_right[ey, ex, eta] + v_cov_left[ey, ex, eta])
                    tan_flux_delta = avg_tan_cov - v_cov_right[ey, ex, eta]
                    u_k[ey, ex, eta, 0] += -v_contra_right[ey, ex, eta] * tan_flux_delta / wx
                    v_k[ey, ex, eta, 0] += u_contra_right[ey, ex, eta] * tan_flux_delta / wx

                    if tangent_diss:
                        diss = -0.5 * c_adv_horz[ey, ex + 1, eta] * (h_right[ey, ex + 1, eta] * u_cov_right[ey, ex + 1, eta] - h_left[ey, ex + 1, eta] * u_cov_left[ey, ex + 1, eta]) / h_ho[ey, ex + 1, eta]
                        u_k[ey, ex, eta, n - 1] -= diss * J_xi[ey, ex, eta, n - 1] / wx
                        diss = -0.5 * c_adv_horz[ey, ex, eta] * (h_right[ey, ex, eta] * u_cov_right[ey, ex, eta] - h_left[ey, ex, eta] * u_cov_left[ey, ex, eta]) / h_ho[ey, ex, eta]
                        u_k[ey, ex, eta, 0] += diss * J_xi[ey, ex, eta, 0] / wx

                        diss = 0.5 * c_adv_horz[ey, ex + 1, eta] * (h_right_flux[ey, ex + 1, eta] - h_left_flux[ey, ex + 1, eta]) / h_ho[ey, ex + 1, eta]
                        u_k[ey, ex, eta, n - 1] -= diss / wx
                        diss = 0.5 * c_adv_horz[ey, ex, eta] * (h_right_flux[ey, ex, eta] - h_left_flux[ey, ex, eta]) / h_ho[ey, ex, eta]
                        u_k[ey, ex, eta, 0] += diss / wx

                        diss = -0.5 * c_adv_horz[ey, ex + 1, eta] * (h_right[ey, ex + 1, eta] * v_cov_right[ey, ex + 1, eta] - h_left[ey, ex + 1, eta] * v_cov_left[ey, ex + 1, eta]) / h_ho[ey, ex + 1, eta]
                        v_k[ey, ex, eta, n - 1] -= diss * J_xi[ey, ex, eta, n - 1] / wx
                        diss = -0.5 * c_adv_horz[ey, ex, eta] * (h_right[ey, ex, eta] * v_cov_right[ey, ex, eta] - h_left[ey, ex, eta] * v_cov_left[ey, ex, eta]) / h_ho[ey, ex, eta]
                        v_k[ey, ex, eta, 0] += diss * J_xi[ey, ex, eta, 0] / wx

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
else:
    _solve_numba_kernel = None


class DGCubedSphereSWENumpy:
    def __init__(
            self, poly_order, nx, ny, g, f, eps, radius=1.0, device='cpu',
            solution=None, a=0.0, ah=0.0, dtype=np.float32, damping=None,
            tau_func=lambda t, dt: t, tau=0, tangent_diss=False,
            nprocx=1, nprocy=1, comm=None, **kwargs):

        self.face_names = ['zp', 'zn', 'xp', 'xn', 'yp', 'yn']
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

        self.faces = {
            name: DGCubedSphereFaceNumpy(
                name, poly_order, nx, ny, g, f, radius, eps, device, a=a, ah=ah, dtype=dtype,
                damping=None, bc='', tau=tau, tangent_diss=tangent_diss,
                x_proc_idx=self.x_proc_idx if self.parallel else 0,
                y_proc_idx=self.y_proc_idx if self.parallel else 0,
                nprocx=self.nprocx if self.parallel else 1,
                nprocy=self.nprocy if self.parallel else 1,
            )
            for name in self.active_face_names
        }
        self.time = 0
        self.cdt = min(self.faces[n].cdt for n in self.active_face_names)
        self.damping = damping
        self.tau_func = tau_func
        self.flag = True
        self.tangent_diss = tangent_diss

        self.time_list = []
        self.energy_list = []
        self.enstrophy_list = []
        self.mass_list = []
        self.vorticity_list = []
        self.vorticity_diagnostic = True # calculates a continuous diagnostic vorticity for plotting

    @staticmethod
    def _get_comm(comm, nprocx, nprocy):
        if comm is not None:
            return comm
        if nprocx * nprocy == 1:
            return None
        try:
            from mpi4py import MPI
        except ImportError as exc:
            raise ImportError("mpi4py is required when nprocx*nprocy > 1.") from exc
        return MPI.COMM_WORLD

    @property
    def nproc(self):
        return 6 * self.nprocx * self.nprocy

    @property
    def x_proc_idx(self):
        return (self.rank - self.tile_idx * self.nprocx * self.nprocy) // self.nprocy

    @property
    def y_proc_idx(self):
        return (self.rank - self.tile_idx * self.nprocx * self.nprocy) % self.nprocy

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
            self._exchange_boundaries_mpi(sol)
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
    def _connection_for_side(face, side):
        for conn in face.connections:
            _, (i1, _) = conn
            if i1 == side:
                return conn
        raise ValueError(f"No cubed-sphere connection for side {side}.")

    @staticmethod
    def _opposite_side(side):
        return {0: 2, 1: 3, 2: 0, 3: 1}[side]

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

    def _neighbor_rank(self, face, side):
        if side == 0:
            if self.x_proc_idx < self.nprocx - 1:
                return self.rank + self.nprocy
            conn = self._connection_for_side(face, side)
            return self.get_proc((conn[0], conn[1][1]), self.y_proc_idx)
        elif side == 2:
            if self.x_proc_idx > 0:
                return self.rank - self.nprocy
            conn = self._connection_for_side(face, side)
            return self.get_proc((conn[0], conn[1][1]), self.y_proc_idx)
        elif side == 1:
            if self.y_proc_idx < self.nprocy - 1:
                return self.rank + 1
            conn = self._connection_for_side(face, side)
            return self.get_proc((conn[0], conn[1][1]), self.x_proc_idx)
        elif side == 3:
            if self.y_proc_idx > 0:
                return self.rank - 1
            conn = self._connection_for_side(face, side)
            return self.get_proc((conn[0], conn[1][1]), self.x_proc_idx)
        else:
            raise ValueError(f"Unknown boundary side {side}.")

    def _exchange_boundaries_mpi(self, sol):
        face = self.faces[self.face_name]
        state = sol[self.face_name]
        reqs = []
        recv_edges = []
        send_edges = []
        for side in (0, 1, 2, 3):
            peer = self._neighbor_rank(face, side)
            send = self._edge_state(face, state, side)
            recv = np.empty_like(send)
            send_edges.append(send)
            recv_edges.append((side, recv))
            reqs.append(self.comm.Irecv(recv, source=peer, tag=self._opposite_side(side)))
            reqs.append(self.comm.Isend(send, dest=peer, tag=side))

        for req in reqs:
            req.Wait()
        for side, recv in recv_edges:
            self._assign_edge_state(face, side, recv)

    def get_dt(self):
        return min(face.get_dt() for face in self.faces.values())

    def positivity_preserving_limiter(self, state):
        for n in self.active_face_names:
            # if state[n][3].min() < 0:
                # print('0 detected')

            cell_means = (state[n][3] * self.faces[n].Jw).sum(axis=(2, 3)) / self.faces[n].Jw.sum(axis=(2, 3))
            cell_diffs = state[n][3] - cell_means[..., None, None]

            cell_mins = state[n][3].min(axis=(2, 3))
            diff_min = cell_mins - cell_means

            target_min = np.minimum(5.0, cell_means)
            needs_limiting = cell_mins < target_min
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
                print("Unfixable")

        return state


    def time_step(self, dt=None, order=3, forcing=None):
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
            u_1 = self.positivity_preserving_limiter(u_1)
            self.boundaries(u_1)
            k_2 = {n: self.faces[n].solve(*u_1[n], self.time, dt) for n in self.active_face_names}

            u_2 = {n: tuple(0.75 * u[n][i] + 0.25 * (u_1[n][i] + dt * k_2[n][i]) for i in range(4)) for n in self.active_face_names}
            u_2 = self.positivity_preserving_limiter(u_2)
            self.boundaries(u_2)
            k_3 = {n: self.faces[n].solve(*u_2[n], self.time, dt) for n in self.active_face_names}

            for n in self.active_face_names:
                self.faces[n].u = (self.faces[n].u + 2 * (u_2[n][0] + dt * k_3[n][0])) / 3
                self.faces[n].v = (self.faces[n].v + 2 * (u_2[n][1] + dt * k_3[n][1])) / 3
                self.faces[n].w = (self.faces[n].w + 2 * (u_2[n][2] + dt * k_3[n][2])) / 3
                self.faces[n].h = (self.faces[n].h + 2 * (u_2[n][3] + dt * k_3[n][3])) / 3

            u = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}
            u = self.positivity_preserving_limiter(u)
            self.boundaries(u)

        elif order == 34:
            u = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}
            self.boundaries(u)
            k_1 = {n: self.faces[n].solve(*u[n], self.time, dt) for n in self.active_face_names}

            u_1 = {n: tuple(u[n][i] + 0.5 * dt * k_1[n][i] for i in range(4)) for n in self.active_face_names}
            u_1 = self.positivity_preserving_limiter(u_1)
            self.boundaries(u_1)
            k_2 = {n: self.faces[n].solve(*u_1[n], self.time, dt) for n in self.active_face_names}

            u_2 = {n: tuple(u_1[n][i] + 0.5 * dt * k_2[n][i] for i in range(4)) for n in self.active_face_names}
            u_2 = self.positivity_preserving_limiter(u_2)
            self.boundaries(u_2)
            k_3 = {n: self.faces[n].solve(*u_2[n], self.time, dt) for n in self.active_face_names}

            u_3 = {n: tuple((2 / 3) * u[n][i] + (1 / 3) * u_2[n][i] + (1 / 6) * dt * k_3[n][i] for i in range(4)) for n in self.active_face_names}
            u_3 = self.positivity_preserving_limiter(u_3)
            self.boundaries(u_3)
            k_4 = {n: self.faces[n].solve(*u_3[n], self.time, dt) for n in self.active_face_names}

            for n in self.active_face_names:
                self.faces[n].u = u_3[n][0] + 0.5 * dt * k_4[n][0]
                self.faces[n].v = u_3[n][1] + 0.5 * dt * k_4[n][1]
                self.faces[n].w = u_3[n][2] + 0.5 * dt * k_4[n][2]
                self.faces[n].h = u_3[n][3] + 0.5 * dt * k_4[n][3]

            u = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}
            u = self.positivity_preserving_limiter(u)
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

    def save_restart(self, fn_template, directory):
        vars = ['u', 'v', 'w', 'h']
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
            data = [np.load(self.make_fp(vars[i], name, fn_template, directory)) for i in range(len(vars))]
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
            solution=None, a=0.0, ah=0.0, dtype=np.float32, damping=None,
            tau_func=lambda t, dt: t, bc='wall', tau=0.0, tangent_diss=False,
            x_proc_idx=0, y_proc_idx=0, nprocx=1, nprocy=1, **kwargs):

        valid_names = ['zp', 'zn', 'xp', 'xn', 'yp', 'yn']
        if not name in valid_names:
            raise ValueError(f'name: expected one of: {valid_names}. Found {name}.')
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
        self.damping = damping
        self.tau_func = tau_func
        self.xperiodic = self.yperiodic = False
        self.bc = bc
        self.geometry = EquiangularFace(name, radius=radius)
        self.connections = self.geometry.connections
        self.tau = tau
        self.tangent_diss = tangent_diss

        [xs_1d, w_x] = gll(poly_order, iterative=True)
        [y_1d, w_y] = gll(poly_order, iterative=True)
        self.gll_nodes = xs_1d

        global_nx = nx - 1
        global_ny = ny - 1
        if global_nx % nprocx != 0:
            raise ValueError(f"nx - 1 must be divisible by nprocx; got nx={nx}, nprocx={nprocx}.")
        if global_ny % nprocy != 0:
            raise ValueError(f"ny - 1 must be divisible by nprocy; got ny={ny}, nprocy={nprocy}.")

        self.global_nx = global_nx
        self.global_ny = global_ny
        self.x_proc_idx = x_proc_idx
        self.y_proc_idx = y_proc_idx
        self.nprocx = nprocx
        self.nprocy = nprocy

        x_edges = np.linspace(-0.5, 0.5, nx)
        y_edges = np.linspace(-0.5, 0.5, ny)
        local_nx = global_nx // nprocx
        local_ny = global_ny // nprocy
        x_start = x_proc_idx * local_nx
        y_start = y_proc_idx * local_ny
        xs = x_edges[x_start:x_start + local_nx + 1]
        ys = y_edges[y_start:y_start + local_ny + 1]
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
        return _solve_numba_kernel(
            u, v, w, h, self.b,
            self.D, self.edge_weights, self.endpoint_weight, self.J, self.Jw,
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
            self.J_vertface, self.J_horzface, self.J_eta, self.J_xi,
            self.dxdxi_up, self.dydxi_up, self.dzdxi_up, self.dxdxi_down, self.dydxi_down, self.dzdxi_down,
            self.dxdxi_right, self.dydxi_right, self.dzdxi_right, self.dxdxi_left, self.dydxi_left, self.dzdxi_left,
            self.dxdeta_up, self.dydeta_up, self.dzdeta_up, self.dxdeta_down, self.dydeta_down, self.dzdeta_down,
            self.dxdeta_right, self.dydeta_right, self.dzdeta_right, self.dxdeta_left, self.dydeta_left, self.dzdeta_left,
            self.g, self.a, self.ah, self.tangent_diss,
        )

    def solve_numpy(self, u, v, w, h, t, dt, *, verbose=False):

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

        c_adv_vert = 0.5 * abs(vel_up + vel_down)
        c_adv_horz = 0.5 * abs(vel_right + vel_left)

        c_snd_ho = 0.5 * (np.sqrt(self.g * self.h_right) + np.sqrt(self.g * self.h_left))
        c_snd_ve = 0.5 * (np.sqrt(self.g * self.h_up) + np.sqrt(self.g * self.h_down))
        # c_ho = c_adv_horz + c_snd_ho
        # c_ve = c_adv_vert + c_snd_ve
        h_ve = 0.5 * (self.h_up + self.h_down)
        h_ho = 0.5 * (self.h_right + self.h_left)

        h_flux_vert = 0.5 * (h_up_flux + h_down_flux) - self.ah * c_snd_ve * (self.h_up - self.h_down)
        h_flux_horz = 0.5 * (h_right_flux + h_left_flux) - self.ah * c_snd_ho * (self.h_right - self.h_left)

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

        uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux) - self.a * c_snd_ho * (h_right_flux - h_left_flux) / h_ho
        uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux) - self.a * c_snd_ve * (h_up_flux - h_down_flux) / h_ve

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

        u_k[:, :, :, -1] -= (uv_flux_horz - uv_left_flux)[:, 1:] / wx
        u_k[:, :, :, 0] -= -(uv_flux_horz - uv_right_flux)[:, :-1] / wx

        u_cov_vert_avg = 0.5 * (u_cov_up + u_cov_down)
        v_cov_horz_avg = 0.5 * (v_cov_right + v_cov_left)

        u_k[:, :, -1] += -v_contra_down[1:] * (u_cov_vert_avg - u_cov_down)[1:] / wx
        u_k[:, :, 0] += v_contra_up[:-1] * (u_cov_vert_avg - u_cov_up)[:-1] / wx
        u_k[:, :, :, -1] += v_contra_left[:, 1:] * (v_cov_horz_avg - v_cov_left)[:, 1:] / wx
        u_k[:, :, :, 0] += -v_contra_right[:, :-1] * (v_cov_horz_avg - v_cov_right)[:, :-1] / wx

        if self.tangent_diss:
            diss = -0.5 * c_adv_vert * (self.h_up * u_cov_up - self.h_down * u_cov_down) / h_ve
            u_k[:, :, -1] -= diss[1:] * self.J_eta[:, :, -1] / wx
            u_k[:, :, 0] += diss[:-1] * self.J_eta[:, :, 0] / wx

            diss = -0.5 * c_adv_horz * (self.h_right * u_cov_right - self.h_left * u_cov_left) / h_ho
            u_k[:, :, :, -1] -= diss[:, 1:] * self.J_xi[:, :, :, -1] / wx
            u_k[:, :, :, 0] += diss[:, :-1] * self.J_xi[:, :, :, 0] / wx

            diss = 0.5 * c_adv_horz * (h_right_flux - h_left_flux) / h_ho
            u_k[:, :, :, -1] -= diss[:, 1:] / wx
            u_k[:, :, :, 0] += diss[:, :-1] / wx

        # handle v
        #######
        ###

        v_k = -self.ddeta(uv_flux)
        v_k += -u_contra * abs_vort_cov

        v_k[:, :, -1] -= (uv_flux_vert - uv_down_flux)[1:] / wx
        v_k[:, :, 0] -= -(uv_flux_vert - uv_up_flux)[:-1] / wx

        v_k[:, :, -1] += u_contra_down[1:] * (u_cov_vert_avg - u_cov_down)[1:] / wx
        v_k[:, :, 0] += -u_contra_up[:-1] * (u_cov_vert_avg - u_cov_up)[:-1] / wx
        v_k[:, :, :, -1] += -u_contra_left[:, 1:] * (v_cov_horz_avg - v_cov_left)[:, 1:] / wx
        v_k[:, :, :, 0] += u_contra_right[:, :-1] * (v_cov_horz_avg - v_cov_right)[:, :-1] / wx

        if self.tangent_diss:
            diss = -0.5 * c_adv_vert * (self.h_up * v_cov_up - self.h_down * v_cov_down) / h_ve
            v_k[:, :, -1] -= diss[1:] * self.J_eta[:, :, -1] / wx
            v_k[:, :, 0] += diss[:-1] * self.J_eta[:, :, 0] / wx

            diss = 0.5 * c_adv_vert * (self.h_up * vel_up - self.h_down * vel_down) / h_ve
            v_k[:, :, -1] -= diss[1:] / wx
            v_k[:, :, 0] += diss[:-1] / wx

            diss = -0.5 * c_adv_horz * (self.h_right * v_cov_right - self.h_left * v_cov_left) / h_ho
            v_k[:, :, :, -1] -= diss[:, 1:] * self.J_xi[:, :, :, -1] / wx
            v_k[:, :, :, 0] += diss[:, :-1] * self.J_xi[:, :, :, 0] / wx

        u_k, v_k, w_k = self.cov_to_phys(u_k, v_k, 0)

        return u_k, v_k, w_k, h_k

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


__all__ = ["DGCubedSphereSWENumpy", "DGCubedSphereFaceNumpy"]
