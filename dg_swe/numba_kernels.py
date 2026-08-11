import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


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
                    amp_0 = h_avg * du_t

                    diss_h = mu_m * amp_m + mu_p * amp_p
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
                        dm_u = scale * diss_u
                        dm_v = scale * diss_v
                        dm_w = scale * diss_w
                        h_k[cell_y, ex, eta_idx, xi] += dh_k
                        u_k[cell_y, ex, eta_idx, xi] += (dm_u - uu_l * dh_k) / hh_l
                        v_k[cell_y, ex, eta_idx, xi] += (dm_v - vv_l * dh_k) / hh_l
                        w_k[cell_y, ex, eta_idx, xi] += (dm_w - ww_l * dh_k) / hh_l

                    if ey < ny:
                        eta_idx = 0
                        scale = -0.5 * vert_lower_edge_factor[ey, ex, xi]
                        dh_k = scale * diss_h
                        dm_u = scale * diss_u
                        dm_v = scale * diss_v
                        dm_w = scale * diss_w
                        h_k[ey, ex, eta_idx, xi] += dh_k
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
                    amp_0 = h_avg * du_t

                    diss_h = mu_m * amp_m + mu_p * amp_p
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
                        dm_u = scale * diss_u
                        dm_v = scale * diss_v
                        dm_w = scale * diss_w
                        h_k[ey, cell_x, eta, xi_idx] += dh_k
                        u_k[ey, cell_x, eta, xi_idx] += (dm_u - uu_l * dh_k) / hh_l
                        v_k[ey, cell_x, eta, xi_idx] += (dm_v - vv_l * dh_k) / hh_l
                        w_k[ey, cell_x, eta, xi_idx] += (dm_w - ww_l * dh_k) / hh_l

                    if ex < nx:
                        xi_idx = 0
                        scale = -0.5 * horz_left_edge_factor[ey, ex, eta]
                        dh_k = scale * diss_h
                        dm_u = scale * diss_u
                        dm_v = scale * diss_v
                        dm_w = scale * diss_w
                        h_k[ey, ex, eta, xi_idx] += dh_k
                        u_k[ey, ex, eta, xi_idx] += (dm_u - uu_r * dh_k) / hh_r
                        v_k[ey, ex, eta, xi_idx] += (dm_v - vv_r * dh_k) / hh_r
                        w_k[ey, ex, eta, xi_idx] += (dm_w - ww_r * dh_k) / hh_r

    @njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
    def _apply_barth_normal_tangent_diss_numba(
        u_k, v_k, w_k, h_k, g,
        vert_upper_edge_factor, vert_lower_edge_factor,
        horz_right_edge_factor, horz_left_edge_factor,
        u_up, v_up, w_up, h_up, u_down, v_down, w_down, h_down,
        u_right, v_right, w_right, h_right, u_left, v_left, w_left, h_left,
        eta_x_up, eta_y_up, eta_z_up, eta_x_down, eta_y_down, eta_z_down,
        xi_x_right, xi_y_right, xi_z_right, xi_x_left, xi_y_left, xi_z_left,
    ):
        ny, nx, n, _ = u_k.shape

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

                    nx_face = 0.5 * (eta_x_down[ey, ex, xi] + eta_x_up[ey, ex, xi])
                    ny_face = 0.5 * (eta_y_down[ey, ex, xi] + eta_y_up[ey, ex, xi])
                    nz_face = 0.5 * (eta_z_down[ey, ex, xi] + eta_z_up[ey, ex, xi])
                    n_norm = np.sqrt(nx_face * nx_face + ny_face * ny_face + nz_face * nz_face)
                    nx_face = nx_face / n_norm
                    ny_face = ny_face / n_norm
                    nz_face = nz_face / n_norm

                    h_avg = 0.5 * (hh_down + hh_up)
                    c = np.sqrt(g * h_avg)
                    u_n = 0.5 * (
                        (uu_down + uu_up) * nx_face
                        + (vv_down + vv_up) * ny_face
                        + (ww_down + ww_up) * nz_face
                    )
                    du_n = (
                        (uu_up - uu_down) * nx_face
                        + (vv_up - vv_down) * ny_face
                        + (ww_up - ww_down) * nz_face
                    )
                    dh = hh_up - hh_down
                    mu_m = abs(u_n - c)
                    mu_p = abs(u_n + c)
                    amp_m = (g * dh - c * du_n) / (2.0 * g)
                    amp_p = (g * dh + c * du_n) / (2.0 * g)
                    diss_h = mu_m * amp_m + mu_p * amp_p
                    diss_un = mu_m * amp_m * (u_n - c) + mu_p * amp_p * (u_n + c)

                    if ey > 0:
                        cell_y = ey - 1
                        eta_idx = n - 1
                        scale = 0.5 * vert_upper_edge_factor[cell_y, ex, xi]
                        dh_k = scale * diss_h
                        un_l = uu_down * nx_face + vv_down * ny_face + ww_down * nz_face
                        dun = (scale * diss_un - un_l * dh_k) / hh_down
                        h_k[cell_y, ex, eta_idx, xi] += dh_k
                        u_k[cell_y, ex, eta_idx, xi] += dun * nx_face
                        v_k[cell_y, ex, eta_idx, xi] += dun * ny_face
                        w_k[cell_y, ex, eta_idx, xi] += dun * nz_face

                    if ey < ny:
                        eta_idx = 0
                        scale = -0.5 * vert_lower_edge_factor[ey, ex, xi]
                        dh_k = scale * diss_h
                        un_r = uu_up * nx_face + vv_up * ny_face + ww_up * nz_face
                        dun = (scale * diss_un - un_r * dh_k) / hh_up
                        h_k[ey, ex, eta_idx, xi] += dh_k
                        u_k[ey, ex, eta_idx, xi] += dun * nx_face
                        v_k[ey, ex, eta_idx, xi] += dun * ny_face
                        w_k[ey, ex, eta_idx, xi] += dun * nz_face

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

                    nx_face = 0.5 * (xi_x_left[ey, ex, eta] + xi_x_right[ey, ex, eta])
                    ny_face = 0.5 * (xi_y_left[ey, ex, eta] + xi_y_right[ey, ex, eta])
                    nz_face = 0.5 * (xi_z_left[ey, ex, eta] + xi_z_right[ey, ex, eta])
                    n_norm = np.sqrt(nx_face * nx_face + ny_face * ny_face + nz_face * nz_face)
                    nx_face = nx_face / n_norm
                    ny_face = ny_face / n_norm
                    nz_face = nz_face / n_norm

                    h_avg = 0.5 * (hh_left + hh_right)
                    c = np.sqrt(g * h_avg)
                    u_n = 0.5 * (
                        (uu_left + uu_right) * nx_face
                        + (vv_left + vv_right) * ny_face
                        + (ww_left + ww_right) * nz_face
                    )
                    du_n = (
                        (uu_right - uu_left) * nx_face
                        + (vv_right - vv_left) * ny_face
                        + (ww_right - ww_left) * nz_face
                    )
                    dh = hh_right - hh_left
                    mu_m = abs(u_n - c)
                    mu_p = abs(u_n + c)
                    amp_m = (g * dh - c * du_n) / (2.0 * g)
                    amp_p = (g * dh + c * du_n) / (2.0 * g)
                    diss_h = mu_m * amp_m + mu_p * amp_p
                    diss_un = mu_m * amp_m * (u_n - c) + mu_p * amp_p * (u_n + c)

                    if ex > 0:
                        cell_x = ex - 1
                        xi_idx = n - 1
                        scale = 0.5 * horz_right_edge_factor[ey, cell_x, eta]
                        dh_k = scale * diss_h
                        un_l = uu_left * nx_face + vv_left * ny_face + ww_left * nz_face
                        dun = (scale * diss_un - un_l * dh_k) / hh_left
                        h_k[ey, cell_x, eta, xi_idx] += dh_k
                        u_k[ey, cell_x, eta, xi_idx] += dun * nx_face
                        v_k[ey, cell_x, eta, xi_idx] += dun * ny_face
                        w_k[ey, cell_x, eta, xi_idx] += dun * nz_face

                    if ex < nx:
                        xi_idx = 0
                        scale = -0.5 * horz_left_edge_factor[ey, ex, eta]
                        dh_k = scale * diss_h
                        un_r = uu_right * nx_face + vv_right * ny_face + ww_right * nz_face
                        dun = (scale * diss_un - un_r * dh_k) / hh_right
                        h_k[ey, ex, eta, xi_idx] += dh_k
                        u_k[ey, ex, eta, xi_idx] += dun * nx_face
                        v_k[ey, ex, eta, xi_idx] += dun * ny_face
                        w_k[ey, ex, eta, xi_idx] += dun * nz_face

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
    _apply_barth_normal_tangent_diss_numba = None
    _solve_numba_old_tangent_kernel = None


__all__ = [
    "_solve_numba_kernel",
    "_solve_numba_lmars_kernel",
    "_apply_barth_diss_numba",
    "_apply_barth_normal_tangent_diss_numba",
    "_solve_numba_old_tangent_kernel",
]
