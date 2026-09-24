import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None


if njit is not None:
    @njit(cache=True, fastmath=True, boundscheck=False, nogil=True, error_model="numpy")
    def _solve_tswe_numba_kernel(
        u, v, w, h, hb,
        D, endpoint_weight, J,
        vert_upper_edge_factor, vert_lower_edge_factor,
        horz_right_edge_factor, horz_left_edge_factor,
        dxidx, dxidy, dxidz, detadx, detady, detadz,
        dxidx_up, dxidy_up, dxidz_up,
        dxidx_down, dxidy_down, dxidz_down,
        dxidx_right, dxidy_right, dxidz_right,
        dxidx_left, dxidy_left, dxidz_left,
        detadx_up, detady_up, detadz_up,
        detadx_down, detady_down, detadz_down,
        detadx_right, detady_right, detadz_right,
        detadx_left, detady_left, detadz_left,
        dxdxi, dydxi, dzdxi, dxdeta, dydeta, dzdeta,
        f,
        u_up, v_up, w_up, h_up, hb_up,
        u_down, v_down, w_down, h_down, hb_down,
        u_right, v_right, w_right, h_right, hb_right,
        u_left, v_left, w_left, h_left, hb_left,
        eta_x_up, eta_y_up, eta_z_up,
        eta_x_down, eta_y_down, eta_z_down,
        xi_x_right, xi_y_right, xi_z_right,
        xi_x_left, xi_y_left, xi_z_left,
        dxdxi_up, dydxi_up, dzdxi_up,
        dxdxi_down, dydxi_down, dzdxi_down,
        dxdxi_right, dydxi_right, dzdxi_right,
        dxdxi_left, dydxi_left, dzdxi_left,
        dxdeta_up, dydeta_up, dzdeta_up,
        dxdeta_down, dydeta_down, dzdeta_down,
        dxdeta_right, dydeta_right, dzdeta_right,
        dxdeta_left, dydeta_left, dzdeta_left,
        g, a, ah, tangent_diss, upwind,
    ):
        ny, nx, n, _ = u.shape

        h_xcontra = np.empty_like(u)
        h_ycontra = np.empty_like(u)
        h_xcontra_J = np.empty_like(u)
        h_ycontra_J = np.empty_like(u)
        hb_xcontra_J = np.empty_like(u)
        hb_ycontra_J = np.empty_like(u)
        b = np.empty_like(u)
        uv_flux = np.empty_like(u)
        u_contra = np.empty_like(u)
        v_contra = np.empty_like(u)
        u_cov = np.empty_like(u)
        v_cov = np.empty_like(u)

        h_k = np.empty_like(u)
        hb_k = np.empty_like(u)
        u_k_cov = np.empty_like(u)
        v_k_cov = np.empty_like(u)

        inv_endpoint_weight = 1.0 / endpoint_weight

        for ey in range(ny):
            for ex in range(nx):
                for eta in range(n):
                    for xi in range(n):
                        uu = u[ey, ex, eta, xi]
                        vv = v[ey, ex, eta, xi]
                        ww = w[ey, ex, eta, xi]
                        hh = h[ey, ex, eta, xi]
                        hhb = hb[ey, ex, eta, xi]
                        j_val = J[ey, ex, eta, xi]

                        uc = (
                            uu * dxidx[ey, ex, eta, xi]
                            + vv * dxidy[ey, ex, eta, xi]
                            + ww * dxidz[ey, ex, eta, xi]
                        )
                        vc = (
                            uu * detadx[ey, ex, eta, xi]
                            + vv * detady[ey, ex, eta, xi]
                            + ww * detadz[ey, ex, eta, xi]
                        )
                        u_contra[ey, ex, eta, xi] = uc
                        v_contra[ey, ex, eta, xi] = vc

                        hx = hh * uc
                        hy = hh * vc
                        h_xcontra[ey, ex, eta, xi] = hx
                        h_ycontra[ey, ex, eta, xi] = hy
                        h_xcontra_J[ey, ex, eta, xi] = hx * j_val
                        h_ycontra_J[ey, ex, eta, xi] = hy * j_val
                        hb_xcontra_J[ey, ex, eta, xi] = hhb * uc * j_val
                        hb_ycontra_J[ey, ex, eta, xi] = hhb * vc * j_val

                        b[ey, ex, eta, xi] = hhb / hh
                        uv_flux[ey, ex, eta, xi] = (
                            0.5 * (uu * uu + vv * vv + ww * ww) + 0.5 * hhb
                        )
                        u_cov[ey, ex, eta, xi] = (
                            uu * dxdxi[ey, ex, eta, xi]
                            + vv * dydxi[ey, ex, eta, xi]
                            + ww * dzdxi[ey, ex, eta, xi]
                        )
                        v_cov[ey, ex, eta, xi] = (
                            uu * dxdeta[ey, ex, eta, xi]
                            + vv * dydeta[ey, ex, eta, xi]
                            + ww * dzdeta[ey, ex, eta, xi]
                        )

        for ey in range(ny):
            for ex in range(nx):
                for eta in range(n):
                    for xi in range(n):
                        ddxi_h = 0.0
                        ddeta_h = 0.0
                        ddxi_hb = 0.0
                        ddeta_hb = 0.0
                        ddxi_b = 0.0
                        ddeta_b = 0.0
                        ddxi_plain_h = 0.0
                        ddeta_plain_h = 0.0
                        ddxi_plain_hb = 0.0
                        ddeta_plain_hb = 0.0
                        ddxi_uv = 0.0
                        ddeta_uv = 0.0
                        ddxi_vcov = 0.0
                        ddeta_ucov = 0.0
                        for l in range(n):
                            d_xi = D[l, xi]
                            d_eta = D[l, eta]
                            ddxi_h += h_xcontra_J[ey, ex, eta, l] * d_xi
                            ddeta_h += d_eta * h_ycontra_J[ey, ex, l, xi]
                            ddxi_hb += hb_xcontra_J[ey, ex, eta, l] * d_xi
                            ddeta_hb += d_eta * hb_ycontra_J[ey, ex, l, xi]
                            ddxi_b += b[ey, ex, eta, l] * d_xi
                            ddeta_b += d_eta * b[ey, ex, l, xi]
                            ddxi_plain_h += h[ey, ex, eta, l] * d_xi
                            ddeta_plain_h += d_eta * h[ey, ex, l, xi]
                            ddxi_plain_hb += hb[ey, ex, eta, l] * d_xi
                            ddeta_plain_hb += d_eta * hb[ey, ex, l, xi]
                            ddxi_uv += uv_flux[ey, ex, eta, l] * d_xi
                            ddeta_uv += d_eta * uv_flux[ey, ex, l, xi]
                            ddxi_vcov += v_cov[ey, ex, eta, l] * d_xi
                            ddeta_ucov += d_eta * u_cov[ey, ex, l, xi]

                        j_val = J[ey, ex, eta, xi]
                        div = (ddxi_h + ddeta_h) / j_val
                        bdiv = (ddxi_hb + ddeta_hb) / j_val
                        bb = b[ey, ex, eta, xi]
                        hh = h[ey, ex, eta, xi]

                        h_k[ey, ex, eta, xi] = -div
                        hb_k[ey, ex, eta, xi] = -0.5 * (
                            bdiv
                            + bb * div
                            + ddxi_b * h_xcontra[ey, ex, eta, xi]
                            + ddeta_b * h_ycontra[ey, ex, eta, xi]
                        )

                        abs_vort_cov = (
                            ddxi_vcov - ddeta_ucov + f[ey, ex, eta, xi] * j_val
                        )
                        u_k_cov[ey, ex, eta, xi] = (
                            -ddxi_uv
                            + v_contra[ey, ex, eta, xi] * abs_vort_cov
                            - 0.25 * (
                                bb * ddxi_plain_h + ddxi_plain_hb - hh * ddxi_b
                            )
                        )
                        v_k_cov[ey, ex, eta, xi] = (
                            -ddeta_uv
                            - u_contra[ey, ex, eta, xi] * abs_vort_cov
                            - 0.25 * (
                                bb * ddeta_plain_h + ddeta_plain_hb - hh * ddeta_b
                            )
                        )

        for ey in range(ny + 1):
            for ex in range(nx):
                for xi in range(n):
                    uu_up = u_up[ey, ex, xi]
                    vv_up = v_up[ey, ex, xi]
                    ww_up = w_up[ey, ex, xi]
                    hh_up = h_up[ey, ex, xi]
                    hhb_up = hb_up[ey, ex, xi]
                    uu_down = u_down[ey, ex, xi]
                    vv_down = v_down[ey, ex, xi]
                    ww_down = w_down[ey, ex, xi]
                    hh_down = h_down[ey, ex, xi]
                    hhb_down = hb_down[ey, ex, xi]

                    h_up_flux = hh_up * (
                        uu_up * eta_x_up[ey, ex, xi]
                        + vv_up * eta_y_up[ey, ex, xi]
                        + ww_up * eta_z_up[ey, ex, xi]
                    )
                    h_down_flux = hh_down * (
                        uu_down * eta_x_down[ey, ex, xi]
                        + vv_down * eta_y_down[ey, ex, xi]
                        + ww_down * eta_z_down[ey, ex, xi]
                    )
                    hb_up_flux = hhb_up * (
                        uu_up * eta_x_up[ey, ex, xi]
                        + vv_up * eta_y_up[ey, ex, xi]
                        + ww_up * eta_z_up[ey, ex, xi]
                    )
                    hb_down_flux = hhb_down * (
                        uu_down * eta_x_down[ey, ex, xi]
                        + vv_down * eta_y_down[ey, ex, xi]
                        + ww_down * eta_z_down[ey, ex, xi]
                    )

                    vel_up = h_up_flux / hh_up
                    vel_down = h_down_flux / hh_down
                    h_ve = 0.5 * (hh_up + hh_down)
                    c_adv = 0.5 * (hh_up * vel_up + hh_down * vel_down) / h_ve
                    c_adv_tangent = 0.5 * (vel_up + vel_down)
                    c_snd = 0.5 * (np.sqrt(hhb_up) + np.sqrt(hhb_down))

                    h_flux = c_adv * h_ve - ah * abs(c_adv) * (hh_up - hh_down)
                    b_up_val = hhb_up / hh_up
                    b_down_val = hhb_down / hh_down
                    b_hat = 0.5 * (b_up_val + b_down_val)
                    if upwind:
                        if h_flux >= 0.0:
                            b_hat = b_down_val
                        else:
                            b_hat = b_up_val
                    hb_flux = b_hat * h_flux

                    uv_up_flux = (
                        0.5 * (uu_up * uu_up + vv_up * vv_up + ww_up * ww_up)
                        + 0.5 * hhb_up
                    )
                    uv_down_flux = (
                        0.5 * (uu_down * uu_down + vv_down * vv_down + ww_down * ww_down)
                        + 0.5 * hhb_down
                    )
                    uv_flux_edge = 0.5 * (uv_up_flux + uv_down_flux) - a * (
                        c_snd + abs(c_adv)
                    ) * (h_up_flux - h_down_flux) / h_ve

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

                    avg_tan_cov = 0.5 * (u_cov_up_val + u_cov_down_val)
                    if tangent_diss:
                        if c_adv_tangent < 0.0:
                            avg_tan_cov = u_cov_up_val
                        else:
                            avg_tan_cov = u_cov_down_val

                    u_flux_up = v_contra_up_val * avg_tan_cov
                    u_flux_down = v_contra_down_val * avg_tan_cov
                    v_flux_up = uv_flux_edge - u_contra_up_val * avg_tan_cov
                    v_flux_down = uv_flux_edge - u_contra_down_val * avg_tan_cov

                    if ey > 0:
                        cell_y = ey - 1
                        h_k[cell_y, ex, n - 1, xi] -= (
                            h_flux - h_down_flux
                        ) * vert_upper_edge_factor[cell_y, ex, xi]
                        hb_k[cell_y, ex, n - 1, xi] -= (
                            hb_flux - hb_down_flux
                        ) * vert_upper_edge_factor[cell_y, ex, xi]
                        u_k_cov[cell_y, ex, n - 1, xi] -= (
                            u_flux_down - v_contra_down_val * u_cov_down_val
                        ) * inv_endpoint_weight
                        v_k_cov[cell_y, ex, n - 1, xi] -= (
                            v_flux_down - (uv_down_flux - u_contra_down_val * u_cov_down_val)
                        ) * inv_endpoint_weight
                        v_k_cov[cell_y, ex, n - 1, xi] -= (
                            0.25 * b_hat * (hh_up - hh_down)
                        ) * inv_endpoint_weight

                    if ey < ny:
                        h_k[ey, ex, 0, xi] += (
                            h_flux - h_up_flux
                        ) * vert_lower_edge_factor[ey, ex, xi]
                        hb_k[ey, ex, 0, xi] += (
                            hb_flux - hb_up_flux
                        ) * vert_lower_edge_factor[ey, ex, xi]
                        u_k_cov[ey, ex, 0, xi] += (
                            u_flux_up - v_contra_up_val * u_cov_up_val
                        ) * inv_endpoint_weight
                        v_k_cov[ey, ex, 0, xi] += (
                            v_flux_up - (uv_up_flux - u_contra_up_val * u_cov_up_val)
                        ) * inv_endpoint_weight
                        v_k_cov[ey, ex, 0, xi] += (
                            0.25 * b_hat * (hh_down - hh_up)
                        ) * inv_endpoint_weight

        for ey in range(ny):
            for ex in range(nx + 1):
                for eta in range(n):
                    uu_right = u_right[ey, ex, eta]
                    vv_right = v_right[ey, ex, eta]
                    ww_right = w_right[ey, ex, eta]
                    hh_right = h_right[ey, ex, eta]
                    hhb_right = hb_right[ey, ex, eta]
                    uu_left = u_left[ey, ex, eta]
                    vv_left = v_left[ey, ex, eta]
                    ww_left = w_left[ey, ex, eta]
                    hh_left = h_left[ey, ex, eta]
                    hhb_left = hb_left[ey, ex, eta]

                    h_right_flux = hh_right * (
                        uu_right * xi_x_right[ey, ex, eta]
                        + vv_right * xi_y_right[ey, ex, eta]
                        + ww_right * xi_z_right[ey, ex, eta]
                    )
                    h_left_flux = hh_left * (
                        uu_left * xi_x_left[ey, ex, eta]
                        + vv_left * xi_y_left[ey, ex, eta]
                        + ww_left * xi_z_left[ey, ex, eta]
                    )
                    hb_right_flux = hhb_right * (
                        uu_right * xi_x_right[ey, ex, eta]
                        + vv_right * xi_y_right[ey, ex, eta]
                        + ww_right * xi_z_right[ey, ex, eta]
                    )
                    hb_left_flux = hhb_left * (
                        uu_left * xi_x_left[ey, ex, eta]
                        + vv_left * xi_y_left[ey, ex, eta]
                        + ww_left * xi_z_left[ey, ex, eta]
                    )

                    vel_right = h_right_flux / hh_right
                    vel_left = h_left_flux / hh_left
                    h_ho = 0.5 * (hh_right + hh_left)
                    c_adv = 0.5 * (hh_right * vel_right + hh_left * vel_left) / h_ho
                    c_adv_tangent = 0.5 * (vel_right + vel_left)
                    c_snd = 0.5 * (np.sqrt(hhb_right) + np.sqrt(hhb_left))

                    h_flux = c_adv * h_ho - ah * abs(c_adv) * (hh_right - hh_left)
                    b_right_val = hhb_right / hh_right
                    b_left_val = hhb_left / hh_left
                    b_hat = 0.5 * (b_right_val + b_left_val)
                    if upwind:
                        if h_flux >= 0.0:
                            b_hat = b_left_val
                        else:
                            b_hat = b_right_val
                    hb_flux = b_hat * h_flux

                    uv_right_flux = (
                        0.5 * (uu_right * uu_right + vv_right * vv_right + ww_right * ww_right)
                        + 0.5 * hhb_right
                    )
                    uv_left_flux = (
                        0.5 * (uu_left * uu_left + vv_left * vv_left + ww_left * ww_left)
                        + 0.5 * hhb_left
                    )
                    uv_flux_edge = 0.5 * (uv_right_flux + uv_left_flux) - a * (
                        c_snd + abs(c_adv)
                    ) * (h_right_flux - h_left_flux) / h_ho

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

                    avg_tan_cov = 0.5 * (v_cov_right_val + v_cov_left_val)
                    if tangent_diss:
                        if c_adv_tangent < 0.0:
                            avg_tan_cov = v_cov_right_val
                        else:
                            avg_tan_cov = v_cov_left_val

                    u_flux_right = uv_flux_edge - v_contra_right_val * avg_tan_cov
                    u_flux_left = uv_flux_edge - v_contra_left_val * avg_tan_cov
                    v_flux_right = u_contra_right_val * avg_tan_cov
                    v_flux_left = u_contra_left_val * avg_tan_cov

                    if ex > 0:
                        cell_x = ex - 1
                        h_k[ey, cell_x, eta, n - 1] -= (
                            h_flux - h_left_flux
                        ) * horz_right_edge_factor[ey, cell_x, eta]
                        hb_k[ey, cell_x, eta, n - 1] -= (
                            hb_flux - hb_left_flux
                        ) * horz_right_edge_factor[ey, cell_x, eta]
                        u_k_cov[ey, cell_x, eta, n - 1] -= (
                            u_flux_left - (uv_left_flux - v_contra_left_val * v_cov_left_val)
                        ) * inv_endpoint_weight
                        v_k_cov[ey, cell_x, eta, n - 1] -= (
                            v_flux_left - u_contra_left_val * v_cov_left_val
                        ) * inv_endpoint_weight
                        u_k_cov[ey, cell_x, eta, n - 1] -= (
                            0.25 * b_hat * (hh_right - hh_left)
                        ) * inv_endpoint_weight

                    if ex < nx:
                        h_k[ey, ex, eta, 0] += (
                            h_flux - h_right_flux
                        ) * horz_left_edge_factor[ey, ex, eta]
                        hb_k[ey, ex, eta, 0] += (
                            hb_flux - hb_right_flux
                        ) * horz_left_edge_factor[ey, ex, eta]
                        u_k_cov[ey, ex, eta, 0] += (
                            u_flux_right - (uv_right_flux - v_contra_right_val * v_cov_right_val)
                        ) * inv_endpoint_weight
                        v_k_cov[ey, ex, eta, 0] += (
                            v_flux_right - u_contra_right_val * v_cov_right_val
                        ) * inv_endpoint_weight
                        u_k_cov[ey, ex, eta, 0] += (
                            0.25 * b_hat * (hh_left - hh_right)
                        ) * inv_endpoint_weight

        u_k = np.empty_like(u)
        v_k = np.empty_like(u)
        w_k = np.empty_like(u)
        for ey in range(ny):
            for ex in range(nx):
                for eta in range(n):
                    for xi in range(n):
                        ukc = u_k_cov[ey, ex, eta, xi]
                        vkc = v_k_cov[ey, ex, eta, xi]
                        u_k[ey, ex, eta, xi] = (
                            ukc * dxidx[ey, ex, eta, xi]
                            + vkc * detadx[ey, ex, eta, xi]
                        )
                        v_k[ey, ex, eta, xi] = (
                            ukc * dxidy[ey, ex, eta, xi]
                            + vkc * detady[ey, ex, eta, xi]
                        )
                        w_k[ey, ex, eta, xi] = (
                            ukc * dxidz[ey, ex, eta, xi]
                            + vkc * detadz[ey, ex, eta, xi]
                        )

        return u_k, v_k, w_k, h_k, hb_k

else:
    _solve_tswe_numba_kernel = None


__all__ = ["_solve_tswe_numba_kernel"]
