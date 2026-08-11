from math import comb, factorial

import numpy as np

from dg_swe.dg_cubed_sphere_swe import DGCubedSphereFace, DGCubedSphereSWE
from dg_swe.utils import lagrange_basis_values, to_numpy as _to_numpy


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


class DGCubedSphereSWESIACMixin:
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



class DGCubedSphereFaceSIACMixin:
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
        ``DGCubedSphereSWESIAC.siac_filter(..., boundary="sphere")`` when a
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


class DGCubedSphereFaceSIAC(DGCubedSphereFaceSIACMixin, DGCubedSphereFace):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._siac_matrix_cache = {}
        self._siac_extension_coord_cache = {}


class DGCubedSphereSWESIAC(DGCubedSphereSWESIACMixin, DGCubedSphereSWE):
    face_class = DGCubedSphereFaceSIAC


__all__ = [
    "DGCubedSphereSWESIAC",
    "DGCubedSphereFaceSIAC",
    "SIACKernel",
    "centered_cardinal_bspline",
    "siac_kernel",
]
