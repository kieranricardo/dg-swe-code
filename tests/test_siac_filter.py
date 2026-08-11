import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np

from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
from dg_swe.siac import DGCubedSphereSWESIAC, siac_kernel


def _make_solver(poly_order=2, grid=17):
    return DGCubedSphereSWESIAC(
        poly_order=poly_order,
        nx=grid,
        ny=grid,
        g=9.81,
        f=1.0e-4,
        eps=0.1,
        radius=1.0,
        dtype=np.float64,
    )


def test_core_solver_has_no_siac_methods():
    solver = DGCubedSphereSWE(
        poly_order=1,
        nx=3,
        ny=3,
        g=9.81,
        f=1.0e-4,
        eps=0.1,
        radius=1.0,
        dtype=np.float64,
    )

    assert not hasattr(solver, "siac_filter")
    assert not hasattr(solver.faces["zp"], "siac_filter")


def _interior_mask(face, kernel, scale=1.0):
    support = max(abs(kernel.breakpoints[0]), abs(kernel.breakpoints[-1]))
    x_margin = support * scale * face.lx
    y_margin = support * scale * face.ly
    return (
        (face.x1 > face.x_min + x_margin)
        & (face.x1 < face.x_max - x_margin)
        & (face.y1 > face.y_min + y_margin)
        & (face.y1 < face.y_max - y_margin)
    )


def test_face_siac_filter_reproduces_polynomial_on_interior_nodes():
    solver = _make_solver(poly_order=2, grid=17)
    face = solver.faces["zp"]
    kernel = siac_kernel(face.poly_order)

    poly = (
        1.5
        + 2.0 * face.x1
        - 0.75 * face.y1
        + 0.5 * face.x1 * face.y1
        - face.x1 ** 2
        + 0.25 * face.y1 ** 2
    )

    filtered = face.siac_filter(poly, kernel=kernel, quadrature_order=10)
    dpoly_dx = face.siac_filter(poly, derivative=(1, 0), kernel=kernel, quadrature_order=10)

    mask = _interior_mask(face, kernel)
    np.testing.assert_allclose(filtered[mask], poly[mask], atol=2.0e-11, rtol=2.0e-11)
    np.testing.assert_allclose(
        dpoly_dx[mask],
        (2.0 + 0.5 * face.y1 - 2.0 * face.x1)[mask],
        atol=2.0e-10,
        rtol=2.0e-10,
    )


def test_face_siac_vorticity_uses_derivative_filter_on_interior_nodes():
    solver = _make_solver(poly_order=2, grid=17)
    face = solver.faces["zp"]
    kernel = siac_kernel(face.poly_order)

    u_cov_local = np.zeros_like(face.x1)
    v_cov_local = face.x1 ** 2 + face.y1
    u_ref_cov = 0.5 * face.lx * u_cov_local
    v_ref_cov = 0.5 * face.ly * v_cov_local
    u, v, w = face.cov_to_phys(u_ref_cov, v_ref_cov, 0.0)
    h = np.ones_like(face.x1)
    face.set_initial_condition(u, v, w, h)

    vort = face.siac_vorticity(kernel=kernel, quadrature_order=10)
    expected = 2.0 * face.x1 / face.local_surface_jacobian() + face.f

    mask = _interior_mask(face, kernel)
    np.testing.assert_allclose(vort[mask], expected[mask], atol=5.0e-10, rtol=5.0e-10)


def test_solver_sphere_siac_filter_preserves_constants_across_faces():
    solver = _make_solver(poly_order=1, grid=9)

    for face in solver.faces.values():
        shape = face.J.shape
        face.set_initial_condition(
            np.zeros(shape),
            np.zeros(shape),
            np.zeros(shape),
            np.full(shape, 2.5),
        )

    filtered_h = solver.siac_filter("h", boundary="sphere", quadrature_order=6)
    vort = solver.siac_vorticity(boundary="sphere", quadrature_order=6)

    for name, face in solver.faces.items():
        np.testing.assert_allclose(filtered_h[name], 2.5, atol=3.0e-12, rtol=3.0e-12)
        np.testing.assert_allclose(vort[name], face.f, atol=3.0e-12, rtol=3.0e-12)
