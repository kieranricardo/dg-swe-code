import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np
import pytest

from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
from dg_swe.dg_cubed_sphere_tswe import DGCubedSphereTSWE
from dg_swe.tswe_numba_kernels import _solve_tswe_numba_kernel


FACE_NAMES = ("zp", "zn", "xp", "xn", "yp", "yn")
COMPONENT_NAMES = ("du", "dv", "dw", "dh", "dhb")


def _make_solver(
    poly_order=1,
    grid=3,
    *,
    a=0.0,
    ah=0.0,
    flux_type="standard",
    tangent_diss=None,
    upwind=False,
    eps=0.05,
):
    kwargs = {}
    if tangent_diss is not None:
        kwargs["tangent_diss"] = tangent_diss
    return DGCubedSphereTSWE(
        poly_order=poly_order,
        nx=grid,
        ny=grid,
        g=9.81,
        f=7.2921e-5,
        eps=eps,
        radius=1.0,
        a=a,
        ah=ah,
        flux_type=flux_type,
        dtype=np.float64,
        upwind=upwind,
        **kwargs,
    )


def _set_smooth_state(solver, *, seed=20261004, velocity_scale=0.01, tracer_scale=0.01):
    rng = np.random.default_rng(seed)

    for name in FACE_NAMES:
        face = solver.faces[name]
        shape = face.J.shape
        u = velocity_scale * (face.ys - 0.15 * face.zs)
        v = velocity_scale * (0.2 * face.xs + face.zs)
        w = velocity_scale * (-face.xs - 0.25 * face.ys)
        h = 1.0 + 0.015 * face.xs + 0.01 * face.ys
        b = 9.81 + tracer_scale * (
            face.xs - 0.5 * face.ys + 0.25 * face.zs
        )
        h += 0.001 * rng.standard_normal(shape)
        b += 0.001 * rng.standard_normal(shape)
        h = np.maximum(h, 0.8)
        b = np.maximum(b, 1.0)
        face.set_initial_condition(u, v, w, h, h * b)

    solver.boundaries()


def _state(solver):
    return {
        name: (
            solver.faces[name].u,
            solver.faces[name].v,
            solver.faces[name].w,
            solver.faces[name].h,
            solver.faces[name].hb,
        )
        for name in FACE_NAMES
    }


def _residual_pass(solver, state, solve_method):
    solver.boundaries(state)
    return {
        name: getattr(solver.faces[name], solve_method)(*state[name], 0.0, 0.0)
        for name in FACE_NAMES
    }


def _assert_outputs_close(reference, candidate, *, abs_tol=5.0e-11, rel_tol=5.0e-12):
    max_abs = 0.0
    max_rel = 0.0
    worst = None

    for name in FACE_NAMES:
        for component, reference_arr, candidate_arr in zip(
            COMPONENT_NAMES, reference[name], candidate[name]
        ):
            abs_err = float(np.max(np.abs(reference_arr - candidate_arr)))
            denom = max(float(np.max(np.abs(reference_arr))), 1.0)
            rel_err = abs_err / denom
            if abs_err > max_abs:
                max_abs = abs_err
                worst = (name, component, abs_err, rel_err)
            max_rel = max(max_rel, rel_err)

    assert max_abs < abs_tol and max_rel < rel_tol, (
        f"TSWE residuals differ: max_abs={max_abs:.3e}, "
        f"max_rel={max_rel:.3e}, worst={worst}"
    )


@pytest.mark.skipif(
    _solve_tswe_numba_kernel is None,
    reason="numba is required to compare TSWE NumPy and Numba backends",
)
@pytest.mark.parametrize(
    "flux_type, ah",
    [
        ("standard", 0.0),
        ("standard_tangent", 0.35),
    ],
)
def test_tswe_numpy_and_numba_residuals_match(flux_type, ah):
    solver = _make_solver(
        poly_order=1,
        grid=3,
        a=0.25,
        ah=ah,
        flux_type=flux_type,
        upwind=True,
    )
    _set_smooth_state(solver)
    state = _state(solver)

    numpy_out = _residual_pass(solver, state, "solve_numpy")
    numba_out = _residual_pass(solver, state, "solve")

    _assert_outputs_close(numpy_out, numba_out)


def test_tswe_tangent_diss_alias_selects_standard_tangent_flux():
    solver = _make_solver(tangent_diss=True)
    assert solver.flux_type == "standard_tangent"


@pytest.mark.parametrize("flux_type", ["standard", "standard_tangent"])
def test_tswe_matches_swe_for_constant_buoyancy(flux_type):
    g = 9.81
    common = dict(
        poly_order=1,
        nx=3,
        ny=3,
        g=g,
        f=7.2921e-5,
        eps=0.0,
        radius=1.0,
        a=0.25,
        ah=0.35,
        flux_type=flux_type,
        dtype=np.float64,
    )
    swe = DGCubedSphereSWE(**common)
    tswe = DGCubedSphereTSWE(**common)
    rng = np.random.default_rng(20261024)

    for name in FACE_NAMES:
        face = swe.faces[name]
        shape = face.J.shape
        u = 0.02 * rng.standard_normal(shape)
        v = 0.02 * rng.standard_normal(shape)
        w = 0.02 * rng.standard_normal(shape)
        h = 1.0 + 0.05 * rng.random(shape)
        swe.faces[name].set_initial_condition(u, v, w, h)
        tswe.faces[name].set_initial_condition(u, v, w, h, g * h)

    swe_state = {
        name: (
            swe.faces[name].u,
            swe.faces[name].v,
            swe.faces[name].w,
            swe.faces[name].h,
        )
        for name in FACE_NAMES
    }
    tswe_state = _state(tswe)

    swe_out = _residual_pass(swe, swe_state, "solve_numpy")
    tswe_out = _residual_pass(tswe, tswe_state, "solve_numpy")

    for name in FACE_NAMES:
        for swe_arr, tswe_arr in zip(swe_out[name], tswe_out[name][:4]):
            np.testing.assert_allclose(tswe_arr, swe_arr, rtol=2.0e-12, atol=2.0e-11)
        np.testing.assert_allclose(tswe_out[name][4], g * swe_out[name][3], rtol=2.0e-12, atol=2.0e-11)


def test_tswe_tracer_variance_is_stable_for_short_split_form_run():
    solver = _make_solver(poly_order=1, grid=2, a=0.0, upwind=False, eps=0.02)
    _set_smooth_state(solver, tracer_scale=0.02)
    initial = solver.integrate(solver.tracer_variance())

    for _ in range(3):
        solver.time_step(dt=0.001, order=3)

    final = solver.integrate(solver.tracer_variance())
    assert np.isfinite(final)
    assert abs(final - initial) / abs(initial) < 5.0e-5


def test_tswe_energy_is_stable_for_short_split_form_run():
    solver = _make_solver(poly_order=1, grid=2, a=0.0, upwind=False, eps=0.02)
    _set_smooth_state(solver, velocity_scale=0.005, tracer_scale=0.005)
    initial = solver.integrate(solver.energy())

    for _ in range(3):
        solver.time_step(dt=0.001, order=3)

    final = solver.integrate(solver.energy())
    assert np.isfinite(final)
    assert abs(final - initial) / abs(initial) < 5.0e-5
