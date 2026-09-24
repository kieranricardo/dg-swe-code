import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np
import pytest

from dg_swe.dg_cubed_sphere_tswe import DGCubedSphereTSWE
from dg_swe.tswe_numba_kernels import _solve_tswe_numba_kernel


FACE_NAMES = ("zp", "zn", "xp", "xn", "yp", "yn")
COMPONENT_NAMES = ("du", "dv", "dw", "dh", "dhb")


def _make_solver(poly_order=1, grid=3, *, a=0.0, upwind=False, eps=0.05):
    return DGCubedSphereTSWE(
        poly_order=poly_order,
        nx=grid,
        ny=grid,
        g=9.81,
        f=7.2921e-5,
        eps=eps,
        radius=1.0,
        a=a,
        dtype=np.float64,
        upwind=upwind,
    )


def _set_smooth_state(solver, *, seed=20261004, velocity_scale=1, tracer_scale=1):
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

        u += rng.standard_normal(shape)
        v += rng.standard_normal(shape)
        w += rng.standard_normal(shape)

        h += 0.1 * rng.standard_normal(shape)
        b += 1 * rng.standard_normal(shape)

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
def test_tswe_numpy_and_numba_residuals_match():
    solver = _make_solver(poly_order=1, grid=3, a=0.25, upwind=True)
    _set_smooth_state(solver)
    state = _state(solver)

    numpy_out = _residual_pass(solver, state, "solve_numpy")
    numba_out = _residual_pass(solver, state, "solve")

    _assert_outputs_close(numpy_out, numba_out)


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
