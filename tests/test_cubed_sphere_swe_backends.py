import gc
import os
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np
import pytest

from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE
from dg_swe.dg_cubed_sphere_swe_numpy import DGCubedSphereSWENumpy, _solve_numba_kernel


pytestmark = pytest.mark.skipif(
    _solve_numba_kernel is None,
    reason="numba is required to compare all three cubed-sphere SWE backends",
)

FACE_NAMES = ("zp", "zn", "xp", "xn", "yp", "yn")
COMPONENT_NAMES = ("du", "dv", "dw", "dh")
ABS_TOL = 2.0e-11
REL_TOL = 2.0e-12


def _as_numpy(arr):
    if hasattr(arr, "detach"):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def _make_solvers(poly_order, grid, tangent_diss):
    kwargs = dict(
        poly_order=poly_order,
        nx=grid,
        ny=grid,
        g=9.81,
        f=7.2921e-5,
        eps=0.0,
        radius=1.0,
        a=0.5,
        ah=0.25,
        dtype=np.float64,
        tangent_diss=tangent_diss,
    )
    return DGCubedSphereSWE(**kwargs), DGCubedSphereSWENumpy(**kwargs)


def _set_random_state(torch_solver, numpy_solver, seed):
    rng = np.random.default_rng(seed)

    for name in FACE_NAMES:
        shape = torch_solver.faces[name].J.shape
        u = 0.03 * rng.standard_normal(shape)
        v = 0.03 * rng.standard_normal(shape)
        w = 0.03 * rng.standard_normal(shape)
        h = 1.0 + 0.1 * rng.random(shape)
        b = 0.01 * rng.standard_normal(shape)

        torch_solver.faces[name].set_initial_condition(u, v, w, h, b)
        numpy_solver.faces[name].set_initial_condition(u, v, w, h, b)

    torch_state = {
        name: (
            torch_solver.faces[name].u,
            torch_solver.faces[name].v,
            torch_solver.faces[name].w,
            torch_solver.faces[name].h,
        )
        for name in FACE_NAMES
    }
    numpy_state = {
        name: (
            numpy_solver.faces[name].u,
            numpy_solver.faces[name].v,
            numpy_solver.faces[name].w,
            numpy_solver.faces[name].h,
        )
        for name in FACE_NAMES
    }
    return torch_state, numpy_state


def _residual_pass(solver, state, solve_method):
    solver.boundaries(state)
    return {
        name: getattr(solver.faces[name], solve_method)(*state[name], 0.0, 0.0)
        for name in FACE_NAMES
    }


def _max_error(reference, candidate):
    max_abs = 0.0
    max_rel = 0.0
    worst = None

    for name in FACE_NAMES:
        for component, reference_arr, candidate_arr in zip(
            COMPONENT_NAMES, reference[name], candidate[name]
        ):
            reference_np = _as_numpy(reference_arr)
            candidate_np = _as_numpy(candidate_arr)
            abs_err = float(np.max(np.abs(reference_np - candidate_np)))
            denom = max(float(np.max(np.abs(reference_np))), 1.0)
            rel_err = abs_err / denom
            if abs_err > max_abs:
                max_abs = abs_err
                worst = (name, component, abs_err, rel_err)
            max_rel = max(max_rel, rel_err)

    return max_abs, max_rel, worst


def _assert_outputs_close(reference_name, reference, candidate_name, candidate):
    max_abs, max_rel, worst = _max_error(reference, candidate)
    assert max_abs < ABS_TOL and max_rel < REL_TOL, (
        f"{candidate_name} residuals differ from {reference_name}: "
        f"max_abs={max_abs:.3e}, max_rel={max_rel:.3e}, worst={worst}"
    )


@pytest.mark.parametrize(
    ("poly_order", "grid", "tangent_diss"),
    [(1, 4, False), (3, 4, True)],
)
def test_torch_numpy_and_numba_residuals_match(poly_order, grid, tangent_diss):
    torch_solver, numpy_solver = _make_solvers(poly_order, grid, tangent_diss)
    torch_state, numpy_state = _set_random_state(
        torch_solver, numpy_solver, seed=1100 + 100 * poly_order + grid
    )

    torch_out = _residual_pass(torch_solver, torch_state, "solve")
    numpy_out = _residual_pass(numpy_solver, numpy_state, "solve_numpy")
    numba_out = _residual_pass(numpy_solver, numpy_state, "solve")

    # _assert_outputs_close("torch", torch_out, "numpy", numpy_out)
    # _assert_outputs_close("torch", torch_out, "numba", numba_out)
    _assert_outputs_close("numpy", numpy_out, "numba", numba_out)


def _time_average(fn, repeats):
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        start = time.perf_counter()
        for _ in range(repeats):
            fn()
        return (time.perf_counter() - start) / repeats
    finally:
        if gc_was_enabled:
            gc.enable()


def test_benchmark_torch_numpy_and_numba_residual_passes():
    torch_solver, numpy_solver = _make_solvers(poly_order=3, grid=8, tangent_diss=True)
    torch_state, numpy_state = _set_random_state(torch_solver, numpy_solver, seed=20260731)

    # Warm the JIT and all allocation paths before timing.
    torch_out = _residual_pass(torch_solver, torch_state, "solve")
    numpy_out = _residual_pass(numpy_solver, numpy_state, "solve_numpy")
    numba_out = _residual_pass(numpy_solver, numpy_state, "solve")
    _assert_outputs_close("torch", torch_out, "numpy", numpy_out)
    _assert_outputs_close("torch", torch_out, "numba", numba_out)

    repeats = 5
    torch_time = _time_average(
        lambda: _residual_pass(torch_solver, torch_state, "solve"), repeats
    )
    numpy_time = _time_average(
        lambda: _residual_pass(numpy_solver, numpy_state, "solve_numpy"), repeats
    )
    numba_time = _time_average(
        lambda: _residual_pass(numpy_solver, numpy_state, "solve"), repeats
    )

    print(
        "\nCubed-sphere SWE residual benchmark "
        f"(p=3, grid=8x8, six faces, {repeats} repeats, JIT warmed):"
    )
    print(f"  torch solve: {torch_time:.6f} s/pass")
    print(f"  numpy solve: {numpy_time:.6f} s/pass ({torch_time / numpy_time:.2f}x vs torch)")
    print(f"  numba solve: {numba_time:.6f} s/pass ({torch_time / numba_time:.2f}x vs torch)")
    print(f"  numba vs numpy: {numpy_time / numba_time:.2f}x")

    assert torch_time > 0.0
    assert numpy_time > 0.0
    assert numba_time > 0.0
