import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np

import dg_swe.dg_cubed_sphere_swe as swe
from dg_swe.dg_cubed_sphere_swe import DGCubedSphereSWE


FACE_NAMES = ("zp", "zn", "xp", "xn", "yp", "yn")


class _FakePersistentRequest:
    def __init__(self, rank, kind, buffer, peer, mailbox):
        self.rank = rank
        self.kind = kind
        self.buffer = buffer
        self.peer = peer
        self.mailbox = mailbox
        self.started = False

    def Start(self):
        self.started = True
        if self.kind == "send":
            key = (self.rank, self.peer)
            assert key not in self.mailbox
            self.mailbox[key] = np.array(self.buffer, copy=True)

    def Wait(self):
        assert self.started
        if self.kind == "recv":
            self.buffer[...] = self.mailbox[(self.peer, self.rank)]
        self.started = False


class _FakeComm:
    def __init__(self, rank, size, mailbox):
        self.rank = rank
        self.size = size
        self.mailbox = mailbox
        self._gather_index = 0

    def Get_rank(self):
        return self.rank

    def Get_size(self):
        return self.size

    def Send_init(self, buffer, dest, tag=None):
        assert tag is None
        return _FakePersistentRequest(self.rank, "send", buffer, dest, self.mailbox)

    def Recv_init(self, buffer, source, tag=None):
        assert tag is None
        return _FakePersistentRequest(self.rank, "recv", buffer, source, self.mailbox)

    def gather(self, value, root=0):
        key = ("gather", self._gather_index, root)
        self._gather_index += 1
        contributions = self.mailbox.setdefault(key, {})
        assert self.rank not in contributions
        contributions[self.rank] = value
        if self.rank == root:
            assert len(contributions) == self.size
            out = [contributions[rank] for rank in range(self.size)]
            del self.mailbox[key]
            return out
        return None

    def Barrier(self):
        pass


def _state(face):
    return face.u, face.v, face.w, face.h


def _external_boundaries(face):
    for attr, idx in (
        ("u_right", (slice(None), -1)),
        ("v_right", (slice(None), -1)),
        ("w_right", (slice(None), -1)),
        ("h_right", (slice(None), -1)),
        ("vort_right", (slice(None), -1)),
        ("u_left", (slice(None), 0)),
        ("v_left", (slice(None), 0)),
        ("w_left", (slice(None), 0)),
        ("h_left", (slice(None), 0)),
        ("vort_left", (slice(None), 0)),
        ("u_up", (-1,)),
        ("v_up", (-1,)),
        ("w_up", (-1,)),
        ("h_up", (-1,)),
        ("vort_up", (-1,)),
        ("u_down", (0,)),
        ("v_down", (0,)),
        ("w_down", (0,)),
        ("h_down", (0,)),
        ("vort_down", (0,)),
    ):
        yield attr, idx, getattr(face, attr)[idx]


def test_numpy_mpi_boundary_buffers_match_serial_exchange():
    solver_kwargs = dict(
        poly_order=1,
        nx=4,
        ny=4,
        g=9.81,
        f=1.0e-4,
        eps=0.1,
        radius=1.0,
        dtype=np.float64,
    )
    serial = DGCubedSphereSWE(**solver_kwargs)

    mailbox = {}
    parallel = [
        DGCubedSphereSWE(
            **solver_kwargs,
            nprocx=1,
            nprocy=1,
            comm=_FakeComm(rank, len(FACE_NAMES), mailbox),
        )
        for rank in range(len(FACE_NAMES))
    ]

    rng = np.random.default_rng(20260731)
    for face_idx, name in enumerate(FACE_NAMES):
        shape = serial.faces[name].J.shape
        u = face_idx + rng.standard_normal(shape)
        v = face_idx + 10.0 + rng.standard_normal(shape)
        w = face_idx + 20.0 + rng.standard_normal(shape)
        h = 100.0 + face_idx + rng.random(shape)
        serial.faces[name].set_initial_condition(u, v, w, h)
        parallel[face_idx].faces[name].set_initial_condition(u, v, w, h)

    serial_state = {name: _state(serial.faces[name]) for name in FACE_NAMES}
    serial.boundaries(serial_state)

    for solver in parallel:
        state = {solver.face_name: _state(solver.faces[solver.face_name])}
        solver.set_vort(state)

    pending = []
    for solver in parallel:
        state = {solver.face_name: _state(solver.faces[solver.face_name])}
        pending.append((solver, solver.fill_boundaries(state)))

    for solver, reqs in pending:
        solver.recv_boundaries(reqs)

    for face_idx, name in enumerate(FACE_NAMES):
        serial_face = serial.faces[name]
        parallel_face = parallel[face_idx].faces[name]
        for attr, idx, reference in _external_boundaries(serial_face):
            np.testing.assert_array_equal(getattr(parallel_face, attr)[idx], reference)


def test_numpy_mpi_face_setup_receives_local_subtile_sizes(monkeypatch):
    calls = []
    original_face_class = swe.DGCubedSphereFace

    class RecordingFace(original_face_class):
        def __init__(self, name, poly_order, nx, ny, *args, **kwargs):
            calls.append((name, nx, ny, kwargs.copy()))
            super().__init__(name, poly_order, nx, ny, *args, **kwargs)

    monkeypatch.setattr(swe, "DGCubedSphereFace", RecordingFace)

    DGCubedSphereSWE(
        poly_order=1,
        nx=9,
        ny=9,
        g=9.81,
        f=1.0e-4,
        eps=0.1,
        radius=1.0,
        dtype=np.float64,
        nprocx=2,
        nprocy=2,
        comm=_FakeComm(rank=5, size=len(FACE_NAMES) * 4, mailbox={}),
    )

    assert len(calls) == 1
    name, nx, ny, kwargs = calls[0]
    assert name == "zn"
    assert nx == 5
    assert ny == 5
    assert kwargs["global_nx"] == 8
    assert kwargs["global_ny"] == 8
    assert kwargs["x_min"] == -0.5
    assert kwargs["x_max"] == 0.0
    assert kwargs["y_min"] == 0.0
    assert kwargs["y_max"] == 0.5


def test_numpy_mpi_restart_files_are_full_faces_for_subtile_ranks(tmp_path):
    solver_kwargs = dict(
        poly_order=1,
        nx=5,
        ny=5,
        g=9.81,
        f=1.0e-4,
        eps=0.1,
        radius=1.0,
        dtype=np.float64,
    )
    nprocx = nprocy = 2
    size = len(FACE_NAMES) * nprocx * nprocy
    mailbox = {}
    solvers = [
        DGCubedSphereSWE(
            **solver_kwargs,
            nprocx=nprocx,
            nprocy=nprocy,
            comm=_FakeComm(rank, size, mailbox),
        )
        for rank in range(size)
    ]

    vars = ("u", "v", "w", "h")
    for solver in solvers:
        face = solver.faces[solver.face_name]
        rank_pattern = np.arange(np.prod(face.J.shape), dtype=face.dtype).reshape(face.J.shape)
        base = (
            1000 * FACE_NAMES.index(solver.face_name)
            + 100 * solver.x_proc_idx
            + 10 * solver.y_proc_idx
        )
        state = tuple(base + var_idx + 0.01 * rank_pattern for var_idx in range(len(vars)))
        face.set_initial_condition(*state)

    template = "subtile_restart.npy"
    for solver in solvers[1:]:
        solver.save_restart(template, tmp_path)
    solvers[0].save_restart(template, tmp_path)

    expected_files = sorted(
        f"{var}_{name}_{template}"
        for name in FACE_NAMES
        for var in vars
    )
    assert sorted(path.name for path in tmp_path.iterdir()) == expected_files

    for name in FACE_NAMES:
        for var in vars:
            data = np.load(tmp_path / f"{var}_{name}_{template}")
            assert data.shape == (4, 4, 2, 2)
            for solver in solvers:
                if solver.face_name != name:
                    continue
                face = solver.faces[name]
                np.testing.assert_array_equal(
                    data[solver._restart_tile_slice(face)],
                    getattr(face, var),
                )

    reloaded = [
        DGCubedSphereSWE(
            **solver_kwargs,
            nprocx=nprocx,
            nprocy=nprocy,
            comm=_FakeComm(rank, size, {}),
        )
        for rank in range(size)
    ]
    for solver in reloaded:
        solver.boundaries = lambda: None
        solver.load_restart(template, tmp_path)

    for expected_solver, actual_solver in zip(solvers, reloaded):
        expected_face = expected_solver.faces[expected_solver.face_name]
        actual_face = actual_solver.faces[actual_solver.face_name]
        for var in vars:
            np.testing.assert_array_equal(getattr(actual_face, var), getattr(expected_face, var))
