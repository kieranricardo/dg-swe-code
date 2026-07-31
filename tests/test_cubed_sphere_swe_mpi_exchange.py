import os

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp")

import numpy as np

from dg_swe.dg_cubed_sphere_swe_numpy import DGCubedSphereSWENumpy


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
    serial = DGCubedSphereSWENumpy(**solver_kwargs)

    mailbox = {}
    parallel = [
        DGCubedSphereSWENumpy(
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
