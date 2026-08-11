import os
import numpy as np

from dg_swe.geometry import EquiangularFace, SadournyFace, face_name_from_cartesian, lat_long_to_cartesian
from dg_swe.numba_kernels import (
    _apply_barth_diss_numba,
    _apply_barth_normal_tangent_diss_numba,
    _solve_numba_kernel,
    _solve_numba_lmars_kernel,
    _solve_numba_old_tangent_kernel,
)
from dg_swe.utils import (
    cross_product,
    continuous_element_projection,
    element_grid_coordinates,
    gll,
    lagrange1st,
    lagrange_basis_values,
    left_right_edge_arrays,
    norm_L2 as _norm_l2,
    to_numpy as _to_numpy,
    up_down_edge_arrays,
)

VALID_FLUX_TYPES = (
    "standard",
    "standard_tangent",
    "old_tangent",
    "lmars",
    "barth",
    "barth_normal_tangent",
)


def _validate_flux_type(flux_type):
    if flux_type not in VALID_FLUX_TYPES:
        raise ValueError(
            f"flux_type: expected one of {VALID_FLUX_TYPES}. Found {flux_type!r}."
        )
    return flux_type


def _reject_legacy_flux_kwargs(kwargs):
    legacy_keys = ("tangent_diss", "old_tangent_diss", "lmars", "barth_diss")
    found = [key for key in legacy_keys if key in kwargs]
    if found:
        keys = ", ".join(found)
        raise TypeError(f"Legacy flux switch keyword(s) removed: {keys}. Use flux_type instead.")






class DGCubedSphereSWE:
    face_class = None

    def __init__(
            self, poly_order, nx, ny, g, f, eps, radius=1.0, device='cpu',
            solution=None, a=0.0, ah=0.0, dtype=np.float64,
            flux_type="standard", nprocx=1, nprocy=1, comm=None, **kwargs):

        _reject_legacy_flux_kwargs(kwargs)
        self.face_names = ['zp', 'zn', 'xp', 'xn', 'yp', 'yn']
        flux_type = _validate_flux_type(flux_type)
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

        face_nx = nx
        face_ny = ny
        face_partition = {}
        if self.parallel:
            global_nx, local_nx, x_min, x_max = self._local_axis_partition(
                nx, self.nprocx, self.x_proc_idx, "nx"
            )
            global_ny, local_ny, y_min, y_max = self._local_axis_partition(
                ny, self.nprocy, self.y_proc_idx, "ny"
            )
            face_nx = local_nx
            face_ny = local_ny
            face_partition = {
                "global_nx": global_nx,
                "global_ny": global_ny,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
            }

        face_class = self.face_class or DGCubedSphereFace
        self.faces = {
            name: face_class(
                name, poly_order, face_nx, face_ny, g, f, radius, eps, device, a=a, ah=ah, dtype=dtype,
                bc='', flux_type=flux_type,
                x_proc_idx=self.x_proc_idx if self.parallel else 0,
                y_proc_idx=self.y_proc_idx if self.parallel else 0,
                nprocx=self.nprocx if self.parallel else 1,
                nprocy=self.nprocy if self.parallel else 1,
                **face_partition,
            )
            for name in self.active_face_names
        }
        if self.parallel:
            self._init_mpi_boundary_exchange()

        self.time = 0
        self.cdt = min(self.faces[n].cdt for n in self.active_face_names)
        self.flux_type = flux_type

        self.time_list = []
        self.energy_list = []
        self.enstrophy_list = []
        self.mass_list = []
        self.vorticity_list = []
        self.store_diagnostics = False

    @staticmethod
    def _get_comm(comm, nprocx, nprocy):
        if comm is not None:
            return comm
        mpi_world_size = max(
            int(os.environ.get(name, "1"))
            for name in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "MPI_LOCALNRANKS")
        )
        if nprocx * nprocy == 1 and mpi_world_size == 1:
            return None
        try:
            from mpi4py import MPI
            return MPI.COMM_WORLD
        except ImportError as exc:
            if nprocx * nprocy == 1:
                return None
            raise ImportError("mpi4py is required when nprocx*nprocy > 1.") from exc


    @property
    def nproc(self):
        return 6 * self.nprocx * self.nprocy

    @staticmethod
    def _local_axis_partition(num_elements, nproc, proc_idx, name):
        if num_elements <= 0:
            raise ValueError(f"{name} must be positive; got {name}={num_elements}.")
        if num_elements % nproc != 0:
            raise ValueError(
                f"{name} must be divisible by nproc; got {name}={num_elements}, nproc={nproc}."
            )

        local_elements = num_elements // nproc
        start = proc_idx * local_elements
        stop = start + local_elements
        dx = 1.0 / num_elements
        return (
            num_elements,
            local_elements,
            -0.5 + start * dx,
            -0.5 + stop * dx,
        )

    @property
    def x_proc_idx(self):
        return (self.rank - self.tile_idx * self.nprocx * self.nprocy) // self.nprocy

    @property
    def y_proc_idx(self):
        return (self.rank - self.tile_idx * self.nprocx * self.nprocy) % self.nprocy

    def _side_connection(self, side):
        face = self.faces[self.face_name]
        for name, (i1, i2) in face.connections:
            if i1 == side:
                return name, i2
        raise ValueError(f"No cubed-sphere connection for side {side}.")

    @property
    def prev_procx(self):
        if self.x_proc_idx == 0:
            return self.get_proc(self._side_connection(2), self.y_proc_idx)
        return self.rank - self.nprocy

    @property
    def next_procx(self):
        if self.x_proc_idx == self.nprocx - 1:
            return self.get_proc(self._side_connection(0), self.y_proc_idx)
        return self.rank + self.nprocy

    @property
    def prev_procy(self):
        if self.y_proc_idx == 0:
            return self.get_proc(self._side_connection(3), self.x_proc_idx)
        return self.rank - 1

    @property
    def next_procy(self):
        if self.y_proc_idx == self.nprocy - 1:
            return self.get_proc(self._side_connection(1), self.x_proc_idx)
        return self.rank + 1

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

    def boundaries(self, sol=None):

        if sol is None:
            sol = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}

        if self.parallel:
            reqs = self.fill_boundaries(sol)
            self.recv_boundaries(reqs)
            return

        for name in self.active_face_names:

            face = self.faces[name]

            for con in face.connections:
                n, (i1, i2) = con

                neighbour = self.faces[n]
                u, v, w, h = sol[n]
                self._assign_edge_state(face, i1, self._edge_state(neighbour, (u, v, w, h), i2))

    @staticmethod
    def _edge_state(face, state, side):
        u, v, w, h = state
        if side == 0:
            data = (u[:, -1, :, -1], v[:, -1, :, -1], w[:, -1, :, -1], h[:, -1, :, -1])
        elif side == 1:
            data = (u[-1, :, -1], v[-1, :, -1], w[-1, :, -1], h[-1, :, -1])
        elif side == 2:
            data = (u[:, 0, :, 0], v[:, 0, :, 0], w[:, 0, :, 0], h[:, 0, :, 0])
        elif side == 3:
            data = (u[0, :, 0], v[0, :, 0], w[0, :, 0], h[0, :, 0])
        else:
            raise ValueError(f"Unknown boundary side {side}.")
        return np.ascontiguousarray(np.stack(data))

    @staticmethod
    def _pack_edge_state(face, state, side, out):
        u, v, w, h = state
        if side == 0:
            out[0] = u[:, -1, :, -1]
            out[1] = v[:, -1, :, -1]
            out[2] = w[:, -1, :, -1]
            out[3] = h[:, -1, :, -1]
        elif side == 1:
            out[0] = u[-1, :, -1]
            out[1] = v[-1, :, -1]
            out[2] = w[-1, :, -1]
            out[3] = h[-1, :, -1]
        elif side == 2:
            out[0] = u[:, 0, :, 0]
            out[1] = v[:, 0, :, 0]
            out[2] = w[:, 0, :, 0]
            out[3] = h[:, 0, :, 0]
        elif side == 3:
            out[0] = u[0, :, 0]
            out[1] = v[0, :, 0]
            out[2] = w[0, :, 0]
            out[3] = h[0, :, 0]
        else:
            raise ValueError(f"Unknown boundary side {side}.")

    @staticmethod
    def _assign_edge_state(face, side, data):
        u, v, w, h = data
        if side == 0:
            face.u_right[:, -1] = u
            face.v_right[:, -1] = v
            face.w_right[:, -1] = w
            face.h_right[:, -1] = h
        elif side == 1:
            face.u_up[-1] = u
            face.v_up[-1] = v
            face.w_up[-1] = w
            face.h_up[-1] = h
        elif side == 2:
            face.u_left[:, 0] = u
            face.v_left[:, 0] = v
            face.w_left[:, 0] = w
            face.h_left[:, 0] = h
        elif side == 3:
            face.u_down[0] = u
            face.v_down[0] = v
            face.w_down[0] = w
            face.h_down[0] = h
        else:
            raise ValueError(f"Unknown boundary side {side}.")

    def _init_mpi_boundary_exchange(self):
        face = self.faces[self.face_name]
        nvars = 4
        dtype = face.dtype

        self.right_boundary_x = np.zeros((nvars, face.ny, face.n), dtype=dtype)
        self.left_boundary_x = np.zeros_like(self.right_boundary_x)
        self.right_boundary_x_send = np.zeros_like(self.right_boundary_x)
        self.left_boundary_x_send = np.zeros_like(self.right_boundary_x)

        self.right_boundary_y = np.zeros((nvars, face.nx, face.n), dtype=dtype)
        self.left_boundary_y = np.zeros_like(self.right_boundary_y)
        self.right_boundary_y_send = np.zeros_like(self.right_boundary_y)
        self.left_boundary_y_send = np.zeros_like(self.right_boundary_y)

        self.req_right_boundary_x_send = self.comm.Send_init(self.right_boundary_x_send, dest=self.next_procx)
        self.req_right_boundary_x_recv = self.comm.Recv_init(self.right_boundary_x, source=self.next_procx)
        self.req_left_boundary_x_send = self.comm.Send_init(self.left_boundary_x_send, dest=self.prev_procx)
        self.req_left_boundary_x_recv = self.comm.Recv_init(self.left_boundary_x, source=self.prev_procx)

        self.req_right_boundary_y_send = self.comm.Send_init(self.right_boundary_y_send, dest=self.next_procy)
        self.req_right_boundary_y_recv = self.comm.Recv_init(self.right_boundary_y, source=self.next_procy)
        self.req_left_boundary_y_send = self.comm.Send_init(self.left_boundary_y_send, dest=self.prev_procy)
        self.req_left_boundary_y_recv = self.comm.Recv_init(self.left_boundary_y, source=self.prev_procy)

    def fill_right_boundary_x(self, sol):
        face = self.faces[self.face_name]
        self._pack_edge_state(face, sol[self.face_name], 0, self.right_boundary_x_send)
        self.req_right_boundary_x_recv.Start()
        self.req_right_boundary_x_send.Start()
        return self.req_right_boundary_x_recv

    def fill_left_boundary_x(self, sol):
        face = self.faces[self.face_name]
        self._pack_edge_state(face, sol[self.face_name], 2, self.left_boundary_x_send)
        self.req_left_boundary_x_recv.Start()
        self.req_left_boundary_x_send.Start()
        return self.req_left_boundary_x_recv

    def fill_right_boundary_y(self, sol):
        face = self.faces[self.face_name]
        self._pack_edge_state(face, sol[self.face_name], 1, self.right_boundary_y_send)
        self.req_right_boundary_y_recv.Start()
        self.req_right_boundary_y_send.Start()
        return self.req_right_boundary_y_recv

    def fill_left_boundary_y(self, sol):
        face = self.faces[self.face_name]
        self._pack_edge_state(face, sol[self.face_name], 3, self.left_boundary_y_send)
        self.req_left_boundary_y_recv.Start()
        self.req_left_boundary_y_send.Start()
        return self.req_left_boundary_y_recv

    def fill_boundaries(self, sol):
        reqs = [
            self.fill_left_boundary_x(sol),
            self.fill_right_boundary_x(sol),
            self.fill_left_boundary_y(sol),
            self.fill_right_boundary_y(sol),
            self.req_right_boundary_x_send,
            self.req_left_boundary_x_send,
            self.req_right_boundary_y_send,
            self.req_left_boundary_y_send,
        ]
        return reqs

    def recv_boundaries(self, reqs):
        for req in reqs:
            if req is not None:
                req.Wait()

        face = self.faces[self.face_name]
        self._assign_edge_state(face, 2, self.left_boundary_x)
        self._assign_edge_state(face, 0, self.right_boundary_x)
        self._assign_edge_state(face, 3, self.left_boundary_y)
        self._assign_edge_state(face, 1, self.right_boundary_y)

    def get_dt(self):
        return min(face.get_dt() for face in self.faces.values())

    def positivity_preserving_limiter(self, state, prev_state):
        for n in self.active_face_names:
            # if state[n][3].min() < 0:
                # print('0 detected')

            cell_means = (state[n][3] * self.faces[n].Jw).sum(axis=(2, 3)) / self.faces[n].Jw.sum(axis=(2, 3))
            cell_diffs = state[n][3] - cell_means[..., None, None]

            cell_mins = state[n][3].min(axis=(2, 3))
            diff_min = cell_mins - cell_means

            target_min = np.minimum(10.0, cell_means)
            needs_limiting = cell_mins <= target_min
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
                print("Negative height detected at time:", self.time)
                speed = np.sqrt(sum(state[n][i]**2 for i in range(3)))
                print("Maximum velocity:", speed.max())
                self.comm.Abort(1)

        return state


    def time_step(self, dt=None, order=3, forcing=None):

        if self.store_diagnostics:
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
            u_1 = self.positivity_preserving_limiter(u_1, prev_state=u)
            self.boundaries(u_1)
            k_2 = {n: self.faces[n].solve(*u_1[n], self.time, dt) for n in self.active_face_names}

            u_2 = {n: tuple(0.75 * u[n][i] + 0.25 * (u_1[n][i] + dt * k_2[n][i]) for i in range(4)) for n in self.active_face_names}
            u_2 = self.positivity_preserving_limiter(u_2, prev_state=u_1)
            self.boundaries(u_2)
            k_3 = {n: self.faces[n].solve(*u_2[n], self.time, dt) for n in self.active_face_names}

            for n in self.active_face_names:
                self.faces[n].u = (self.faces[n].u + 2 * (u_2[n][0] + dt * k_3[n][0])) / 3
                self.faces[n].v = (self.faces[n].v + 2 * (u_2[n][1] + dt * k_3[n][1])) / 3
                self.faces[n].w = (self.faces[n].w + 2 * (u_2[n][2] + dt * k_3[n][2])) / 3
                self.faces[n].h = (self.faces[n].h + 2 * (u_2[n][3] + dt * k_3[n][3])) / 3

            u = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}
            u = self.positivity_preserving_limiter(u, prev_state=u_2)
            self.boundaries(u)

        elif order == 34:
            u = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}
            self.boundaries(u)
            k_1 = {n: self.faces[n].solve(*u[n], self.time, dt) for n in self.active_face_names}

            u_1 = {n: tuple(u[n][i] + 0.5 * dt * k_1[n][i] for i in range(4)) for n in self.active_face_names}
            u_1 = self.positivity_preserving_limiter(u_1, prev_state=u)
            self.boundaries(u_1)
            k_2 = {n: self.faces[n].solve(*u_1[n], self.time, dt) for n in self.active_face_names}

            u_2 = {n: tuple(u_1[n][i] + 0.5 * dt * k_2[n][i] for i in range(4)) for n in self.active_face_names}
            u_2 = self.positivity_preserving_limiter(u_2, prev_state=u_1)
            self.boundaries(u_2)
            k_3 = {n: self.faces[n].solve(*u_2[n], self.time, dt) for n in self.active_face_names}

            u_3 = {n: tuple((2 / 3) * u[n][i] + (1 / 3) * u_2[n][i] + (1 / 6) * dt * k_3[n][i] for i in range(4)) for n in self.active_face_names}
            u_3 = self.positivity_preserving_limiter(u_3, prev_state=u_2)
            self.boundaries(u_3)
            k_4 = {n: self.faces[n].solve(*u_3[n], self.time, dt) for n in self.active_face_names}

            for n in self.active_face_names:
                self.faces[n].u = u_3[n][0] + 0.5 * dt * k_4[n][0]
                self.faces[n].v = u_3[n][1] + 0.5 * dt * k_4[n][1]
                self.faces[n].w = u_3[n][2] + 0.5 * dt * k_4[n][2]
                self.faces[n].h = u_3[n][3] + 0.5 * dt * k_4[n][3]

            u = {n: (self.faces[n].u, self.faces[n].v, self.faces[n].w, self.faces[n].h) for n in self.active_face_names}
            u = self.positivity_preserving_limiter(u, prev_state=u_3)
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

    def _coeffs_by_face(self, coeffs):
        if coeffs is None:
            return {name: self.faces[name].h for name in self.active_face_names}
        if isinstance(coeffs, str):
            return {name: getattr(self.faces[name], coeffs) for name in self.active_face_names}
        if isinstance(coeffs, dict):
            return coeffs
        return {
            name: coeffs[self.face_names.index(name)]
            for name in self.active_face_names
        }

    @staticmethod
    def _side_name(side):
        names = {0: "right", 1: "up", 2: "left", 3: "down"}
        try:
            return names[side]
        except KeyError as exc:
            raise ValueError(f"Unknown boundary side {side}.") from exc

    @staticmethod
    def _edge_scalar_values(field, side):
        if side == 0:
            return np.ascontiguousarray(field[:, -1, :, -1])
        if side == 1:
            return np.ascontiguousarray(field[-1, :, -1])
        if side == 2:
            return np.ascontiguousarray(field[:, 0, :, 0])
        if side == 3:
            return np.ascontiguousarray(field[0, :, 0])
        raise ValueError(f"Unknown boundary side {side}.")

    def _current_state(self):
        return {
            name: (
                self.faces[name].u,
                self.faces[name].v,
                self.faces[name].w,
                self.faces[name].h,
            )
            for name in self.active_face_names
        }

    def _exchange_scalar_boundary_values(self, field):
        face = self.faces[self.face_name]
        recv = {
            2: np.empty((face.ny, face.n), dtype=field.dtype),
            0: np.empty((face.ny, face.n), dtype=field.dtype),
            3: np.empty((face.nx, face.n), dtype=field.dtype),
            1: np.empty((face.nx, face.n), dtype=field.dtype),
        }
        send = {
            side: self._edge_scalar_values(field, side)
            for side in recv
        }
        peers = {
            2: self.prev_procx,
            0: self.next_procx,
            3: self.prev_procy,
            1: self.next_procy,
        }

        reqs = []
        for side, peer in peers.items():
            if hasattr(self.comm, "Irecv"):
                reqs.append((self.comm.Irecv(recv[side], source=peer), False))
            else:
                req = self.comm.Recv_init(recv[side], source=peer)
                req.Start()
                reqs.append((req, True))

        for side, peer in peers.items():
            if hasattr(self.comm, "Isend"):
                reqs.append((self.comm.Isend(send[side], dest=peer), False))
            else:
                req = self.comm.Send_init(send[side], dest=peer)
                req.Start()
                reqs.append((req, True))

        for req, is_persistent in reqs:
            req.Wait()
            if is_persistent and hasattr(req, "Free"):
                req.Free()

        return {self._side_name(side): values for side, values in recv.items()}

    def _scalar_boundary_values(self, fields):
        if self.parallel:
            return {
                self.face_name: self._exchange_scalar_boundary_values(fields[self.face_name])
            }

        out = {name: {} for name in self.active_face_names}
        for name in self.active_face_names:
            face = self.faces[name]
            for neighbour_name, (side, neighbour_side) in face.connections:
                out[name][self._side_name(side)] = self._edge_scalar_values(
                    fields[neighbour_name], neighbour_side
                )
        return out

    def continuous_projection(self, coeffs):
        fields = self._coeffs_by_face(coeffs)
        boundary_values = self._scalar_boundary_values(fields)
        return {
            name: self.faces[name].continuous_projection(fields[name], boundary_values[name])
            for name in self.active_face_names
        }

    def evaluate_cartesian(self, x, y, z, coeffs):
        """
        Evaluate scalar DG coefficients at Cartesian points on the sphere.

        ``coeffs`` accepts the same forms as ``evaluate_latlong``: a dict keyed
        by face name, or an array whose first axis follows ``self.face_names``.
        """
        x, y, z = np.broadcast_arrays(x, y, z)
        out = self._evaluate_cartesian_many(x, y, z, [coeffs])[0]
        return out

    def _evaluate_cartesian_many(self, x, y, z, coeffs_list):
        coeffs_by_face = [self._coeffs_by_face(coeffs) for coeffs in coeffs_list]
        return self._evaluate_cartesian_many_by_face(x, y, z, coeffs_by_face)

    def _evaluate_cartesian_many_by_face(self, x, y, z, coeffs_by_face):
        if self.parallel:
            raise NotImplementedError(
                "Sphere-wide Cartesian evaluation is only available in serial runs."
            )

        x, y, z = np.broadcast_arrays(x, y, z)
        out_shape = x.shape
        x_flat = x.ravel()
        y_flat = y.ravel()
        z_flat = z.ravel()
        face_names = face_name_from_cartesian(x_flat, y_flat, z_flat).ravel()

        outputs = [None for _ in coeffs_by_face]

        for name in self.active_face_names:
            mask = face_names == name
            if not mask.any():
                continue

            face = self.faces[name]
            x1, y1 = face.geometry.to_cubed_sphere(x_flat[mask], y_flat[mask], z_flat[mask])
            for idx, coeffs in enumerate(coeffs_by_face):
                values = face.evaluate(x1, y1, coeffs[name])
                if outputs[idx] is None:
                    outputs[idx] = np.empty(x_flat.shape, dtype=values.dtype)
                outputs[idx][mask] = values

        for idx, out in enumerate(outputs):
            if out is None:
                outputs[idx] = np.empty(x_flat.shape)
            outputs[idx] = outputs[idx].reshape(out_shape)

        return outputs


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
        local_total = sum(f.integrate(q[n]) for n, f in self.faces.items())
        if not self.parallel:
            return local_total

        if hasattr(self.comm, "reduce"):
            return self.comm.reduce(local_total, root=0)

        gathered = self.comm.gather(local_total, root=0)
        if self.rank == 0:
            return sum(gathered)
        return None

    def entropy(self):
        return {n: f.entropy() for n, f in self.faces.items()}

    def enstrophy(self):
        self.boundaries(self._current_state())
        return {n: f.enstrophy() for n, f in self.faces.items()}

    def vorticity(self, *, continuous=False):
        self.boundaries(self._current_state())
        vort = {n: f.vorticity() for n, f in self.faces.items()}
        if continuous:
            return self.continuous_projection(vort)
        return vort

    def q(self, *, continuous=False):
        self.boundaries(self._current_state())
        q = {n: f.q() for n, f in self.faces.items()}
        if continuous:
            return self.continuous_projection(q)
        return q

    def divergence(self, *, continuous=False):
        self.boundaries(self._current_state())
        div = {n: f.divergence() for n, f in self.faces.items()}
        if continuous:
            return self.continuous_projection(div)
        return div

    def mass(self):
        return {n: f.h for n, f in self.faces.items()}

    def _restart_tile(self, var):
        face = self.faces[self.face_name]
        return (
            self.face_name,
            self.x_proc_idx,
            self.y_proc_idx,
            np.ascontiguousarray(_to_numpy(getattr(face, var))),
        )

    def _restart_tile_slice(self, face):
        y0 = face.y_proc_idx * face.ny
        x0 = face.x_proc_idx * face.nx
        return np.s_[y0:y0 + face.ny, x0:x0 + face.nx, :, :]

    def _assemble_restart_face(self, tiles):
        expected_tiles = self.nprocx * self.nprocy
        if len(tiles) != expected_tiles:
            raise ValueError(
                f"Restart gather expected {expected_tiles} tiles per face; found {len(tiles)}."
            )

        local_shape = tiles[0][2].shape
        if len(local_shape) != 4:
            raise ValueError(f"Restart tile must be 4D; found shape {local_shape}.")

        local_ny, local_nx, n_eta, n_xi = local_shape
        out = np.empty(
            (local_ny * self.nprocy, local_nx * self.nprocx, n_eta, n_xi),
            dtype=tiles[0][2].dtype,
        )

        seen = set()
        for x_idx, y_idx, tile in tiles:
            if not (0 <= x_idx < self.nprocx and 0 <= y_idx < self.nprocy):
                raise ValueError(
                    f"Restart tile index ({x_idx}, {y_idx}) is outside "
                    f"({self.nprocx}, {self.nprocy})."
                )
            if tile.shape != local_shape:
                raise ValueError(
                    f"Restart tiles for a face must all have shape {local_shape}; "
                    f"found {tile.shape}."
                )
            key = (x_idx, y_idx)
            if key in seen:
                raise ValueError(f"Duplicate restart tile index {key}.")
            seen.add(key)

            y0 = y_idx * local_ny
            x0 = x_idx * local_nx
            out[y0:y0 + local_ny, x0:x0 + local_nx] = tile

        if len(seen) != expected_tiles:
            raise ValueError(
                f"Restart gather found {len(seen)} unique tiles; expected {expected_tiles}."
            )
        return out

    def _save_parallel_restart_var(self, var, fn_template, directory):
        gathered = self.comm.gather(self._restart_tile(var), root=0)
        if self.rank != 0:
            return

        by_face = {name: [] for name in self.face_names}
        for name, x_idx, y_idx, tile in gathered:
            by_face[name].append((x_idx, y_idx, _to_numpy(tile)))

        for name in self.face_names:
            data = self._assemble_restart_face(by_face[name])
            np.save(self.make_fp(var, name, fn_template, directory), data)

    def _load_restart_data(self, var, name, fn_template, directory):
        data = np.load(self.make_fp(var, name, fn_template, directory))
        face = self.faces[name]
        expected_shape = (face.global_ny, face.global_nx, face.n, face.n)
        if data.shape != expected_shape:
            raise ValueError(
                f"Restart {var}_{name} has shape {data.shape}; expected {expected_shape}."
            )
        return data[self._restart_tile_slice(face)]

    def _restart_barrier(self):
        if self.comm is None:
            return
        if hasattr(self.comm, "Barrier"):
            self.comm.Barrier()
        elif hasattr(self.comm, "barrier"):
            self.comm.barrier()

    def save_restart(self, fn_template, directory):
        vars = ['u', 'v', 'w', 'h']
        if self.parallel:
            for var in vars:
                self._save_parallel_restart_var(var, fn_template, directory)
            self._restart_barrier()
            return

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
            data = [self._load_restart_data(vars[i], name, fn_template, directory) for i in range(len(vars))]
            b = self.faces[name].b
            self.faces[name].set_initial_condition(*data)
            self.faces[name].b = b

        self.boundaries()

    def save_diagnostics(self, fn_template, directory):
        if self.parallel and self.rank != 0:
            return

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


class DGCubedSphereFace:
    """
    One face of the cubed sphere.
    """

    def __init__(
            self, name, poly_order, nx, ny, g, f, radius, eps, device='cpu',
            solution=None, a=0.0, ah=0.0, dtype=np.float64, bc='wall',
            flux_type="standard",
            x_proc_idx=0, y_proc_idx=0, nprocx=1, nprocy=1,
            global_nx=None, global_ny=None, x_min=None, x_max=None, y_min=None, y_max=None,
            **kwargs
        ):

        _reject_legacy_flux_kwargs(kwargs)
        valid_names = ['zp', 'zn', 'xp', 'xn', 'yp', 'yn']
        if not name in valid_names:
            raise ValueError(f'name: expected one of: {valid_names}. Found {name}.')
        flux_type = _validate_flux_type(flux_type)
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
        self.xperiodic = self.yperiodic = False
        self.bc = bc
        self.geometry = EquiangularFace(name, radius=radius)
        self.connections = self.geometry.connections
        self.flux_type = flux_type

        [xs_1d, w_x] = gll(poly_order, iterative=True)
        [y_1d, w_y] = gll(poly_order, iterative=True)
        self.gll_nodes = xs_1d

        explicit_domain = any(
            bound is not None for bound in (x_min, x_max, y_min, y_max)
        )
        if explicit_domain and any(
            bound is None for bound in (x_min, x_max, y_min, y_max)
        ):
            raise ValueError(
                "x_min, x_max, y_min, and y_max must all be provided together."
            )

        if explicit_domain:
            local_nx = nx
            local_ny = ny
            if global_nx is None:
                global_nx = local_nx
            if global_ny is None:
                global_ny = local_ny
        else:
            global_nx = nx if global_nx is None else global_nx
            global_ny = ny if global_ny is None else global_ny
            if global_nx <= 0 or global_ny <= 0:
                raise ValueError(
                    f"Face dimensions must be positive; got nx={global_nx}, ny={global_ny}."
                )
            if global_nx % nprocx != 0:
                raise ValueError(
                    f"nx must be divisible by nprocx; got nx={nx}, nprocx={nprocx}."
                )
            if global_ny % nprocy != 0:
                raise ValueError(
                    f"ny must be divisible by nprocy; got ny={ny}, nprocy={nprocy}."
                )
            local_nx = global_nx // nprocx
            local_ny = global_ny // nprocy
            x_start = x_proc_idx * local_nx
            y_start = y_proc_idx * local_ny
            dx = 1.0 / global_nx
            dy = 1.0 / global_ny
            x_min = -0.5 + x_start * dx
            x_max = -0.5 + (x_start + local_nx) * dx
            y_min = -0.5 + y_start * dy
            y_max = -0.5 + (y_start + local_ny) * dy

        if local_nx <= 0 or local_ny <= 0:
            raise ValueError(
                f"Local face dimensions must be positive; got local_nx={local_nx}, local_ny={local_ny}."
            )

        self.global_nx = global_nx
        self.global_ny = global_ny
        self.x_proc_idx = x_proc_idx
        self.y_proc_idx = y_proc_idx
        self.nprocx = nprocx
        self.nprocy = nprocy

        xs = np.linspace(x_min, x_max, local_nx + 1)
        ys = np.linspace(y_min, y_max, local_ny + 1)
        self.x_min = xs[0]
        self.y_min = ys[0]
        self.x_max = xs[-1]
        self.y_max = ys[-1]

        self.x1, self.y1, lx, ly = element_grid_coordinates(xs, ys, xs_1d, y_1d)
        self.lx = lx
        self.ly = ly

        self.cdt = eps * radius * min(lx, ly) / (2 * poly_order + 1)  # this should be multiplied by pi / (2 * sqrt(2)) = 1.11... but eh a slightly smaller time step can't hurt

        w_x, w_y = np.meshgrid(w_x, w_y)
        self.weights_x = w_x[0][None, None, ...]
        self.weights = w_x * w_y

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
        edge_weights = self.edge_weights[None, None, :]
        self.vert_upper_edge_factor = self.J_vertface[:, :, -1, :] * edge_weights * self.inv_Jw[:, :, -1, :]
        self.vert_lower_edge_factor = self.J_vertface[:, :, 0, :] * edge_weights * self.inv_Jw[:, :, 0, :]
        self.horz_right_edge_factor = self.J_horzface[:, :, :, -1] * edge_weights * self.inv_Jw[:, :, :, -1]
        self.horz_left_edge_factor = self.J_horzface[:, :, :, 0] * edge_weights * self.inv_Jw[:, :, :, 0]

    def make_left_right_arrays(self, arr):
        return left_right_edge_arrays(arr, self.ny, self.nx, self.n, self.dtype)

    def make_up_down_arrays(self, arr):
        return up_down_edge_arrays(arr, self.ny, self.nx, self.n, self.dtype)

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
        return 0.5 * h * (u ** 2 + v ** 2 + w ** 2 + self.g * h) + h * self.b * self.g

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

        self.boundaries(u, v, w, h, self.time)

        u_cov, v_cov, _ = self.phys_to_cov(u, v, w)
        vort_cov = self.ddxi(v_cov) - self.ddeta(u_cov)

        u_cov_up = self.u_up * self.dxdxi_up + self.v_up * self.dydxi_up + self.w_up * self.dzdxi_up
        u_cov_down = self.u_down * self.dxdxi_down + self.v_down * self.dydxi_down + self.w_down * self.dzdxi_down
        v_cov_right = self.u_right * self.dxdeta_right + self.v_right * self.dydeta_right + self.w_right * self.dzdeta_right
        v_cov_left = self.u_left * self.dxdeta_left + self.v_left * self.dydeta_left + self.w_left * self.dzdeta_left

        u_cov_vert = 0.5 * (u_cov_up + u_cov_down)
        v_cov_horz = 0.5 * (v_cov_right + v_cov_left)
        endpoint_weight = self.endpoint_weight

        vort_cov[:, :, -1] -= (u_cov_vert[1:] - u_cov_down[1:]) / endpoint_weight
        vort_cov[:, :, 0] += (u_cov_vert[:-1] - u_cov_up[:-1]) / endpoint_weight
        vort_cov[:, :, :, -1] += (v_cov_horz[:, 1:] - v_cov_left[:, 1:]) / endpoint_weight
        vort_cov[:, :, :, 0] -= (v_cov_horz[:, :-1] - v_cov_right[:, :-1]) / endpoint_weight

        return vort_cov / self.J + self.f

    def vorticity(self, u=None, v=None, w=None, h=None, *, continuous=False, boundary_values=None):
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if h is None:
            h = self.h

        vort = self.dg_vort(u, v, w, h)
        if continuous:
            return self.continuous_projection(vort, boundary_values=boundary_values)
        return vort

    def dg_divergence(self, u=None, v=None, w=None, h=None):
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if h is None:
            h = self.h

        self.boundaries(u, v, w, h, self.time)

        u_contra, v_contra = self.phys_to_contra(u, v, w)
        div = (self.ddxi(u_contra * self.J) + self.ddeta(v_contra * self.J)) / self.J

        normal_up = self.u_up * self.eta_x_up + self.v_up * self.eta_y_up + self.w_up * self.eta_z_up
        normal_down = self.u_down * self.eta_x_down + self.v_down * self.eta_y_down + self.w_down * self.eta_z_down
        normal_right = self.u_right * self.xi_x_right + self.v_right * self.xi_y_right + self.w_right * self.xi_z_right
        normal_left = self.u_left * self.xi_x_left + self.v_left * self.xi_y_left + self.w_left * self.xi_z_left

        flux_vert = 0.5 * (normal_up + normal_down)
        flux_horz = 0.5 * (normal_right + normal_left)

        correction = np.zeros_like(div)
        correction[:, :, -1] += (
            (flux_vert[1:] - normal_down[1:]) * self.weights_x * self.J_vertface[:, :, -1]
        )
        correction[:, :, 0] -= (
            (flux_vert[:-1] - normal_up[:-1]) * self.weights_x * self.J_vertface[:, :, 0]
        )
        correction[:, :, :, -1] += (
            (flux_horz[:, 1:] - normal_left[:, 1:]) * self.weights_x * self.J_horzface[:, :, :, -1]
        )
        correction[:, :, :, 0] -= (
            (flux_horz[:, :-1] - normal_right[:, :-1]) * self.weights_x * self.J_horzface[:, :, :, 0]
        )

        return div + correction / self.Jw

    def divergence(self, u=None, v=None, w=None, h=None, *, continuous=False, boundary_values=None):
        div = self.dg_divergence(u, v, w, h)
        if continuous:
            return self.continuous_projection(div, boundary_values=boundary_values)
        return div

    def q(self, u=None, v=None, w=None, h=None):
        if h is None:
            h = self.h
        return self.dg_vort(u, v, w, h) / h

    def continuous_projection(self, field, boundary_values=None):
        return continuous_element_projection(
            field,
            self.Jw,
            boundary_values=boundary_values,
            xperiodic=self.xperiodic,
            yperiodic=self.yperiodic,
        )

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
        if self.flux_type == "barth":
            u_k, v_k, w_k, h_k = _solve_numba_kernel(
                u, v, w, h, self.b,
                self.D, self.endpoint_weight, self.J,
                self.vert_upper_edge_factor, self.vert_lower_edge_factor,
                self.horz_right_edge_factor, self.horz_left_edge_factor,
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
                self.dxdxi_up, self.dydxi_up, self.dzdxi_up, self.dxdxi_down, self.dydxi_down, self.dzdxi_down,
                self.dxdxi_right, self.dydxi_right, self.dzdxi_right, self.dxdxi_left, self.dydxi_left, self.dzdxi_left,
                self.dxdeta_up, self.dydeta_up, self.dzdeta_up, self.dxdeta_down, self.dydeta_down, self.dzdeta_down,
                self.dxdeta_right, self.dydeta_right, self.dzdeta_right, self.dxdeta_left, self.dydeta_left, self.dzdeta_left,
                self.g, 0.0, 0.0, False,
            )
            _apply_barth_diss_numba(
                u_k, v_k, w_k, h_k, self.g,
                self.vert_upper_edge_factor, self.vert_lower_edge_factor,
                self.horz_right_edge_factor, self.horz_left_edge_factor,
                self.u_up, self.v_up, self.w_up, self.h_up, self.u_down, self.v_down, self.w_down, self.h_down,
                self.u_right, self.v_right, self.w_right, self.h_right, self.u_left, self.v_left, self.w_left, self.h_left,
                self.eta_x_up, self.eta_y_up, self.eta_z_up, self.eta_x_down, self.eta_y_down, self.eta_z_down,
                self.xi_x_right, self.xi_y_right, self.xi_z_right, self.xi_x_left, self.xi_y_left, self.xi_z_left,
                self.dxdxi_up, self.dydxi_up, self.dzdxi_up, self.dxdxi_down, self.dydxi_down, self.dzdxi_down,
                self.dxdxi_right, self.dydxi_right, self.dzdxi_right, self.dxdxi_left, self.dydxi_left, self.dzdxi_left,
                self.dxdeta_up, self.dydeta_up, self.dzdeta_up, self.dxdeta_down, self.dydeta_down, self.dzdeta_down,
                self.dxdeta_right, self.dydeta_right, self.dzdeta_right, self.dxdeta_left, self.dydeta_left, self.dzdeta_left,
            )
            return u_k, v_k, w_k, h_k

        if self.flux_type == "barth_normal_tangent":
            u_k, v_k, w_k, h_k = _solve_numba_kernel(
                u, v, w, h, self.b,
                self.D, self.endpoint_weight, self.J,
                self.vert_upper_edge_factor, self.vert_lower_edge_factor,
                self.horz_right_edge_factor, self.horz_left_edge_factor,
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
                self.dxdxi_up, self.dydxi_up, self.dzdxi_up, self.dxdxi_down, self.dydxi_down, self.dzdxi_down,
                self.dxdxi_right, self.dydxi_right, self.dzdxi_right, self.dxdxi_left, self.dydxi_left, self.dzdxi_left,
                self.dxdeta_up, self.dydeta_up, self.dzdeta_up, self.dxdeta_down, self.dydeta_down, self.dzdeta_down,
                self.dxdeta_right, self.dydeta_right, self.dzdeta_right, self.dxdeta_left, self.dydeta_left, self.dzdeta_left,
                self.g, 0.0, 0.0, False,
            )
            _apply_barth_normal_tangent_diss_numba(
                u_k, v_k, w_k, h_k, self.g,
                self.vert_upper_edge_factor, self.vert_lower_edge_factor,
                self.horz_right_edge_factor, self.horz_left_edge_factor,
                self.u_up, self.v_up, self.w_up, self.h_up, self.u_down, self.v_down, self.w_down, self.h_down,
                self.u_right, self.v_right, self.w_right, self.h_right, self.u_left, self.v_left, self.w_left, self.h_left,
                self.eta_x_up, self.eta_y_up, self.eta_z_up, self.eta_x_down, self.eta_y_down, self.eta_z_down,
                self.xi_x_right, self.xi_y_right, self.xi_z_right, self.xi_x_left, self.xi_y_left, self.xi_z_left,
            )
            return u_k, v_k, w_k, h_k

        if self.flux_type == "old_tangent":
            return _solve_numba_old_tangent_kernel(
                u, v, w, h, self.b,
                self.D, self.endpoint_weight, self.J,
                self.vert_upper_edge_factor, self.vert_lower_edge_factor,
                self.horz_right_edge_factor, self.horz_left_edge_factor,
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
                self.J_eta, self.J_xi,
                self.dxdxi_up, self.dydxi_up, self.dzdxi_up, self.dxdxi_down, self.dydxi_down, self.dzdxi_down,
                self.dxdxi_right, self.dydxi_right, self.dzdxi_right, self.dxdxi_left, self.dydxi_left, self.dzdxi_left,
                self.dxdeta_up, self.dydeta_up, self.dzdeta_up, self.dxdeta_down, self.dydeta_down, self.dzdeta_down,
                self.dxdeta_right, self.dydeta_right, self.dzdeta_right, self.dxdeta_left, self.dydeta_left, self.dzdeta_left,
                self.g, self.a, self.ah,
            )

        kernel = _solve_numba_lmars_kernel if self.flux_type == "lmars" else _solve_numba_kernel
        return kernel(
            u, v, w, h, self.b,
            self.D, self.endpoint_weight, self.J,
            self.vert_upper_edge_factor, self.vert_lower_edge_factor,
            self.horz_right_edge_factor, self.horz_left_edge_factor,
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
            self.dxdxi_up, self.dydxi_up, self.dzdxi_up, self.dxdxi_down, self.dydxi_down, self.dzdxi_down,
            self.dxdxi_right, self.dydxi_right, self.dzdxi_right, self.dxdxi_left, self.dydxi_left, self.dzdxi_left,
            self.dxdeta_up, self.dydeta_up, self.dzdeta_up, self.dxdeta_down, self.dydeta_down, self.dzdeta_down,
            self.dxdeta_right, self.dydeta_right, self.dzdeta_right, self.dxdeta_left, self.dydeta_left, self.dzdeta_left,
            self.g, self.a, self.ah, self.flux_type == "standard_tangent",
        )

    def solve_numpy(self, u, v, w, h, t, dt, *, verbose=False):
        if self.flux_type == "barth":
            return self.solve_numpy_barth_diss(u, v, w, h, t, dt, verbose=verbose)
        if self.flux_type == "barth_normal_tangent":
            return self.solve_numpy_barth_normal_tangent_diss(u, v, w, h, t, dt, verbose=verbose)
        if self.flux_type == "lmars":
            return self.solve_numpy_lmars(u, v, w, h, t, dt, verbose=verbose)
        if self.flux_type == "old_tangent":
            return self.solve_numpy_old_tangent(u, v, w, h, t, dt, verbose=verbose)
        return self._solve_numpy_standard(
            u, v, w, h, t, dt, tangent_diss=self.flux_type == "standard_tangent", verbose=verbose
        )

    def solve_numpy_lmars(self, u, v, w, h, t, dt, *, verbose=False):
        return self._solve_numpy_lmars_kernel(
            u, v, w, h, t, dt, tangent_diss=False, verbose=verbose
        )

    def solve_numpy_old_tangent(self, u, v, w, h, t, dt, *, verbose=False):
        u_k, v_k, w_k, h_k = self._solve_numpy_standard(
            u, v, w, h, t, dt, tangent_diss=False, verbose=verbose
        )
        self._apply_old_tangent_diss_numpy(u_k, v_k, w_k)
        return u_k, v_k, w_k, h_k

    def solve_numpy_barth_diss(self, u, v, w, h, t, dt, *, verbose=False):
        u_k, v_k, w_k, h_k = self._solve_numpy_standard(
            u, v, w, h, t, dt, tangent_diss=False, a=0.0, ah=0.0, verbose=verbose
        )
        self._apply_barth_diss_numpy(u_k, v_k, w_k, h_k)
        return u_k, v_k, w_k, h_k

    def solve_numpy_barth_normal_tangent_diss(self, u, v, w, h, t, dt, *, verbose=False):
        u_k, v_k, w_k, h_k = self._solve_numpy_standard(
            u, v, w, h, t, dt, tangent_diss=False, a=0.0, ah=0.0, verbose=verbose
        )
        self._apply_barth_normal_tangent_diss_numpy(u_k, v_k, w_k, h_k)
        return u_k, v_k, w_k, h_k

    def _solve_numpy_standard(self, u, v, w, h, t, dt, *, tangent_diss, a=None, ah=None, verbose=False):
        if a is None:
            a = self.a
        if ah is None:
            ah = self.ah

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

        c_snd_ho = 0.5 * (np.sqrt(self.g * self.h_right) + np.sqrt(self.g * self.h_left))
        c_snd_ve = 0.5 * (np.sqrt(self.g * self.h_up) + np.sqrt(self.g * self.h_down))
        # c_ho = c_adv_horz + c_snd_ho
        # c_ve = c_adv_vert + c_snd_ve
        h_ve = 0.5 * (self.h_up + self.h_down)
        h_ho = 0.5 * (self.h_right + self.h_left)

        c_adv_vert = 0.5 * (self.h_up * vel_up + self.h_down * vel_down) / h_ve #- self.g * self.ah * (self.h_up - self.h_down) / (c_snd_ve * h_ve)
        c_adv_horz = 0.5 * (self.h_right * vel_right + self.h_left * vel_left) / h_ho #- self.g * self.ah * (self.h_right - self.h_left) / (c_snd_ho * h_ho)

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

        h_flux_vert = c_adv_vert * h_ve - ah * abs(c_adv_vert) * (self.h_up - self.h_down)
        h_flux_horz = c_adv_horz * h_ho - ah * abs(c_adv_horz) * (self.h_right - self.h_left)

        uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux) - a * (c_snd_ho + abs(c_adv_horz)) * (h_right_flux - h_left_flux) / h_ho
        uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux) - a * (c_snd_ve + abs(c_adv_vert)) * (h_up_flux - h_down_flux) / h_ve

        if tangent_diss:
            u_cov_vert_avg = (c_adv_vert < 0) * u_cov_up + (c_adv_vert >= 0) * u_cov_down
            v_cov_horz_avg = (c_adv_horz < 0) * v_cov_right + (c_adv_horz >= 0) * v_cov_left
        else:
            u_cov_vert_avg = 0.5 * (u_cov_up + u_cov_down)
            v_cov_horz_avg = 0.5 * (v_cov_right + v_cov_left)

        u_flux_vert_up = v_contra_up * u_cov_vert_avg
        u_flux_vert_down = v_contra_down * u_cov_vert_avg
        v_flux_vert_up = uv_flux_vert - u_contra_up * u_cov_vert_avg
        v_flux_vert_down = uv_flux_vert - u_contra_down * u_cov_vert_avg

        u_flux_horz_right = uv_flux_horz - v_contra_right * v_cov_horz_avg
        u_flux_horz_left = uv_flux_horz - v_contra_left * v_cov_horz_avg
        v_flux_horz_right = u_contra_right * v_cov_horz_avg
        v_flux_horz_left = u_contra_left * v_cov_horz_avg

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

        u_k[:, :, -1] -= (u_flux_vert_down[1:] - (v_contra_down * u_cov_down)[1:]) / wx
        u_k[:, :, 0] += (u_flux_vert_up[:-1] - (v_contra_up * u_cov_up)[:-1]) / wx
        u_k[:, :, :, -1] -= (
            u_flux_horz_left[:, 1:] - (uv_left_flux - v_contra_left * v_cov_left)[:, 1:]
        ) / wx
        u_k[:, :, :, 0] += (
            u_flux_horz_right[:, :-1] - (uv_right_flux - v_contra_right * v_cov_right)[:, :-1]
        ) / wx

        # handle v
        #######
        ###

        v_k = -self.ddeta(uv_flux)
        v_k += -u_contra * abs_vort_cov

        v_k[:, :, -1] -= (
            v_flux_vert_down[1:] - (uv_down_flux - u_contra_down * u_cov_down)[1:]
        ) / wx
        v_k[:, :, 0] += (
            v_flux_vert_up[:-1] - (uv_up_flux - u_contra_up * u_cov_up)[:-1]
        ) / wx
        v_k[:, :, :, -1] -= (
            v_flux_horz_left[:, 1:] - (u_contra_left * v_cov_left)[:, 1:]
        ) / wx
        v_k[:, :, :, 0] += (
            v_flux_horz_right[:, :-1] - (u_contra_right * v_cov_right)[:, :-1]
        ) / wx

        u_k, v_k, w_k = self.cov_to_phys(u_k, v_k, 0)

        return u_k, v_k, w_k, h_k

    def _barth_dissipation_numpy(
            self,
            u_l, v_l, w_l, h_l,
            u_r, v_r, w_r, h_r,
            n_x, n_y, n_z,
            t_x, t_y, t_z,
    ):
        n_norm = np.sqrt(n_x ** 2 + n_y ** 2 + n_z ** 2)
        n_x = n_x / n_norm
        n_y = n_y / n_norm
        n_z = n_z / n_norm

        t_norm = np.sqrt(t_x ** 2 + t_y ** 2 + t_z ** 2)
        t_x = t_x / t_norm
        t_y = t_y / t_norm
        t_z = t_z / t_norm

        h_avg = 0.5 * (h_l + h_r)
        c = np.sqrt(self.g * h_avg)
        u_avg = 0.5 * (u_l + u_r)
        v_avg = 0.5 * (v_l + v_r)
        w_avg = 0.5 * (w_l + w_r)
        du = u_r - u_l
        dv = v_r - v_l
        dw = w_r - w_l
        dh = h_r - h_l

        u_n = u_avg * n_x + v_avg * n_y + w_avg * n_z
        du_n = du * n_x + dv * n_y + dw * n_z
        du_t = du * t_x + dv * t_y + dw * t_z

        mu_m = np.abs(u_n - c)
        mu_0 = np.abs(u_n)
        mu_p = np.abs(u_n + c)
        amp_m = (self.g * dh - c * du_n) / (2.0 * self.g)
        amp_p = (self.g * dh + c * du_n) / (2.0 * self.g)

        diss_h = mu_m * amp_m + mu_p * amp_p
        amp_0 = h_avg * du_t
        diss_u = (
            mu_m * amp_m * (u_avg - c * n_x)
            + mu_p * amp_p * (u_avg + c * n_x)
            + mu_0 * amp_0 * t_x
        )
        diss_v = (
            mu_m * amp_m * (v_avg - c * n_y)
            + mu_p * amp_p * (v_avg + c * n_y)
            + mu_0 * amp_0 * t_y
        )
        diss_w = (
            mu_m * amp_m * (w_avg - c * n_z)
            + mu_p * amp_p * (w_avg + c * n_z)
            + mu_0 * amp_0 * t_z
        )
        return diss_u, diss_v, diss_w, diss_h

    def _apply_barth_diss_numpy(self, u_k, v_k, w_k, h_k):
        diss_u, diss_v, diss_w, diss_h = self._barth_dissipation_numpy(
            self.u_down, self.v_down, self.w_down, self.h_down,
            self.u_up, self.v_up, self.w_up, self.h_up,
            0.5 * (self.eta_x_down + self.eta_x_up),
            0.5 * (self.eta_y_down + self.eta_y_up),
            0.5 * (self.eta_z_down + self.eta_z_up),
            0.5 * (self.dxdxi_down + self.dxdxi_up),
            0.5 * (self.dydxi_down + self.dydxi_up),
            0.5 * (self.dzdxi_down + self.dzdxi_up),
        )

        scale = 0.5 * self.vert_upper_edge_factor
        dh_k = scale * diss_h[1:]
        dm_u = scale * diss_u[1:]
        dm_v = scale * diss_v[1:]
        dm_w = scale * diss_w[1:]
        h_l = self.h_down[1:]
        h_k[:, :, -1] += dh_k
        u_k[:, :, -1] += (dm_u - self.u_down[1:] * dh_k) / h_l
        v_k[:, :, -1] += (dm_v - self.v_down[1:] * dh_k) / h_l
        w_k[:, :, -1] += (dm_w - self.w_down[1:] * dh_k) / h_l

        scale = -0.5 * self.vert_lower_edge_factor
        dh_k = scale * diss_h[:-1]
        dm_u = scale * diss_u[:-1]
        dm_v = scale * diss_v[:-1]
        dm_w = scale * diss_w[:-1]
        h_r = self.h_up[:-1]
        h_k[:, :, 0] += dh_k
        u_k[:, :, 0] += (dm_u - self.u_up[:-1] * dh_k) / h_r
        v_k[:, :, 0] += (dm_v - self.v_up[:-1] * dh_k) / h_r
        w_k[:, :, 0] += (dm_w - self.w_up[:-1] * dh_k) / h_r

        diss_u, diss_v, diss_w, diss_h = self._barth_dissipation_numpy(
            self.u_left, self.v_left, self.w_left, self.h_left,
            self.u_right, self.v_right, self.w_right, self.h_right,
            0.5 * (self.xi_x_left + self.xi_x_right),
            0.5 * (self.xi_y_left + self.xi_y_right),
            0.5 * (self.xi_z_left + self.xi_z_right),
            0.5 * (self.dxdeta_left + self.dxdeta_right),
            0.5 * (self.dydeta_left + self.dydeta_right),
            0.5 * (self.dzdeta_left + self.dzdeta_right),
        )

        scale = 0.5 * self.horz_right_edge_factor
        dh_k = scale * diss_h[:, 1:]
        dm_u = scale * diss_u[:, 1:]
        dm_v = scale * diss_v[:, 1:]
        dm_w = scale * diss_w[:, 1:]
        h_l = self.h_left[:, 1:]
        h_k[:, :, :, -1] += dh_k
        u_k[:, :, :, -1] += (dm_u - self.u_left[:, 1:] * dh_k) / h_l
        v_k[:, :, :, -1] += (dm_v - self.v_left[:, 1:] * dh_k) / h_l
        w_k[:, :, :, -1] += (dm_w - self.w_left[:, 1:] * dh_k) / h_l

        scale = -0.5 * self.horz_left_edge_factor
        dh_k = scale * diss_h[:, :-1]
        dm_u = scale * diss_u[:, :-1]
        dm_v = scale * diss_v[:, :-1]
        dm_w = scale * diss_w[:, :-1]
        h_r = self.h_right[:, :-1]
        h_k[:, :, :, 0] += dh_k
        u_k[:, :, :, 0] += (dm_u - self.u_right[:, :-1] * dh_k) / h_r
        v_k[:, :, :, 0] += (dm_v - self.v_right[:, :-1] * dh_k) / h_r
        w_k[:, :, :, 0] += (dm_w - self.w_right[:, :-1] * dh_k) / h_r

    def _barth_1d_normal_dissipation_numpy(
            self, u_l, v_l, w_l, h_l, u_r, v_r, w_r, h_r, n_x, n_y, n_z):
        n_norm = np.sqrt(n_x ** 2 + n_y ** 2 + n_z ** 2)
        n_x = n_x / n_norm
        n_y = n_y / n_norm
        n_z = n_z / n_norm

        un_l = u_l * n_x + v_l * n_y + w_l * n_z
        un_r = u_r * n_x + v_r * n_y + w_r * n_z
        h_avg = 0.5 * (h_l + h_r)
        c = np.sqrt(self.g * h_avg)
        u_n = 0.5 * (un_l + un_r)
        du_n = un_r - un_l
        dh = h_r - h_l

        mu_m = np.abs(u_n - c)
        mu_p = np.abs(u_n + c)
        amp_m = (self.g * dh - c * du_n) / (2.0 * self.g)
        amp_p = (self.g * dh + c * du_n) / (2.0 * self.g)

        diss_h = mu_m * amp_m + mu_p * amp_p
        diss_un = mu_m * amp_m * (u_n - c) + mu_p * amp_p * (u_n + c)
        return n_x, n_y, n_z, un_l, un_r, diss_h, diss_un

    def _apply_barth_normal_tangent_diss_numpy(self, u_k, v_k, w_k, h_k):
        n_x, n_y, n_z, un_l, un_r, diss_h, diss_un = self._barth_1d_normal_dissipation_numpy(
            self.u_down, self.v_down, self.w_down, self.h_down,
            self.u_up, self.v_up, self.w_up, self.h_up,
            0.5 * (self.eta_x_down + self.eta_x_up),
            0.5 * (self.eta_y_down + self.eta_y_up),
            0.5 * (self.eta_z_down + self.eta_z_up),
        )

        scale = 0.5 * self.vert_upper_edge_factor
        dh_k = scale * diss_h[1:]
        h_l = self.h_down[1:]
        dun = (scale * diss_un[1:] - un_l[1:] * dh_k) / h_l
        h_k[:, :, -1] += dh_k
        u_k[:, :, -1] += dun * n_x[1:]
        v_k[:, :, -1] += dun * n_y[1:]
        w_k[:, :, -1] += dun * n_z[1:]

        scale = -0.5 * self.vert_lower_edge_factor
        dh_k = scale * diss_h[:-1]
        h_r = self.h_up[:-1]
        dun = (scale * diss_un[:-1] - un_r[:-1] * dh_k) / h_r
        h_k[:, :, 0] += dh_k
        u_k[:, :, 0] += dun * n_x[:-1]
        v_k[:, :, 0] += dun * n_y[:-1]
        w_k[:, :, 0] += dun * n_z[:-1]

        n_x, n_y, n_z, un_l, un_r, diss_h, diss_un = self._barth_1d_normal_dissipation_numpy(
            self.u_left, self.v_left, self.w_left, self.h_left,
            self.u_right, self.v_right, self.w_right, self.h_right,
            0.5 * (self.xi_x_left + self.xi_x_right),
            0.5 * (self.xi_y_left + self.xi_y_right),
            0.5 * (self.xi_z_left + self.xi_z_right),
        )

        scale = 0.5 * self.horz_right_edge_factor
        dh_k = scale * diss_h[:, 1:]
        h_l = self.h_left[:, 1:]
        dun = (scale * diss_un[:, 1:] - un_l[:, 1:] * dh_k) / h_l
        h_k[:, :, :, -1] += dh_k
        u_k[:, :, :, -1] += dun * n_x[:, 1:]
        v_k[:, :, :, -1] += dun * n_y[:, 1:]
        w_k[:, :, :, -1] += dun * n_z[:, 1:]

        scale = -0.5 * self.horz_left_edge_factor
        dh_k = scale * diss_h[:, :-1]
        h_r = self.h_right[:, :-1]
        dun = (scale * diss_un[:, :-1] - un_r[:, :-1] * dh_k) / h_r
        h_k[:, :, :, 0] += dh_k
        u_k[:, :, :, 0] += dun * n_x[:, :-1]
        v_k[:, :, :, 0] += dun * n_y[:, :-1]
        w_k[:, :, :, 0] += dun * n_z[:, :-1]

    def _solve_numpy_lmars_kernel(self, u, v, w, h, t, dt, *, tangent_diss, verbose=False):

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

        c_snd_ho = 0.5 * (np.sqrt(self.g * self.h_right) + np.sqrt(self.g * self.h_left)) - 0.25 * (vel_right - vel_left)
        c_snd_ve = 0.5 * (np.sqrt(self.g * self.h_up) + np.sqrt(self.g * self.h_down)) - 0.25 * (vel_up - vel_down)
        # c_ho = c_adv_horz + c_snd_ho
        # c_ve = c_adv_vert + c_snd_ve
        h_ve = 0.5 * (self.h_up + self.h_down)
        h_ho = 0.5 * (self.h_right + self.h_left)
        h_star_ve = c_snd_ve**2 / self.g
        h_star_ho = c_snd_ho**2 / self.g

        c_adv_vert = 0.5 * (vel_up + vel_down) + np.sqrt(self.g * self.h_down) - np.sqrt(self.g * self.h_up)
        c_adv_horz = 0.5 * (vel_right + vel_left) + np.sqrt(self.g * self.h_left) - np.sqrt(self.g * self.h_right)

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

        # h_flux_vert = c_adv_vert * h_ve - 0.5 * abs(c_adv_vert) * (self.h_up - self.h_down)
        # h_flux_horz = c_adv_horz * h_ho - 0.5 * abs(c_adv_horz) * (self.h_right - self.h_left)
        h_flux_vert = c_adv_vert * h_star_ve
        h_flux_horz = c_adv_horz * h_star_ho

        # uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux) - 0.5 * c_snd_ho * (h_right_flux - h_left_flux) / h_ho
        # uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux) - 0.5 * c_snd_ve * (h_up_flux - h_down_flux) / h_ve

        uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux) - 0.5 * (0.5 * vel_right**2 + 0.5 * vel_left**2 + self.g * self.h_right + self.g * self.h_left)
        uv_flux_horz += 0.5 * c_adv_horz**2 + self.g * h_star_ho
        uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux) - 0.5 * (0.5 * vel_up**2 + 0.5 * vel_down**2 + self.g * self.h_up + self.g * self.h_down)
        uv_flux_vert += 0.5 * c_adv_vert**2 + self.g * h_star_ve

        u_cov_vert_avg = (c_adv_vert < 0) * u_cov_up + (c_adv_vert >= 0) * u_cov_down
        v_cov_horz_avg = (c_adv_horz < 0) * v_cov_right + (c_adv_horz >= 0) * v_cov_left

        u_flux_vert_up = v_contra_up * u_cov_vert_avg
        u_flux_vert_down = v_contra_down * u_cov_vert_avg
        v_flux_vert_up = uv_flux_vert - u_contra_up * u_cov_vert_avg
        v_flux_vert_down = uv_flux_vert - u_contra_down * u_cov_vert_avg

        u_flux_horz_right = uv_flux_horz - v_contra_right * v_cov_horz_avg
        u_flux_horz_left = uv_flux_horz - v_contra_left * v_cov_horz_avg
        v_flux_horz_right = u_contra_right * v_cov_horz_avg
        v_flux_horz_left = u_contra_left * v_cov_horz_avg

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

        u_k[:, :, -1] -= (u_flux_vert_down[1:] - (v_contra_down * u_cov_down)[1:]) / wx
        u_k[:, :, 0] += (u_flux_vert_up[:-1] - (v_contra_up * u_cov_up)[:-1]) / wx
        u_k[:, :, :, -1] -= (
            u_flux_horz_left[:, 1:] - (uv_left_flux - v_contra_left * v_cov_left)[:, 1:]
        ) / wx
        u_k[:, :, :, 0] += (
            u_flux_horz_right[:, :-1] - (uv_right_flux - v_contra_right * v_cov_right)[:, :-1]
        ) / wx

        # handle v
        #######
        ###

        v_k = -self.ddeta(uv_flux)
        v_k += -u_contra * abs_vort_cov

        v_k[:, :, -1] -= (
            v_flux_vert_down[1:] - (uv_down_flux - u_contra_down * u_cov_down)[1:]
        ) / wx
        v_k[:, :, 0] += (
            v_flux_vert_up[:-1] - (uv_up_flux - u_contra_up * u_cov_up)[:-1]
        ) / wx
        v_k[:, :, :, -1] -= (
            v_flux_horz_left[:, 1:] - (u_contra_left * v_cov_left)[:, 1:]
        ) / wx
        v_k[:, :, :, 0] += (
            v_flux_horz_right[:, :-1] - (u_contra_right * v_cov_right)[:, :-1]
        ) / wx

        u_k, v_k, w_k = self.cov_to_phys(u_k, v_k, 0)

        return u_k, v_k, w_k, h_k

    def _apply_old_tangent_diss_numpy(self, u_k, v_k, w_k):
        h_up_flux = self.h_up * (
            self.u_up * self.eta_x_up
            + self.v_up * self.eta_y_up
            + self.w_up * self.eta_z_up
        )
        h_down_flux = self.h_down * (
            self.u_down * self.eta_x_down
            + self.v_down * self.eta_y_down
            + self.w_down * self.eta_z_down
        )
        h_right_flux = self.h_right * (
            self.u_right * self.xi_x_right
            + self.v_right * self.xi_y_right
            + self.w_right * self.xi_z_right
        )
        h_left_flux = self.h_left * (
            self.u_left * self.xi_x_left
            + self.v_left * self.xi_y_left
            + self.w_left * self.xi_z_left
        )

        vel_up = h_up_flux / self.h_up
        vel_down = h_down_flux / self.h_down
        vel_right = h_right_flux / self.h_right
        vel_left = h_left_flux / self.h_left
        h_ve = 0.5 * (self.h_up + self.h_down)
        h_ho = 0.5 * (self.h_right + self.h_left)
        c_adv_vert = 0.5 * abs(vel_up + vel_down)
        c_adv_horz = 0.5 * abs(vel_right + vel_left)

        u_cov_up = self.u_up * self.dxdxi_up + self.v_up * self.dydxi_up + self.w_up * self.dzdxi_up
        u_cov_down = self.u_down * self.dxdxi_down + self.v_down * self.dydxi_down + self.w_down * self.dzdxi_down
        u_cov_right = self.u_right * self.dxdxi_right + self.v_right * self.dydxi_right + self.w_right * self.dzdxi_right
        u_cov_left = self.u_left * self.dxdxi_left + self.v_left * self.dydxi_left + self.w_left * self.dzdxi_left

        v_cov_up = self.u_up * self.dxdeta_up + self.v_up * self.dydeta_up + self.w_up * self.dzdeta_up
        v_cov_down = self.u_down * self.dxdeta_down + self.v_down * self.dydeta_down + self.w_down * self.dzdeta_down
        v_cov_right = self.u_right * self.dxdeta_right + self.v_right * self.dydeta_right + self.w_right * self.dzdeta_right
        v_cov_left = self.u_left * self.dxdeta_left + self.v_left * self.dydeta_left + self.w_left * self.dzdeta_left

        wx = self.endpoint_weight

        diss_u = -0.5 * c_adv_vert * (self.h_up * u_cov_up - self.h_down * u_cov_down) / h_ve
        diss_v = -0.5 * c_adv_vert * (self.h_up * v_cov_up - self.h_down * v_cov_down) / h_ve
        diss_vel = 0.5 * c_adv_vert * (h_up_flux - h_down_flux) / h_ve

        du_cov = -diss_u[1:] * self.J_eta[:, :, -1] / wx
        dv_cov = -(diss_v[1:] * self.J_eta[:, :, -1] + diss_vel[1:]) / wx
        u_k[:, :, -1] += du_cov * self.dxidx[:, :, -1] + dv_cov * self.detadx[:, :, -1]
        v_k[:, :, -1] += du_cov * self.dxidy[:, :, -1] + dv_cov * self.detady[:, :, -1]
        w_k[:, :, -1] += du_cov * self.dxidz[:, :, -1] + dv_cov * self.detadz[:, :, -1]

        du_cov = diss_u[:-1] * self.J_eta[:, :, 0] / wx
        dv_cov = (diss_v[:-1] * self.J_eta[:, :, 0] + diss_vel[:-1]) / wx
        u_k[:, :, 0] += du_cov * self.dxidx[:, :, 0] + dv_cov * self.detadx[:, :, 0]
        v_k[:, :, 0] += du_cov * self.dxidy[:, :, 0] + dv_cov * self.detady[:, :, 0]
        w_k[:, :, 0] += du_cov * self.dxidz[:, :, 0] + dv_cov * self.detadz[:, :, 0]

        diss_u = -0.5 * c_adv_horz * (self.h_right * u_cov_right - self.h_left * u_cov_left) / h_ho
        diss_v = -0.5 * c_adv_horz * (self.h_right * v_cov_right - self.h_left * v_cov_left) / h_ho
        diss_vel = 0.5 * c_adv_horz * (h_right_flux - h_left_flux) / h_ho

        du_cov = -(diss_u[:, 1:] * self.J_xi[:, :, :, -1] + diss_vel[:, 1:]) / wx
        dv_cov = -diss_v[:, 1:] * self.J_xi[:, :, :, -1] / wx
        u_k[:, :, :, -1] += du_cov * self.dxidx[:, :, :, -1] + dv_cov * self.detadx[:, :, :, -1]
        v_k[:, :, :, -1] += du_cov * self.dxidy[:, :, :, -1] + dv_cov * self.detady[:, :, :, -1]
        w_k[:, :, :, -1] += du_cov * self.dxidz[:, :, :, -1] + dv_cov * self.detadz[:, :, :, -1]

        du_cov = (diss_u[:, :-1] * self.J_xi[:, :, :, 0] + diss_vel[:, :-1]) / wx
        dv_cov = diss_v[:, :-1] * self.J_xi[:, :, :, 0] / wx
        u_k[:, :, :, 0] += du_cov * self.dxidx[:, :, :, 0] + dv_cov * self.detadx[:, :, :, 0]
        v_k[:, :, :, 0] += du_cov * self.dxidy[:, :, :, 0] + dv_cov * self.detady[:, :, :, 0]
        w_k[:, :, :, 0] += du_cov * self.dxidz[:, :, :, 0] + dv_cov * self.detadz[:, :, :, 0]

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


__all__ = ["DGCubedSphereSWE", "DGCubedSphereFace"]
