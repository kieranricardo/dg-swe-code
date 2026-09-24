import os
import numpy as np

from dg_swe.dg_cubed_sphere_swe import DGCubedSphereFace, DGCubedSphereSWE
from dg_swe.tswe_numba_kernels import _solve_tswe_numba_kernel
from dg_swe.utils import cross_product, to_numpy as _to_numpy


class DGCubedSphereTSWE(DGCubedSphereSWE):
    """Thermal shallow-water solver on the cubed sphere."""

    face_class = None

    def __init__(
        self,
        poly_order,
        nx,
        ny,
        g,
        f,
        eps,
        radius=1.0,
        device="cpu",
        solution=None,
        a=0.0,
        ah=0.0,
        dtype=np.float64,
        flux_type="standard",
        upwind=False,
        nprocx=1,
        nprocy=1,
        comm=None,
        **kwargs,
    ):
        if flux_type != "standard":
            raise ValueError("DGCubedSphereTSWE currently supports flux_type='standard'.")

        self.face_class = DGCubedSphereFaceTSWE
        super().__init__(
            poly_order,
            nx,
            ny,
            g,
            f,
            eps,
            radius=radius,
            device=device,
            solution=solution,
            a=a,
            ah=ah,
            dtype=dtype,
            flux_type=flux_type,
            upwind=upwind,
            nprocx=nprocx,
            nprocy=nprocy,
            comm=comm,
            **kwargs,
        )
        self.upwind = upwind
        self.variance_list = []
        self.buoyancy_list = []
        for face in self.faces.values():
            face.upwind = self.upwind


    def boundaries(self, sol=None):
        if sol is None:
            sol = {
                n: (
                    self.faces[n].u,
                    self.faces[n].v,
                    self.faces[n].w,
                    self.faces[n].h,
                    self.faces[n].hb,
                )
                for n in self.active_face_names
            }

        if self.parallel:
            reqs = self.fill_boundaries(sol)
            self.recv_boundaries(reqs)
            return

        for name in self.active_face_names:
            face = self.faces[name]
            for neighbour_name, (side, neighbour_side) in face.connections:
                neighbour = self.faces[neighbour_name]
                self._assign_edge_state(
                    face,
                    side,
                    self._edge_state(neighbour, sol[neighbour_name], neighbour_side),
                )

    @staticmethod
    def _edge_state(face, state, side):
        u, v, w, h, hb = state
        if side == 0:
            data = (
                u[:, -1, :, -1],
                v[:, -1, :, -1],
                w[:, -1, :, -1],
                h[:, -1, :, -1],
                hb[:, -1, :, -1],
            )
        elif side == 1:
            data = (
                u[-1, :, -1],
                v[-1, :, -1],
                w[-1, :, -1],
                h[-1, :, -1],
                hb[-1, :, -1],
            )
        elif side == 2:
            data = (
                u[:, 0, :, 0],
                v[:, 0, :, 0],
                w[:, 0, :, 0],
                h[:, 0, :, 0],
                hb[:, 0, :, 0],
            )
        elif side == 3:
            data = (
                u[0, :, 0],
                v[0, :, 0],
                w[0, :, 0],
                h[0, :, 0],
                hb[0, :, 0],
            )
        else:
            raise ValueError(f"Unknown boundary side {side}.")
        return np.ascontiguousarray(np.stack(data))

    @staticmethod
    def _pack_edge_state(face, state, side, out):
        u, v, w, h, hb = state
        if side == 0:
            out[0] = u[:, -1, :, -1]
            out[1] = v[:, -1, :, -1]
            out[2] = w[:, -1, :, -1]
            out[3] = h[:, -1, :, -1]
            out[4] = hb[:, -1, :, -1]
        elif side == 1:
            out[0] = u[-1, :, -1]
            out[1] = v[-1, :, -1]
            out[2] = w[-1, :, -1]
            out[3] = h[-1, :, -1]
            out[4] = hb[-1, :, -1]
        elif side == 2:
            out[0] = u[:, 0, :, 0]
            out[1] = v[:, 0, :, 0]
            out[2] = w[:, 0, :, 0]
            out[3] = h[:, 0, :, 0]
            out[4] = hb[:, 0, :, 0]
        elif side == 3:
            out[0] = u[0, :, 0]
            out[1] = v[0, :, 0]
            out[2] = w[0, :, 0]
            out[3] = h[0, :, 0]
            out[4] = hb[0, :, 0]
        else:
            raise ValueError(f"Unknown boundary side {side}.")

    @staticmethod
    def _assign_edge_state(face, side, data):
        u, v, w, h, hb = data
        if side == 0:
            face.u_right[:, -1] = u
            face.v_right[:, -1] = v
            face.w_right[:, -1] = w
            face.h_right[:, -1] = h
            face.hb_right[:, -1] = hb
        elif side == 1:
            face.u_up[-1] = u
            face.v_up[-1] = v
            face.w_up[-1] = w
            face.h_up[-1] = h
            face.hb_up[-1] = hb
        elif side == 2:
            face.u_left[:, 0] = u
            face.v_left[:, 0] = v
            face.w_left[:, 0] = w
            face.h_left[:, 0] = h
            face.hb_left[:, 0] = hb
        elif side == 3:
            face.u_down[0] = u
            face.v_down[0] = v
            face.w_down[0] = w
            face.h_down[0] = h
            face.hb_down[0] = hb
        else:
            raise ValueError(f"Unknown boundary side {side}.")

    def _init_mpi_boundary_exchange(self):
        face = self.faces[self.face_name]
        nvars = 5
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

    def positivity_preserving_limiter(self, state, prev_state):
        limited = super().positivity_preserving_limiter(state, prev_state)
        for name in self.active_face_names:
            h = limited[name][3]
            hb = limited[name][4]
            hb[:] = np.maximum(hb, np.finfo(hb.dtype).eps)
        return limited

    def time_step(self, dt=None, order=3, forcing=None):
        if self.store_diagnostics:
            self.time_list.append(self.time)
            self.energy_list.append(self.integrate(self.energy()))
            self.variance_list.append(self.integrate(self.tracer_variance()))
            self.buoyancy_list.append(self.integrate(self.buoyancy()))
            self.mass_list.append(self.integrate(self.mass()))

        self.h = {n: f.h for n, f in self.faces.items()}
        self.hb = {n: f.hb for n, f in self.faces.items()}
        if dt is None:
            dt = self.get_dt()

        if order == 3:
            u = {n: self._face_state(n) for n in self.active_face_names}
            self.boundaries(u)
            k_1 = {n: self.faces[n].solve(*u[n], self.time, dt) for n in self.active_face_names}

            u_1 = {
                n: tuple(u[n][i] + dt * k_1[n][i] for i in range(5))
                for n in self.active_face_names
            }
            # u_1 = self.positivity_preserving_limiter(u_1, prev_state=u)
            self.boundaries(u_1)
            k_2 = {n: self.faces[n].solve(*u_1[n], self.time, dt) for n in self.active_face_names}

            u_2 = {
                n: tuple(0.75 * u[n][i] + 0.25 * (u_1[n][i] + dt * k_2[n][i]) for i in range(5))
                for n in self.active_face_names
            }
            # u_2 = self.positivity_preserving_limiter(u_2, prev_state=u_1)
            self.boundaries(u_2)
            k_3 = {n: self.faces[n].solve(*u_2[n], self.time, dt) for n in self.active_face_names}

            for n in self.active_face_names:
                self.faces[n].u = (self.faces[n].u + 2 * (u_2[n][0] + dt * k_3[n][0])) / 3
                self.faces[n].v = (self.faces[n].v + 2 * (u_2[n][1] + dt * k_3[n][1])) / 3
                self.faces[n].w = (self.faces[n].w + 2 * (u_2[n][2] + dt * k_3[n][2])) / 3
                self.faces[n].h = (self.faces[n].h + 2 * (u_2[n][3] + dt * k_3[n][3])) / 3
                self.faces[n].hb = (self.faces[n].hb + 2 * (u_2[n][4] + dt * k_3[n][4])) / 3

            u = {n: self._face_state(n) for n in self.active_face_names}
            # u = self.positivity_preserving_limiter(u, prev_state=u_2)
            self.boundaries(u)

        elif order == 34:
            u = {n: self._face_state(n) for n in self.active_face_names}
            self.boundaries(u)
            k_1 = {n: self.faces[n].solve(*u[n], self.time, dt) for n in self.active_face_names}

            u_1 = {
                n: tuple(u[n][i] + 0.5 * dt * k_1[n][i] for i in range(5))
                for n in self.active_face_names
            }
            # u_1 = self.positivity_preserving_limiter(u_1, prev_state=u)
            self.boundaries(u_1)
            k_2 = {n: self.faces[n].solve(*u_1[n], self.time, dt) for n in self.active_face_names}

            u_2 = {
                n: tuple(u_1[n][i] + 0.5 * dt * k_2[n][i] for i in range(5))
                for n in self.active_face_names
            }
            # u_2 = self.positivity_preserving_limiter(u_2, prev_state=u_1)
            self.boundaries(u_2)
            k_3 = {n: self.faces[n].solve(*u_2[n], self.time, dt) for n in self.active_face_names}

            u_3 = {
                n: tuple((2 / 3) * u[n][i] + (1 / 3) * u_2[n][i] + (1 / 6) * dt * k_3[n][i] for i in range(5))
                for n in self.active_face_names
            }
            # u_3 = self.positivity_preserving_limiter(u_3, prev_state=u_2)
            self.boundaries(u_3)
            k_4 = {n: self.faces[n].solve(*u_3[n], self.time, dt) for n in self.active_face_names}

            for n in self.active_face_names:
                self.faces[n].u = u_3[n][0] + 0.5 * dt * k_4[n][0]
                self.faces[n].v = u_3[n][1] + 0.5 * dt * k_4[n][1]
                self.faces[n].w = u_3[n][2] + 0.5 * dt * k_4[n][2]
                self.faces[n].h = u_3[n][3] + 0.5 * dt * k_4[n][3]
                self.faces[n].hb = u_3[n][4] + 0.5 * dt * k_4[n][4]

            u = {n: self._face_state(n) for n in self.active_face_names}
            # u = self.positivity_preserving_limiter(u, prev_state=u_3)
            self.boundaries(u)
        else:
            raise ValueError(f"order: expected one of [3, 34], found {order}.")

        for n in self.active_face_names:
            self.faces[n].time += dt
        self.time += dt

    def _face_state(self, name):
        face = self.faces[name]
        return face.u, face.v, face.w, face.h, face.hb

    def tracer_variance(self):
        return {n: f.tracer_variance() for n, f in self.faces.items()}

    entropy = tracer_variance

    def energy(self):
        return {n: f.energy() for n, f in self.faces.items()}

    def buoyancy(self):
        return {n: f.hb for n, f in self.faces.items()}

    def mass(self):
        return {n: f.h for n, f in self.faces.items()}

    def save_restart(self, fn_template, directory):
        vars = ["u", "v", "w", "h", "hb"]
        if self.parallel:
            for var in vars:
                self._save_parallel_restart_var(var, fn_template, directory)
            self._restart_barrier()
            return

        for name in self.active_face_names:
            state = self._face_state(name)
            for i, var in enumerate(vars):
                np.save(self.make_fp(var, name, fn_template, directory), _to_numpy(state[i]))

    def load_restart(self, fn_template, directory):
        vars = ["u", "v", "w", "h", "hb"]
        for name in self.active_face_names:
            data = [
                self._load_restart_data(var, name, fn_template, directory)
                for var in vars
            ]
            self.faces[name].set_initial_condition(*data)
        self.boundaries()

    def save_diagnostics(self, fn_template, directory):
        diagnostics = np.stack(
            [
                self.time_list,
                self.energy_list,
                self.variance_list,
                self.buoyancy_list,
                self.mass_list,
            ]
        )
        np.save(os.path.join(directory, f"diagnostics_{fn_template}"), diagnostics)

    def plot_diagnostics(self, fn_template, directory, fig_int, label):
        from matplotlib import pyplot as plt

        diagnostics = np.load(os.path.join(directory, f"diagnostics_{fn_template}"))
        times = diagnostics[0] / (24 * 3600)
        energy = diagnostics[1]
        variance = diagnostics[2]
        buoyancy = diagnostics[3]
        mass = diagnostics[4]

        plt.figure(fig_int, figsize=(7, 7))
        tunit = " (days)"

        ax = plt.subplot(2, 2, 1)
        ax.set_ylabel("Energy")
        ax.set_xticks([], [])
        ax.plot(times, (energy - energy[0]) / energy[0], label=label)
        ax.set_yscale("symlog", linthresh=1e-15)
        ax.grid(True, which="both")

        ax = plt.subplot(2, 2, 2)
        ax.set_ylabel("Mass")
        ax.set_xticks([], [])
        ax.plot(times, (mass - mass[0]) / mass[0], label=label)
        ax.set_yscale("symlog", linthresh=1e-16)
        ax.grid(True, which="both")

        ax = plt.subplot(2, 2, 3)
        ax.set_ylabel("Tracer variance")
        ax.set_xlabel("Time" + tunit)
        ax.plot(times, (variance - variance[0]) / variance[0], label=label)
        ax.set_yscale("symlog", linthresh=1e-15)
        ax.grid(True, which="both")

        ax = plt.subplot(2, 2, 4)
        ax.set_ylabel("Buoyancy")
        ax.set_xlabel("Time" + tunit)
        ax.plot(times, (buoyancy - buoyancy[0]) / buoyancy[0], label=label)
        ax.set_yscale("symlog", linthresh=1e-16)
        ax.grid(True, which="both")

        plt.legend()
        plt.tight_layout()

    def _current_state(self):
        return {
            name: (
                self.faces[name].u,
                self.faces[name].v,
                self.faces[name].w,
                self.faces[name].h,
                self.faces[name].hb,
            )
            for name in self.active_face_names
        }


class DGCubedSphereFaceTSWE(DGCubedSphereFace):
    """One thermal shallow-water cubed-sphere face."""

    def __init__(self, *args, upwind=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.hb = None
        self.upwind = upwind

        self.kx_up, self.kx_down = self.make_up_down_arrays(self.kx)
        self.ky_up, self.ky_down = self.make_up_down_arrays(self.ky)
        self.kz_up, self.kz_down = self.make_up_down_arrays(self.kz)
        self.kx_right, self.kx_left = self.make_left_right_arrays(self.kx)
        self.ky_right, self.ky_left = self.make_left_right_arrays(self.ky)
        self.kz_right, self.kz_left = self.make_left_right_arrays(self.kz)

        self.vert_upper_cov_factor = (
            self.J_vertface[:, :, -1] / self.J_eta[:, :, -1]
        ) / (self.J[:, :, -1] * self.endpoint_weight)
        self.vert_lower_cov_factor = (
            self.J_vertface[:, :, 0] / self.J_eta[:, :, 0]
        ) / (self.J[:, :, 0] * self.endpoint_weight)
        self.horz_right_cov_factor = (
            self.J_horzface[:, :, :, -1] / self.J_xi[:, :, :, -1]
        ) / (self.J[:, :, :, -1] * self.endpoint_weight)
        self.horz_left_cov_factor = (
            self.J_horzface[:, :, :, 0] / self.J_xi[:, :, :, 0]
        ) / (self.J[:, :, :, 0] * self.endpoint_weight)

        self.vert_upper_perp_factor = self.vert_upper_cov_factor / self.J[:, :, -1]
        self.vert_lower_perp_factor = self.vert_lower_cov_factor / self.J[:, :, 0]
        self.horz_right_perp_factor = self.horz_right_cov_factor / self.J[:, :, :, -1]
        self.horz_left_perp_factor = self.horz_left_cov_factor / self.J[:, :, :, 0]

    def boundaries(self, u, v, w, h, hb, t):
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

        self.hb_up[:-1] = hb[:, :, 0, :]
        self.hb_down[1:] = hb[:, :, -1, :]
        self.hb_right[:, :-1] = hb[:, :, :, 0]
        self.hb_left[:, 1:] = hb[:, :, :, -1]

        if self.bc.lower() == "wall":
            self.h_up[-1] = h[-1, :, -1, :]
            self.h_down[0] = h[0, :, 0, :]
            self.h_right[:, -1] = h[:, -1, :, -1]
            self.h_left[:, 0] = h[:, 0, :, 0]
            self.hb_up[-1] = hb[-1, :, -1, :]
            self.hb_down[0] = hb[0, :, 0, :]
            self.hb_right[:, -1] = hb[:, -1, :, -1]
            self.hb_left[:, 0] = hb[:, 0, :, 0]

            u_, v_ = self.phys_to_contra(u, v, w)
            u_, v_, w_ = self.contra_to_phys(u_, 0 * v_)
            self.u_down[0], self.v_down[0], self.w_down[0] = u_[0, :, 0, :], v_[0, :, 0, :], w_[0, :, 0, :]
            self.u_up[-1], self.v_up[-1], self.w_up[-1] = u_[-1, :, -1, :], v_[-1, :, -1, :], w_[-1, :, -1, :]

            u_, v_ = self.phys_to_contra(u, v, w)
            u_, v_, w_ = self.contra_to_phys(0 * u_, v_)
            self.u_left[:, 0], self.v_left[:, 0], self.w_left[:, 0] = u_[:, 0, :, 0], v_[:, 0, :, 0], w_[:, 0, :, 0]
            self.u_right[:, -1], self.v_right[:, -1], self.w_right[:, -1] = u_[:, -1, :, -1], v_[:, -1, :, -1], w_[:, -1, :, -1]

    def set_initial_condition(self, u, v, w, h, hb):
        self.u = _to_numpy(u, dtype=self.dtype, copy=True)
        self.v = _to_numpy(v, dtype=self.dtype, copy=True)
        self.w = _to_numpy(w, dtype=self.dtype, copy=True)
        self.h = _to_numpy(h, dtype=self.dtype, copy=True)

        self.hb = _to_numpy(hb, dtype=self.dtype, copy=True)

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

        self.hb_left = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.hb_right = np.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype)
        self.hb_up = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)
        self.hb_down = np.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype)

        self.boundaries(self.u, self.v, self.w, self.h, self.hb, 0)

    def get_dt(self):
        speed = self.wave_speed(self.u, self.v, self.w, self.h, self.hb)
        return self.cdt / np.max(speed)

    def hflux(self, u, v, w, h):
        return u * h, v * h, w * h

    def hbflux(self, u, v, w, hb):
        return u * hb, v * hb, w * hb

    def uv_flux(self, u, v, w, h, hb):
        return 0.5 * (u ** 2 + v ** 2 + w ** 2) + 0.5 * hb

    def wave_speed(self, u, v, w, h, hb):
        return np.sqrt(u ** 2 + v ** 2 + w ** 2) + np.sqrt(hb)

    def solve(self, u, v, w, h, hb, t, dt, *, verbose=False):
        if _solve_tswe_numba_kernel is None:
            return self.solve_numpy(u, v, w, h, hb, t, dt, verbose=verbose)

        self.boundaries(u, v, w, h, hb, t)
        return _solve_tswe_numba_kernel(
            u, v, w, h, hb,
            self.D, self.endpoint_weight, self.J,
            self.vert_upper_edge_factor, self.vert_lower_edge_factor,
            self.horz_right_edge_factor, self.horz_left_edge_factor,
            self.vert_upper_cov_factor, self.vert_lower_cov_factor,
            self.horz_right_cov_factor, self.horz_left_cov_factor,
            self.vert_upper_perp_factor, self.vert_lower_perp_factor,
            self.horz_right_perp_factor, self.horz_left_perp_factor,
            self.dxidx, self.dxidy, self.dxidz, self.detadx, self.detady, self.detadz,
            self.dxdxi, self.dydxi, self.dzdxi, self.dxdeta, self.dydeta, self.dzdeta,
            self.kx, self.ky, self.kz, self.f,
            self.u_up, self.v_up, self.w_up, self.h_up, self.hb_up,
            self.u_down, self.v_down, self.w_down, self.h_down, self.hb_down,
            self.u_right, self.v_right, self.w_right, self.h_right, self.hb_right,
            self.u_left, self.v_left, self.w_left, self.h_left, self.hb_left,
            self.eta_x_up, self.eta_y_up, self.eta_z_up,
            self.eta_x_down, self.eta_y_down, self.eta_z_down,
            self.xi_x_right, self.xi_y_right, self.xi_z_right,
            self.xi_x_left, self.xi_y_left, self.xi_z_left,
            self.dxdxi_up, self.dydxi_up, self.dzdxi_up,
            self.dxdxi_down, self.dydxi_down, self.dzdxi_down,
            self.dxdxi_right, self.dydxi_right, self.dzdxi_right,
            self.dxdxi_left, self.dydxi_left, self.dzdxi_left,
            self.dxdeta_up, self.dydeta_up, self.dzdeta_up,
            self.dxdeta_down, self.dydeta_down, self.dzdeta_down,
            self.dxdeta_right, self.dydeta_right, self.dzdeta_right,
            self.dxdeta_left, self.dydeta_left, self.dzdeta_left,
            self.kx_up, self.ky_up, self.kz_up,
            self.kx_down, self.ky_down, self.kz_down,
            self.kx_right, self.ky_right, self.kz_right,
            self.kx_left, self.ky_left, self.kz_left,
            self.g, self.a, self.upwind,
        )

    def solve_numpy(self, u, v, w, h, hb, t, dt, *, verbose=False):
        self.boundaries(u, v, w, h, hb, t)

        b = hb / h

        h_xflux, h_yflux, h_zflux = self.hflux(u, v, w, h)
        h_xflux, h_yflux = self.phys_to_contra(h_xflux, h_yflux, h_zflux)
        div = (self.ddxi(h_xflux * self.J) + self.ddeta(h_yflux * self.J)) / self.J
        h_k = -div

        hb_xflux, hb_yflux, hb_zflux = self.hbflux(u, v, w, hb)
        hb_xflux, hb_yflux = self.phys_to_contra(hb_xflux, hb_yflux, hb_zflux)
        bdiv = (self.ddxi(hb_xflux * self.J) + self.ddeta(hb_yflux * self.J)) / self.J
        dbdxi = self.ddxi(b)
        dbdeta = self.ddeta(b)
        dhbdxi = self.ddxi(hb)
        dhbdeta = self.ddeta(hb)
        dhdxi = self.ddxi(h)
        dhdeta = self.ddeta(h)
        hb_k = -0.5 * (bdiv + b * div + dbdxi * h_xflux + dbdeta * h_yflux)

        h_up_flux = self._normal_h_flux(self.u_up, self.v_up, self.w_up, self.h_up, self.eta_x_up, self.eta_y_up, self.eta_z_up)
        h_down_flux = self._normal_h_flux(self.u_down, self.v_down, self.w_down, self.h_down, self.eta_x_down, self.eta_y_down, self.eta_z_down)
        h_right_flux = self._normal_h_flux(self.u_right, self.v_right, self.w_right, self.h_right, self.xi_x_right, self.xi_y_right, self.xi_z_right)
        h_left_flux = self._normal_h_flux(self.u_left, self.v_left, self.w_left, self.h_left, self.xi_x_left, self.xi_y_left, self.xi_z_left)

        hb_up_flux = self._normal_h_flux(self.u_up, self.v_up, self.w_up, self.hb_up, self.eta_x_up, self.eta_y_up, self.eta_z_up)
        hb_down_flux = self._normal_h_flux(self.u_down, self.v_down, self.w_down, self.hb_down, self.eta_x_down, self.eta_y_down, self.eta_z_down)
        hb_right_flux = self._normal_h_flux(self.u_right, self.v_right, self.w_right, self.hb_right, self.xi_x_right, self.xi_y_right, self.xi_z_right)
        hb_left_flux = self._normal_h_flux(self.u_left, self.v_left, self.w_left, self.hb_left, self.xi_x_left, self.xi_y_left, self.xi_z_left)

        h_flux_vert = 0.5 * (h_up_flux + h_down_flux)
        h_flux_horz = 0.5 * (h_right_flux + h_left_flux)

        b_up = self.hb_up / self.h_up
        b_down = self.hb_down / self.h_down
        b_right = self.hb_right / self.h_right
        b_left = self.hb_left / self.h_left

        if self.upwind:
            b_hat_ve = np.where(h_flux_vert >= 0.0, b_down, b_up)
            b_hat_ho = np.where(h_flux_horz >= 0.0, b_left, b_right)
        else:
            b_hat_ve = 0.5 * (b_up + b_down)
            b_hat_ho = 0.5 * (b_right + b_left)

        hb_flux_vert = b_hat_ve * h_flux_vert
        hb_flux_horz = b_hat_ho * h_flux_horz

        h_k[:, :, -1] -= (h_flux_vert[1:] - h_down_flux[1:]) * self.vert_upper_edge_factor
        h_k[:, :, 0] += (h_flux_vert[:-1] - h_up_flux[:-1]) * self.vert_lower_edge_factor
        h_k[:, :, :, -1] -= (h_flux_horz[:, 1:] - h_left_flux[:, 1:]) * self.horz_right_edge_factor
        h_k[:, :, :, 0] += (h_flux_horz[:, :-1] - h_right_flux[:, :-1]) * self.horz_left_edge_factor

        hb_k[:, :, -1] -= (hb_flux_vert[1:] - hb_down_flux[1:]) * self.vert_upper_edge_factor
        hb_k[:, :, 0] += (hb_flux_vert[:-1] - hb_up_flux[:-1]) * self.vert_lower_edge_factor
        hb_k[:, :, :, -1] -= (hb_flux_horz[:, 1:] - hb_left_flux[:, 1:]) * self.horz_right_edge_factor
        hb_k[:, :, :, 0] += (hb_flux_horz[:, :-1] - hb_right_flux[:, :-1]) * self.horz_left_edge_factor

        uv_flux = self.uv_flux(u, v, w, h, hb)
        u_contra, v_contra = self.phys_to_contra(u, v, w)
        u_cov, v_cov, _ = self.phys_to_cov(u, v, w)
        vort = (self.ddxi(v_cov) - self.ddeta(u_cov)) / self.J + self.f

        velocity_perp = cross_product([self.kx, self.ky, self.kz], [u, v, w])
        u_perp, v_perp, _ = self.phys_to_cov(*velocity_perp)

        u_k_cov = -self.ddxi(uv_flux) - vort * u_perp
        u_k_cov -= 0.25 * (b * dhdxi + dhbdxi - h * dbdxi)
        v_k_cov = -self.ddeta(uv_flux) - vort * v_perp
        v_k_cov -= 0.25 * (b * dhdeta + dhbdeta - h * dbdeta)

        uv_up_flux = self.uv_flux(self.u_up, self.v_up, self.w_up, self.h_up, self.hb_up)
        uv_down_flux = self.uv_flux(self.u_down, self.v_down, self.w_down, self.h_down, self.hb_down)
        uv_right_flux = self.uv_flux(self.u_right, self.v_right, self.w_right, self.h_right, self.hb_right)
        uv_left_flux = self.uv_flux(self.u_left, self.v_left, self.w_left, self.h_left, self.hb_left)

        c_ve = 0.5 * (
            self.wave_speed(self.u_up, self.v_up, self.w_up, self.h_up, self.hb_up)
            + self.wave_speed(self.u_down, self.v_down, self.w_down, self.h_down, self.hb_down)
        )
        c_ho = 0.5 * (
            self.wave_speed(self.u_right, self.v_right, self.w_right, self.h_right, self.hb_right)
            + self.wave_speed(self.u_left, self.v_left, self.w_left, self.h_left, self.hb_left)
        )
        uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux)
        uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux)
        if self.a != 0.0:
            uv_flux_vert -= self.a * (self.g / c_ve) * (h_up_flux - h_down_flux)
            uv_flux_horz -= self.a * (self.g / c_ho) * (h_right_flux - h_left_flux)

        u_k_cov[:, :, :, -1] -= (uv_flux_horz[:, 1:] - uv_left_flux[:, 1:]) * self.horz_right_cov_factor
        u_k_cov[:, :, :, 0] += (uv_flux_horz[:, :-1] - uv_right_flux[:, :-1]) * self.horz_left_cov_factor
        u_k_cov[:, :, :, -1] -= 0.25 * b_hat_ho[:, 1:] * (self.h_right[:, 1:] - self.h_left[:, 1:]) * self.horz_right_cov_factor
        u_k_cov[:, :, :, 0] += 0.25 * b_hat_ho[:, :-1] * (self.h_left[:, :-1] - self.h_right[:, :-1]) * self.horz_left_cov_factor

        v_k_cov[:, :, -1] -= (uv_flux_vert[1:] - uv_down_flux[1:]) * self.vert_upper_cov_factor
        v_k_cov[:, :, 0] += (uv_flux_vert[:-1] - uv_up_flux[:-1]) * self.vert_lower_cov_factor
        v_k_cov[:, :, -1] -= 0.25 * b_hat_ve[1:] * (self.h_up[1:] - self.h_down[1:]) * self.vert_upper_cov_factor
        v_k_cov[:, :, 0] += 0.25 * b_hat_ve[:-1] * (self.h_down[:-1] - self.h_up[:-1]) * self.vert_lower_cov_factor

        # tangent work

        u_cov_up = self.u_up * self.dxdxi_up + self.v_up * self.dydxi_up + self.w_up * self.dzdxi_up
        u_cov_down = self.u_down * self.dxdxi_down + self.v_down * self.dydxi_down + self.w_down * self.dzdxi_down
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

        # u_cov_up = self.u_up * self.dxdxi_up + self.v_up * self.dydxi_up + self.w_up * self.dzdxi_up
        # u_cov_down = self.u_down * self.dxdxi_down + self.v_down * self.dydxi_down + self.w_down * self.dzdxi_down
        # v_cov_right = self.u_right * self.dxdeta_right + self.v_right * self.dydeta_right + self.w_right * self.dzdeta_right
        # v_cov_left = self.u_left * self.dxdeta_left + self.v_left * self.dydeta_left + self.w_left * self.dzdeta_left
        #
        # u_perp_up, v_perp_up = self._edge_perp_cov(self.kx_up, self.ky_up, self.kz_up, self.u_up, self.v_up, self.w_up, self.dxdxi_up, self.dydxi_up, self.dzdxi_up, self.dxdeta_up, self.dydeta_up, self.dzdeta_up)
        # u_perp_down, v_perp_down = self._edge_perp_cov(self.kx_down, self.ky_down, self.kz_down, self.u_down, self.v_down, self.w_down, self.dxdxi_down, self.dydxi_down, self.dzdxi_down, self.dxdeta_down, self.dydeta_down, self.dzdeta_down)
        # u_perp_right, v_perp_right = self._edge_perp_cov(self.kx_right, self.ky_right, self.kz_right, self.u_right, self.v_right, self.w_right, self.dxdxi_right, self.dydxi_right, self.dzdxi_right, self.dxdeta_right, self.dydeta_right, self.dzdeta_right)
        # u_perp_left, v_perp_left = self._edge_perp_cov(self.kx_left, self.ky_left, self.kz_left, self.u_left, self.v_left, self.w_left, self.dxdxi_left, self.dydxi_left, self.dzdxi_left, self.dxdeta_left, self.dydeta_left, self.dzdeta_left)
        #
        # u_cov_jump = u_cov_up - u_cov_down
        # v_cov_jump = v_cov_right - v_cov_left
        #
        # u_k_cov[:, :, -1] += 0.5 * u_perp_down[1:] * u_cov_jump[1:] * self.vert_upper_perp_factor
        # u_k_cov[:, :, 0] += 0.5 * u_perp_up[:-1] * u_cov_jump[:-1] * self.vert_lower_perp_factor
        #
        # u_k_cov[:, :, :, -1] -= 0.5 * u_perp_left[:, 1:] * v_cov_jump[:, 1:] * self.horz_right_perp_factor
        # u_k_cov[:, :, :, 0] -= 0.5 * u_perp_right[:, :-1] * v_cov_jump[:, :-1] * self.horz_left_perp_factor
        #
        # v_k_cov[:, :, -1] += 0.5 * v_perp_down[1:] * u_cov_jump[1:] * self.vert_upper_perp_factor
        # v_k_cov[:, :, 0] += 0.5 * v_perp_up[:-1] * u_cov_jump[:-1] * self.vert_lower_perp_factor
        # v_k_cov[:, :, :, -1] -= 0.5 * v_perp_left[:, 1:] * v_cov_jump[:, 1:] * self.horz_right_perp_factor
        # v_k_cov[:, :, :, 0] -= 0.5 * v_perp_right[:, :-1] * v_cov_jump[:, :-1] * self.horz_left_perp_factor

        u_k, v_k, w_k = self.cov_to_phys(u_k_cov, v_k_cov, 0)
        return u_k, v_k, w_k, h_k, hb_k

    @staticmethod
    def _normal_h_flux(u, v, w, h, nx, ny, nz):
        return h * (u * nx + v * ny + w * nz)

    @staticmethod
    def _edge_perp_cov(kx, ky, kz, u, v, w, dxdxi, dydxi, dzdxi, dxdeta, dydeta, dzdeta):
        px = ky * w - kz * v
        py = kz * u - kx * w
        pz = kx * v - ky * u
        u_perp = px * dxdxi + py * dydxi + pz * dzdxi
        v_perp = px * dxdeta + py * dydeta + pz * dzdeta
        return u_perp, v_perp

    def tracer_variance(self, u=None, v=None, w=None, h=None, hb=None):
        if h is None:
            h = self.h
        if hb is None:
            hb = self.hb
        b = hb / h
        return 0.5 * h * b ** 2

    entropy = tracer_variance

    def energy(self, u=None, v=None, w=None, h=None, hb=None):
        if u is None:
            u = self.u
        if v is None:
            v = self.v
        if w is None:
            w = self.w
        if h is None:
            h = self.h
        if hb is None:
            hb = self.hb
        return 0.5 * h * (u ** 2 + v ** 2 + w ** 2 + hb)

    def dEdt(self):
        u, v, w, h, hb = self.u, self.v, self.w, self.h, self.hb
        dudt, dvdt, dwdt, dhdt, dhbdt = self.solve_numpy(u, v, w, h, hb, 0, 0)
        dEdt = h * (u * dudt + v * dvdt + w * dwdt)
        dEdt += (0.5 * (u ** 2 + v ** 2 + w ** 2) + 0.5 * hb) * dhdt
        dEdt += 0.5 * h * dhbdt
        return dEdt


__all__ = ["DGCubedSphereTSWE", "DGCubedSphereFaceTSWE"]
