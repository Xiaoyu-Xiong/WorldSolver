"""Physically-based elastic solid solver for the suspended elastic bunny pendulum."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized

from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self) -> None:
        self.time: float = 0.0
        self.positions: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.velocities: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.masses: np.ndarray = np.empty(0, dtype=np.float64)
        self.fixed_mask: np.ndarray = np.empty(0, dtype=bool)
        self.initial_positions: np.ndarray = np.empty((0, 3), dtype=np.float64)
        self.faces: np.ndarray = np.empty((0, 3), dtype=np.int32)
        self.gravity: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=np.float64)

        # Cumulative impulse tracking
        self.cumulative_external_impulse: np.ndarray = np.zeros(3, dtype=np.float64)
        self.reference_momentum: float = 1.0

        # Simulation substepping parameters
        self.sub_dt: float = 0.002
        self.inv_Dm: np.ndarray = np.empty((0, 2, 2), dtype=np.float64)

        # Linear FEM system
        self.free_indices: np.ndarray = np.empty(0, dtype=np.int32)
        self.solver_step = None
        self.K_free = None
        self.M_free_diag = None
        self.rayleigh_alpha: float = 0.02
        self.rayleigh_beta: float = 0.005

    def _load_faces(self, path: Path) -> np.ndarray:
        faces: list[list[int]] = []
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("f "):
                    faces.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
        return np.asarray(faces, dtype=np.int32)

    def initialize(self, scene, seed: int) -> None:
        initial_frame = scene.initial_state()
        self.time = float(initial_frame.time)
        self.positions = np.array(initial_frame.position, dtype=np.float64, copy=True)
        self.velocities = np.array(initial_frame.velocity, dtype=np.float64, copy=True)
        self.masses = np.array(initial_frame.mass, dtype=np.float64, copy=True)
        self.initial_positions = np.array(initial_frame.position, dtype=np.float64, copy=True)

        fixed_attr = initial_frame.attributes.get("fixed", np.zeros(len(self.positions)))
        self.fixed_mask = np.asarray(fixed_attr, dtype=bool)

        bunny_path = Path(__file__).resolve().parent.parent / "assets" / "bunny.obj"
        self.faces = self._load_faces(bunny_path)

        meta = scene.metadata() if hasattr(scene, "metadata") else {}
        self.gravity = np.array(meta.get("gravity", [0.0, 0.0, -9.81]), dtype=np.float64)

        total_mass = np.sum(self.masses)
        # Characteristic pendulum velocity scale: v ~ sqrt(2 * g * L), L ~ 1.5m -> v ~ 5.4 m/s
        self.reference_momentum = float(total_mass * np.sqrt(2.0 * 9.81 * 1.5))
        if self.reference_momentum <= 0:
            self.reference_momentum = 1.0

        self.cumulative_external_impulse = np.zeros(3, dtype=np.float64)

        # Precompute face material coordinate inverse basis for 2D manifold deformation gradient
        # Reference coordinates in local face tangent plane
        num_faces = len(self.faces)
        self.inv_Dm = np.zeros((num_faces, 2, 2), dtype=np.float64)
        self.face_normals_0 = np.zeros((num_faces, 3), dtype=np.float64)
        self.e1_0 = np.zeros((num_faces, 3), dtype=np.float64)
        self.e2_0 = np.zeros((num_faces, 3), dtype=np.float64)

        X = self.initial_positions
        for e_idx, f in enumerate(self.faces):
            x0, x1, x2 = X[f[0]], X[f[1]], X[f[2]]
            d1 = x1 - x0
            d2 = x2 - x0
            n = np.cross(d1, d2)
            n_len = np.linalg.norm(n)
            if n_len > 1e-12:
                n = n / n_len
            else:
                n = np.array([0.0, 0.0, 1.0])
            self.face_normals_0[e_idx] = n

            # Local 2D orthogonal basis in reference configuration
            d1_len = np.linalg.norm(d1)
            if d1_len > 1e-12:
                e1 = d1 / d1_len
            else:
                e1 = np.array([1.0, 0.0, 0.0])
            e2 = np.cross(n, e1)
            e2_len = np.linalg.norm(e2)
            if e2_len > 1e-12:
                e2 = e2 / e2_len
            else:
                e2 = np.array([0.0, 1.0, 0.0])

            self.e1_0[e_idx] = e1
            self.e2_0[e_idx] = e2

            # Dm in local 2D material frame
            Dm = np.array([
                [np.dot(d1, e1), np.dot(d2, e1)],
                [np.dot(d1, e2), np.dot(d2, e2)]
            ], dtype=np.float64)
            det = Dm[0, 0] * Dm[1, 1] - Dm[0, 1] * Dm[1, 0]
            if abs(det) > 1e-12:
                self.inv_Dm[e_idx] = np.linalg.inv(Dm)
            else:
                self.inv_Dm[e_idx] = np.eye(2)

        # Build elastic stiffness matrix K for surface linear elasticity / spring network
        # Material parameters
        youngs = 30000.0  # Pa
        poisson = 0.45
        # Effective 2D shell / surface elasticity
        mu = youngs / (2.0 * (1.0 + poisson))
        lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))

        n_verts = len(self.positions)
        I_list = []
        J_list = []
        V_list = []

        # Compute element stiffness for all triangular faces
        for e_idx, f in enumerate(self.faces):
            x0, x1, x2 = X[f[0]], X[f[1]], X[f[2]]
            d1 = x1 - x0
            d2 = x2 - x0
            area = 0.5 * np.linalg.norm(np.cross(d1, d2))
            if area < 1e-12:
                continue

            invD = self.inv_Dm[e_idx]
            # Shape function gradients in local coordinates:
            # dN/dX = [-invD[0,0]-invD[1,0], invD[0,0], invD[1,0]]
            # dN/dY = [-invD[0,1]-invD[1,1], invD[0,1], invD[1,1]]
            beta = np.zeros((3, 2), dtype=np.float64)
            beta[1, 0] = invD[0, 0]
            beta[1, 1] = invD[0, 1]
            beta[2, 0] = invD[1, 0]
            beta[2, 1] = invD[1, 1]
            beta[0, 0] = -invD[0, 0] - invD[1, 0]
            beta[0, 1] = -invD[0, 1] - invD[1, 1]

            # Plane stress / strain constitutive matrix C (2D Voigt: e_xx, e_yy, 2*e_xy)
            # Plane stress formulation for 2D surface elements:
            factor = youngs / (1.0 - poisson**2)
            C = factor * np.array([
                [1.0, poisson, 0.0],
                [poisson, 1.0, 0.0],
                [0.0, 0.0, 0.5 * (1.0 - poisson)]
            ], dtype=np.float64)

            # For 3D displacements of 3 nodes (9 DOF), local strain in tangent plane:
            # Let transformation to local tangent coordinates be Q = [e1, e2, n]^T (3x3)
            e1 = self.e1_0[e_idx]
            e2 = self.e2_0[e_idx]

            # B matrix: 3 x 9 (Voigt strain in tangent frame from 9 global nodal displacements)
            B = np.zeros((3, 9), dtype=np.float64)
            for i in range(3):
                b_ix = beta[i, 0]
                b_iy = beta[i, 1]
                # displacement u_i in global coordinates
                # u_local_x = e1 . u_i, u_local_y = e2 . u_i
                # eps_xx = b_ix * u_local_x = b_ix * (e1 . u_i)
                B[0, 3*i : 3*i+3] = b_ix * e1
                # eps_yy = b_iy * u_local_y = b_iy * (e2 . u_i)
                B[1, 3*i : 3*i+3] = b_iy * e2
                # 2*eps_xy = b_iy * u_local_x + b_ix * u_local_y = b_iy * e1 + b_ix * e2
                B[2, 3*i : 3*i+3] = b_iy * e1 + b_ix * e2

            # Element stiffness K_elem = area * thickness * B^T C B (assume thickness ~ 0.05m)
            thickness = 0.06
            Ke = area * thickness * (B.T @ C @ B)

            # Out-of-plane bending / transverse stiffness regularization
            # Add small Laplacian stiffness for non-planar modes
            k_bend = 0.05 * youngs * area * thickness
            for i in range(3):
                for j in range(3):
                    val = k_bend if i == j else -0.5 * k_bend
                    Ke[3*i:3*i+3, 3*j:3*j+3] += val * np.eye(3)

            # Assemble into global sparse matrix
            for i in range(3):
                ni = f[i]
                for j in range(3):
                    nj = f[j]
                    sub_K = Ke[3*i:3*i+3, 3*j:3*j+3]
                    for di in range(3):
                        for dj in range(3):
                            I_list.append(3 * ni + di)
                            J_list.append(3 * nj + dj)
                            V_list.append(sub_K[di, dj])

        K_global = sparse.coo_matrix((V_list, (I_list, J_list)), shape=(3 * n_verts, 3 * n_verts)).tocsc()

        # Free DOFs (unfixed vertices)
        self.free_dofs = np.where(~np.repeat(self.fixed_mask, 3))[0]
        self.free_verts = np.where(~self.fixed_mask)[0]

        # Mass matrix
        M_diag = np.repeat(self.masses, 3)
        self.M_free_diag = M_diag[self.free_dofs]

        self.K_free = K_global[self.free_dofs, :][:, self.free_dofs].tocsc()

        # Implicit Newmark / Backward Euler system matrix:
        # (M + dt * D + dt^2 * K) Delta_v = dt * (f_ext - K (u_n - u_0) - D v_n)
        # Rayleigh damping D = alpha * M + beta * K
        # With sub_dt = 0.002:
        dt = self.sub_dt
        alpha = self.rayleigh_alpha
        beta = self.rayleigh_beta

        M_sp = sparse.diags(self.M_free_diag, 0, format="csc")
        D_free = alpha * M_sp + beta * self.K_free

        # Effective system matrix for implicit velocity step:
        # (M + dt * D + dt^2 * K) v_{n+1} = M v_n + dt * f_ext - dt * K (u_n - u_0)
        A = M_sp + dt * D_free + (dt**2) * self.K_free
        self.solver_step = factorized(A.tocsc())
        self.D_free = D_free

    def _physics_step(self, dt: float) -> None:
        # Current displacements from reference
        u_free = (self.positions - self.initial_positions).reshape(-1)[self.free_dofs]
        v_free = self.velocities.reshape(-1)[self.free_dofs]

        # External force on free DOFs (gravity)
        # M_free_diag * g
        g_free = np.tile(self.gravity, len(self.free_verts))
        f_ext = self.M_free_diag * g_free

        # Elastic internal force: -K * u_free
        f_int = -self.K_free.dot(u_free)

        # Right hand side for implicit step:
        # M v_n + dt * f_ext + dt * f_int
        rhs = self.M_free_diag * v_free + dt * f_ext + dt * f_int

        # Solve for new velocity
        v_next = self.solver_step(rhs)

        # Update free velocity and position
        self.velocities.reshape(-1)[self.free_dofs] = v_next
        self.positions.reshape(-1)[self.free_dofs] += dt * v_next

        # Fixed vertices remain strictly stationary at initial position
        self.positions[self.fixed_mask] = self.initial_positions[self.fixed_mask]
        self.velocities[self.fixed_mask] = 0.0

        # External impulse on system from gravity: dt * total_mass * g
        # plus constraint reaction forces
        total_mass = np.sum(self.masses)
        self.cumulative_external_impulse += dt * total_mass * self.gravity

    def advance_to(self, target_time: float) -> StateFrame:
        # Integrate forward in time using sub-steps
        while self.time + 1e-9 < target_time:
            dt = min(self.sub_dt, target_time - self.time)
            if dt <= 1e-12:
                break
            self._physics_step(dt)
            self.time += dt

        self.time = target_time

        # Ensure fixed vertices stay pinned exactly
        self.positions[self.fixed_mask] = self.initial_positions[self.fixed_mask]
        self.velocities[self.fixed_mask] = 0.0

        # Compute deformation gradient per face element
        num_faces = len(self.faces)
        F_elements = np.zeros((num_faces, 3, 3), dtype=np.float64)

        for e_idx, f in enumerate(self.faces):
            x0, x1, x2 = self.positions[f[0]], self.positions[f[1]], self.positions[f[2]]
            d1 = x1 - x0
            d2 = x2 - x0

            # Current face normal
            n = np.cross(d1, d2)
            n_len = np.linalg.norm(n)
            if n_len > 1e-12:
                n = n / n_len
            else:
                n = self.face_normals_0[e_idx]

            # Current local deformation: Ds = [d1, d2, n]
            # In material coordinates (local tangent frame of reference element):
            # The reference basis is [e1_0, e2_0, n_0]
            # Ds = [d1, d2, n] mapped from [Dm, n_0]
            invD = self.inv_Dm[e_idx]

            # 3D spatial vectors d1, d2 in global coordinates
            # F_2d = [d1, d2] * inv(Dm) (3x2 matrix)
            F_2d = np.column_stack((d1, d2)) @ invD

            # Full 3D deformation gradient: F = [F_2d[:, 0], F_2d[:, 1], n]
            F = np.column_stack((F_2d[:, 0], F_2d[:, 1], n))
            F_elements[e_idx] = F

        N = len(self.positions)
        G = 1  # 1 budget domain (global)
        K = 1  # 1 force term

        # Total external force on domain: gravity on entire mass
        total_external_force = np.sum(self.masses[:, None] * self.gravity[None, :], axis=0)
        budget_ext_force = np.zeros((G, K, 3), dtype=np.float64)
        budget_ext_force[0, 0] = total_external_force

        budget_weights = np.ones((G, N), dtype=np.float64)
        budget_cum_impulse = np.zeros((G, 3), dtype=np.float64)
        budget_cum_impulse[0] = self.cumulative_external_impulse

        frame = StateFrame(
            time=self.time,
            position=self.positions.copy(),
            velocity=self.velocities.copy(),
            mass=self.masses.copy(),
            attributes={
                "fixed": self.fixed_mask.astype(np.float64),
                "mass_quantity": self.masses.copy(),
            },
            fields={
                "budget_cumulative_external_impulse": budget_cum_impulse,
                "budget_cumulative_inflow_momentum": np.zeros((G, 3), dtype=np.float64),
                "budget_cumulative_outflow_momentum": np.zeros((G, 3), dtype=np.float64),
                "budget_domain_weights": budget_weights,
                "budget_external_force": budget_ext_force,
                "cumulative_mass_source": np.zeros(K, dtype=np.float64),
                "element_deformation_gradient": F_elements,
            },
            scalars={
                "reference_momentum": self.reference_momentum,
                "density_kg_m3": 1000.0,
            },
        )
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "time": self.time,
            "num_particles": len(self.positions),
            "num_faces": len(self.faces),
        }

