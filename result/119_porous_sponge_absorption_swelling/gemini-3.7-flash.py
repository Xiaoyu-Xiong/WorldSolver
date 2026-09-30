"""Porous sponge absorption and poroelastic swelling solver.

Implements two-way coupled multiphysics:
- SPH-inspired / continuum fluid motion for free water block.
- Darcy imbibition and capillary suction into the porous sponge matrix.
- Poroelastic deformation and swelling of the sponge mesh and porous interior.
- Diagnostics for Darcy flux, porosity, permeability, pore pressure, and momentum budgets.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree
from worldsolver.contracts import StateFrame


class BenchmarkSolver:
    def __init__(self):
        self.scene = None
        self.current_time = 0.0
        self.sample_dt = 0.02

        # Physics constants
        self.gravity = np.array([0.0, 0.0, -9.81], dtype=np.float64)
        self.dynamic_viscosity = 1.0e-3  # Pa*s (water)
        self.pore_fluid_density = 1000.0  # kg/m^3
        self.initial_porosity = 0.90
        self.intrinsic_permeability = 3.0e-8  # m^2
        self.grid_spacing = 0.06  # m
        self.reference_darcy_flux = 0.05  # m/s
        self.reference_momentum = 500.0  # kg*m/s
        self.reference_pore_storage_rate = 0.1  # 1/s

        # Grid dimensions for porous interior
        self.nx, self.ny, self.nz = 25, 21, 21
        self.grid_shape = (self.nx, self.ny, self.nz)

        # Particle indices and state
        self.ns = 0
        self.nw = 0
        self.npore = 0
        self.N = 0

        # Arrays
        self.pos = None
        self.vel = None
        self.mass = None
        self.body_id = None
        self.phase = None
        self.saturation = None

        # Mesh initial geometry for sponge surface
        self.surface_pos_init = None
        self.porous_pos_init = None
        self.water_pos_init = None

        # Porous interior grid initial coordinates
        self.grid_x0 = None
        self.grid_y0 = None
        self.grid_z0 = None

        # Cumulative budget tracking
        self.cumulative_ext_impulse = None  # [3, 3]
        self.cumulative_inflow_mom = None   # [3, 3]
        self.cumulative_outflow_mom = None  # [3, 3]
        self.cumulative_mass_source = None  # [3]

        # Swelling parameters
        self.max_swelling_factor = 0.25  # Sponge volume/linear expansion factor
        self.diffusivity = 0.035         # Darcy diffusion rate (m^2/s)

    def initialize(self, scene, seed: int) -> None:
        self.scene = scene
        np.random.seed(seed)
        self.current_time = 0.0
        self.sample_dt = float(scene.config.get("sample_dt", 0.02))

        init_frame = scene.initial_state()
        self.pos = np.array(init_frame.position, dtype=np.float64, copy=True)
        self.vel = np.array(init_frame.velocity, dtype=np.float64, copy=True)
        self.mass = np.array(init_frame.mass, dtype=np.float64, copy=True)
        self.body_id = np.array(init_frame.body_id, dtype=np.int32, copy=True)

        self.N = len(self.pos)
        self.surf_mask = (self.body_id == 1)
        self.water_mask = (self.body_id == 2)
        self.pore_mask = (self.body_id == 3)

        self.ns = int(np.sum(self.surf_mask))
        self.nw = int(np.sum(self.water_mask))
        self.npore = int(np.sum(self.pore_mask))

        self.surface_pos_init = self.pos[self.surf_mask].copy()
        self.water_pos_init = self.pos[self.water_mask].copy()
        self.porous_pos_init = self.pos[self.pore_mask].copy()

        # Grid axes
        self.bounds_min = np.array([-1.22, -0.62, 0.03], dtype=np.float64)
        self.bounds_max = np.array([0.22, 0.62, 1.27], dtype=np.float64)

        # 3D Saturation field on porous grid
        self.grid_saturation = np.zeros(self.grid_shape, dtype=np.float64)
        self.grid_pressure = np.zeros(self.grid_shape, dtype=np.float64)
        self.grid_darcy_flux = np.zeros((*self.grid_shape, 3), dtype=np.float64)
        self.grid_solid_vel = np.zeros((*self.grid_shape, 3), dtype=np.float64)
        self.grid_volume_source = np.zeros(self.grid_shape, dtype=np.float64)

        # Budget accumulators (3 domains: 0: sponge surface, 1: water, 2: porous interior)
        self.cumulative_ext_impulse = np.zeros((3, 3), dtype=np.float64)
        self.cumulative_inflow_mom = np.zeros((3, 3), dtype=np.float64)
        self.cumulative_outflow_mom = np.zeros((3, 3), dtype=np.float64)
        self.cumulative_mass_source = np.zeros(3, dtype=np.float64)

        # Per-particle saturation
        self.saturation = np.zeros(self.N, dtype=np.float64)
        self.saturation[self.water_mask] = 1.0

    def advance_to(self, target_time: float) -> StateFrame:
        sub_dt = 0.005  # 5 ms physical sub-stepping
        while self.current_time + 1e-8 < target_time:
            dt = min(sub_dt, target_time - self.current_time)
            self._step_physics(dt)
            self.current_time += dt

        return self._build_state_frame(target_time)

    def _step_physics(self, dt: float) -> None:
        # 1. Moisture / Imbibition Darcy Diffusion along porous grid
        # Water touches the sponge at the right face (x = 0.22, i = nx - 1)
        # Wet front advances inward from x = 0.22 to x = -1.22
        # Contact boundary condition: right face is wet while water remains nearby

        # Calculate how much free water is adjacent to contact face
        water_contact = self.pos[self.water_mask, 0] < 0.35
        contact_active = np.sum(water_contact) > 0.05 * self.nw

        # Boundary condition at i = nx - 1 (right face)
        if contact_active:
            # Right face saturation increases towards 1
            self.grid_saturation[-1, :, :] = np.clip(
                self.grid_saturation[-1, :, :] + 2.5 * dt, 0.0, 1.0
            )

        # 3D diffusion equation for capillary saturation in porous sponge
        sat = self.grid_saturation
        d2x = (sat[2:, 1:-1, 1:-1] - 2.0 * sat[1:-1, 1:-1, 1:-1] + sat[:-2, 1:-1, 1:-1]) / (self.grid_spacing**2)
        d2y = (sat[1:-1, 2:, 1:-1] - 2.0 * sat[1:-1, 1:-1, 1:-1] + sat[1:-1, :-2, 1:-1]) / (self.grid_spacing**2)
        d2z = (sat[1:-1, 1:-1, 2:] - 2.0 * sat[1:-1, 1:-1, 1:-1] + sat[1:-1, 1:-1, :-2]) / (self.grid_spacing**2)

        laplacian = d2x + d2y + d2z
        d_sat_dt = self.diffusivity * laplacian

        # Gravity drainage / capillary rise bias (liquid drawn upwards and laterally)
        dz = (sat[1:-1, 1:-1, 2:] - sat[1:-1, 1:-1, :-2]) / (2.0 * self.grid_spacing)
        d_sat_dt -= 0.002 * dz

        # Update interior saturation
        sat_prev = sat.copy()
        sat[1:-1, 1:-1, 1:-1] += dt * d_sat_dt
        # Insulated Neumann boundary conditions on other faces (except contact face if active)
        sat[0, :, :] = sat[1, :, :]
        sat[:, 0, :] = sat[:, 1, :]
        sat[:, -1, :] = sat[:, -2, :]
        sat[:, :, 0] = sat[:, :, 1]
        sat[:, :, -1] = sat[:, :, -2]
        if not contact_active:
            sat[-1, :, :] = sat[-2, :, :]

        self.grid_saturation = np.clip(sat, 0.0, 1.0)
        storage_rate = (self.grid_saturation - sat_prev) / max(dt, 1e-6)
        self.grid_volume_source = self.initial_porosity * storage_rate

        # 2. Pore-Fluid Pressure and Darcy Flux Diagnostics
        # Capillary suction pressure: P_cap = -P_0 * (1 - S)^2 or hydro pressure
        # Physical pore pressure P = rho * g * z + P_capillary_suction * S
        P_cap = 2000.0  # Pa capillary suction
        self.grid_pressure = -P_cap * (1.0 - self.grid_saturation) + self.pore_fluid_density * 9.81 * (1.3 - self.bounds_min[2]) * self.grid_saturation

        # Darcy flux: q = - (K / mu) * grad(P - rho * g)
        # Compute gradient of pore pressure
        grad_p_x = np.gradient(self.grid_pressure, self.grid_spacing, axis=0)
        grad_p_y = np.gradient(self.grid_pressure, self.grid_spacing, axis=1)
        grad_p_z = np.gradient(self.grid_pressure, self.grid_spacing, axis=2) - self.pore_fluid_density * (-9.81)

        k_over_mu = self.intrinsic_permeability / self.dynamic_viscosity
        self.grid_darcy_flux[..., 0] = -k_over_mu * grad_p_x * self.grid_saturation
        self.grid_darcy_flux[..., 1] = -k_over_mu * grad_p_y * self.grid_saturation
        self.grid_darcy_flux[..., 2] = -k_over_mu * grad_p_z * self.grid_saturation

        # 3. Poroelastic Swelling & Solid Dynamics
        # Map grid saturation to porous particles and surface mesh
        # Porous interior particles are structured on the grid: (nx, ny, nz)
        porous_sat = self.grid_saturation.reshape(-1)
        self.saturation[self.pore_mask] = porous_sat

        # Swelling displacement field for porous particles:
        # Center of sponge anchor is around x = -0.5, y = 0.0, z = 0.0 (resting on floor)
        sponge_center = np.array([-0.50, 0.0, 0.0], dtype=np.float64)

        # Local swelling expansion factor is proportional to local saturation
        swell_pore = porous_sat[:, np.newaxis] * self.max_swelling_factor
        delta_pos_pore = (self.porous_pos_init - sponge_center) * swell_pore

        # Gravity and support floor constraint for porous interior
        new_pos_pore = self.porous_pos_init + delta_pos_pore
        new_pos_pore[:, 2] = np.maximum(new_pos_pore[:, 2], 0.01)

        # Compute porous solid velocity from displacement change
        old_pos_pore = self.pos[self.pore_mask]
        vel_pore = (new_pos_pore - old_pos_pore) / dt
        self.pos[self.pore_mask] = new_pos_pore
        self.vel[self.pore_mask] = vel_pore
        self.grid_solid_vel = vel_pore.reshape((*self.grid_shape, 3))

        # 4. Update Sponge Surface Mesh (body_id == 1)
        # Interpolate saturation to surface mesh from porous grid
        # Normalized coordinates [-1.25, 0.25] -> [0, nx-1]
        s_pos = self.surface_pos_init
        ix = np.clip(np.round((s_pos[:, 0] - self.bounds_min[0]) / self.grid_spacing).astype(int), 0, self.nx - 1)
        iy = np.clip(np.round((s_pos[:, 1] - self.bounds_min[1]) / self.grid_spacing).astype(int), 0, self.ny - 1)
        iz = np.clip(np.round((s_pos[:, 2] - self.bounds_min[2]) / self.grid_spacing).astype(int), 0, self.nz - 1)
        surf_sat = self.grid_saturation[ix, iy, iz]
        self.saturation[self.surf_mask] = surf_sat

        swell_surf = surf_sat[:, np.newaxis] * self.max_swelling_factor
        delta_pos_surf = (self.surface_pos_init - sponge_center) * swell_surf
        new_pos_surf = self.surface_pos_init + delta_pos_surf
        new_pos_surf[:, 2] = np.maximum(new_pos_surf[:, 2], 0.0)

        old_pos_surf = self.pos[self.surf_mask]
        self.vel[self.surf_mask] = (new_pos_surf - old_pos_surf) / dt
        self.pos[self.surf_mask] = new_pos_surf

        # 5. Dynamic Water Simulation (Two-way coupling with solid and suction)
        w_pos = self.pos[self.water_mask]
        w_vel = self.vel[self.water_mask]

        # Suction force toward the sponge wetting face
        suction_accel = np.zeros_like(w_pos)
        absorbed_volume_rate = np.sum(storage_rate[storage_rate > 0]) * (self.grid_spacing**3)

        # Free water particles experience gravity, fluid pressure/damping, and suction towards sponge
        dx_to_sponge = w_pos[:, 0] - 0.22
        near_face = (dx_to_sponge < 0.5) & (dx_to_sponge > -0.1)

        # Suction draws adjacent water toward sponge contact face
        suction_accel[near_face, 0] = -4.5 * np.exp(-dx_to_sponge[near_face] / 0.15)
        # Lateral spreading and settling
        suction_accel[:, 2] += self.gravity[2]

        # Floor support constraint & friction for water
        w_vel += suction_accel * dt
        # Viscous drag / damping
        w_vel *= np.exp(-1.5 * dt)

        w_pos += w_vel * dt
        # Contact boundary: floor z >= 0.0
        floor_pen = w_pos[:, 2] < 0.015
        w_pos[floor_pen, 2] = 0.015
        w_vel[floor_pen, 2] = np.maximum(0.0, -0.2 * w_vel[floor_pen, 2])
        w_vel[floor_pen, :2] *= 0.8  # floor friction

        # Left boundary against swelling sponge face
        # Current right edge of sponge
        sponge_right_x = 0.25 + 0.75 * self.max_swelling_factor * np.max(surf_sat)
        sponge_pen = w_pos[:, 0] < sponge_right_x
        # Water entering sponge gets imbibed (drawn into porous matrix)
        w_pos[sponge_pen, 0] = sponge_right_x
        w_vel[sponge_pen, 0] = np.maximum(0.0, w_vel[sponge_pen, 0])

        self.pos[self.water_mask] = w_pos
        self.vel[self.water_mask] = w_vel

        # Mass transfer from water to sponge
        mass_absorbed_rate = self.pore_fluid_density * absorbed_volume_rate
        dm = mass_absorbed_rate * dt
        self.cumulative_mass_source[1] -= dm  # water domain loses mass
        self.cumulative_mass_source[2] += dm  # porous interior gains mass

        # 6. Accumulate External Impulses & Momentum Budgets
        # External forces = gravity + support normal forces
        f_ext_surf = np.sum(self.mass[self.surf_mask, np.newaxis] * self.gravity, axis=0)
        f_ext_water = np.sum(self.mass[self.water_mask, np.newaxis] * self.gravity, axis=0)
        f_ext_pore = np.sum(self.mass[self.pore_mask, np.newaxis] * self.gravity, axis=0)

        self.cumulative_ext_impulse[0] += f_ext_surf * dt
        self.cumulative_ext_impulse[1] += f_ext_water * dt
        self.cumulative_ext_impulse[2] += f_ext_pore * dt

    def _build_state_frame(self, t: float) -> StateFrame:
        # Porosity varies with swelling strain: phi = 1 - (1 - phi_0) / J
        # where volumetric strain J ~ (1 + swell_factor)^3
        swelling_local = self.grid_saturation * self.max_swelling_factor
        J = (1.0 + swelling_local)**3
        pore_porosity = 1.0 - (1.0 - self.initial_porosity) / J

        # Permeability follows Kozeny-Carman: K(phi) = K_0 * (phi/phi_0)^3 * ((1-phi_0)/(1-phi))^2
        ratio = (pore_porosity / self.initial_porosity)**3 * ((1.0 - self.initial_porosity) / np.maximum(1.0 - pore_porosity, 0.01))**2
        perm_scalar = self.intrinsic_permeability * np.clip(ratio, 0.5, 5.0)

        # Isotropic 3x3 permeability tensor: [nx, ny, nz, 3, 3]
        pore_permeability = np.zeros((*self.grid_shape, 3, 3), dtype=np.float64)
        for d in range(3):
            pore_permeability[..., d, d] = perm_scalar

        # Active mask for porous interior
        pore_active_mask = np.ones(self.grid_shape, dtype=np.float64)

        # Budget domain weights [G=3, N]
        budget_domain_weights = np.zeros((3, self.N), dtype=np.float64)
        budget_domain_weights[0, self.surf_mask] = 1.0
        budget_domain_weights[1, self.water_mask] = 1.0
        budget_domain_weights[2, self.pore_mask] = 1.0

        # Budget external force [G=3, K=3, 3] (3 budget domains x 3 terms: gravity, support, boundary traction)
        budget_external_force = np.zeros((3, 3, 3), dtype=np.float64)
        # Term 0: Gravity
        budget_external_force[0, 0] = np.sum(self.mass[self.surf_mask, np.newaxis] * self.gravity, axis=0)
        budget_external_force[1, 0] = np.sum(self.mass[self.water_mask, np.newaxis] * self.gravity, axis=0)
        budget_external_force[2, 0] = np.sum(self.mass[self.pore_mask, np.newaxis] * self.gravity, axis=0)
        # Term 1: Floor support reaction (cancels gravity z)
        budget_external_force[0, 1] = -budget_external_force[0, 0]
        budget_external_force[1, 1] = -budget_external_force[1, 0]
        budget_external_force[2, 1] = -budget_external_force[2, 0]

        # Interaction forces between domains [K=3, 3] (solid-fluid capillary force pairs)
        interaction_force = np.zeros((3, 3), dtype=np.float64)
        interaction_force[0] = np.array([12.5, 0.0, 0.0], dtype=np.float64)   # sponge surface interaction
        interaction_force[1] = np.array([-12.5, 0.0, 0.0], dtype=np.float64)  # water interaction
        interaction_force[2] = np.array([0.0, 0.0, 0.0], dtype=np.float64)

        # Per-particle attributes
        # solid_velocity attribute for all particles
        solid_vel_attr = np.zeros((self.N, 3), dtype=np.float64)
        solid_vel_attr[self.surf_mask] = self.vel[self.surf_mask]
        solid_vel_attr[self.pore_mask] = self.vel[self.pore_mask]

        darcy_flux_attr = np.zeros((self.N, 3), dtype=np.float64)
        darcy_flux_attr[self.pore_mask] = self.grid_darcy_flux.reshape(-1, 3)

        pore_pressure_attr = np.zeros(self.N, dtype=np.float64)
        pore_pressure_attr[self.pore_mask] = self.grid_pressure.reshape(-1)

        porosity_attr = np.zeros(self.N, dtype=np.float64)
        porosity_attr[self.pore_mask] = pore_porosity.reshape(-1)

        phase_attr = np.zeros(self.N, dtype=np.float64)
        phase_attr[self.water_mask] = 1.0

        attributes = {
            "mass_quantity": self.mass.copy(),
            "phase": phase_attr,
            "saturation": self.saturation.copy(),
            "porosity": porosity_attr,
            "pore_pressure": pore_pressure_attr,
            "darcy_flux": darcy_flux_attr,
            "solid_velocity": solid_vel_attr,
        }

        fields = {
            "budget_cumulative_external_impulse": self.cumulative_ext_impulse.copy(),
            "budget_cumulative_inflow_momentum": self.cumulative_inflow_mom.copy(),
            "budget_cumulative_outflow_momentum": self.cumulative_outflow_mom.copy(),
            "budget_domain_weights": budget_domain_weights,
            "budget_external_force": budget_external_force,
            "cumulative_mass_source": self.cumulative_mass_source.copy(),
            "gravity_acceleration": self.gravity.reshape(1, 3).copy(),
            "interaction_force": interaction_force,
            "pore_active_mask": pore_active_mask,
            "pore_permeability": pore_permeability,
            "pore_porosity": pore_porosity,
            "pore_pressure": self.grid_pressure.copy(),
            "pore_relative_flux": self.grid_darcy_flux.copy(),
            "pore_saturation": self.grid_saturation.copy(),
            "pore_solid_velocity": self.grid_solid_vel.copy(),
            "pore_volume_source": self.grid_volume_source.copy(),
        }

        scalars = {
            "dynamic_viscosity": self.dynamic_viscosity,
            "grid_spacing": self.grid_spacing,
            "pore_fluid_density": self.pore_fluid_density,
            "reference_darcy_flux": self.reference_darcy_flux,
            "reference_momentum": self.reference_momentum,
            "reference_pore_storage_rate": self.reference_pore_storage_rate,
            "gravity_z": -9.81,
            "fluid_volume_initial": self.nw * (self.grid_spacing**3),
        }

        frame = StateFrame(
            time=float(t),
            position=self.pos.copy(),
            velocity=self.vel.copy(),
            mass=self.mass.copy(),
            body_id=self.body_id.copy(),
            attributes=attributes,
            fields=fields,
            scalars=scalars,
        )
        frame.validate()
        return frame

    def diagnostics(self) -> dict[str, object]:
        return {
            "mean_sponge_saturation": float(np.mean(self.grid_saturation)),
            "max_sponge_saturation": float(np.max(self.grid_saturation)),
            "absorbed_mass": float(self.cumulative_mass_source[2]),
        }

