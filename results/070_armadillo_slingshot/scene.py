"""Deterministic solver-neutral Armadillo slingshot scene."""
from __future__ import annotations

from pathlib import Path
import numpy as np
from worldsolver.contracts import StateFrame

ROOT = Path(__file__).resolve().parent
OBJ = ROOT / "assets" / "armadillo.obj"
SOLID = ROOT / "assets" / "armadillo_solid.npz"

# Ellipsoids form a readable low-poly Armadillo: torso, head, muzzle, ears,
# four limbs, broad palms/feet, and a backward tail.  Shared state covers the
# whole volumetric character; the topology asset supplies solver-neutral cells.
PARTS = [
    ((0,0,1.45),(0.48,.28,.70)), ((0,0,2.18),(.36,.27,.40)),
    ((0,-.08,2.43),(.25,.30,.22)), ((-.18,0,2.70),(.10,.10,.30)),
    ((.18,0,2.70),(.10,.10,.30)), ((-.58,0,2.12),(.42,.15,.16)),
    ((.58,0,2.12),(.42,.15,.16)), ((-.92,0,2.12),(.16,.12,.18)),
    ((.92,0,2.12),(.16,.12,.18)), ((-.25,0,.72),(.19,.18,.58)),
    ((.25,0,.72),(.19,.18,.58)), ((-.28,-.04,.14),(.25,.23,.13)),
    ((.28,-.04,.14),(.25,.23,.13)), ((0,.10,1.18),(.20,.16,.75)),
]

def _make_assets() -> None:
    """Create deterministic surface and cell assets if the packaged copies are absent."""
    OBJ.parent.mkdir(exist_ok=True)
    vertices, faces = [], []
    nu, nv = 16, 9
    for center, scale in PARTS:
        base = len(vertices)
        for j in range(nv + 1):
            phi = -np.pi/2 + np.pi*j/nv
            for i in range(nu):
                theta = 2*np.pi*i/nu
                q = np.array([np.cos(phi)*np.cos(theta), np.cos(phi)*np.sin(theta), np.sin(phi)])
                vertices.append(np.asarray(center)+np.asarray(scale)*q)
        for j in range(nv):
            for i in range(nu):
                a=base+j*nu+i; b=base+j*nu+(i+1)%nu
                c=base+(j+1)*nu+(i+1)%nu; d=base+(j+1)*nu+i
                faces.extend([(a,b,c),(a,c,d)])
    v=np.asarray(vertices,dtype=np.float64); f=np.asarray(faces,dtype=np.int32)
    with OBJ.open("w",encoding="ascii",newline="\n") as out:
        out.write("# deterministic Armadillo slingshot surface\n")
        for p in v: out.write(f"v {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n")
        for tri in f+1: out.write(f"f {tri[0]} {tri[1]} {tri[2]}\n")
    # Each ellipsoid has a center node and surface triangles swept to that
    # center, a conforming per-part tetrahedral volumetric solid description.
    centers=np.asarray([p[0] for p in PARTS],dtype=np.float64)
    points=np.vstack((v,centers)); tets=[]
    tris_per=nu*nv*2
    for k in range(len(PARTS)):
        ci=len(v)+k
        tets.extend([[ci,*tri] for tri in f[k*tris_per:(k+1)*tris_per]])
    np.savez(SOLID, vertices=points, faces=f, points=points,
             tetrahedra=np.asarray(tets,dtype=np.int32), surface_faces=f)

_make_assets()

def _vertices() -> np.ndarray:
    data=np.load(SOLID)
    return np.asarray(data["points"],dtype=np.float64)

class BenchmarkScene:
    def __init__(self, config: dict): self.config=config

    def metadata(self) -> dict:
        return {
            "scene_id": self.config["scene_id"], "seed": self.config["seed"],
            "phenomenon": "temporary upward pull and released elastic slingshot recoil",
            "solid_dimension": 3, "fixed_attribute": "fixed_grasp",
            "temporary_constraint_attribute": "pulled_patch",
            "event_timeline": self.config["event_timeline"],
        }

    def initial_state(self) -> StateFrame:
        p=_vertices(); n=len(p)
        left=np.linalg.norm(p-np.array([-.92,0,2.12]),axis=1)<.22
        right=np.linalg.norm(p-np.array([.92,0,2.12]),axis=1)<.22
        pulled=(p[:,2]<.42) & (np.abs(p[:,0])<.58)
        frame=StateFrame(0.0,p,np.zeros_like(p),mass=np.full(n,1.0/n),attributes={
            "fixed_grasp":(left|right).astype(np.float64),
            "pulled_patch":pulled.astype(np.float64),
        },scalars={"density_kg_m3":1000.0})
        frame.validate(); return frame

    def sample_times(self) -> list[float]:
        dt=float(self.config["sample_dt"]); duration=float(self.config["duration"])
        return (np.arange(round(duration/dt)+1,dtype=np.float64)*dt).tolist()
