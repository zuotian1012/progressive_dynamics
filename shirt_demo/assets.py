"""Read the reference assets without merging OBJ seams or changing units twice."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.sparse import csr_matrix, save_npz
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "external/genesis-ipc-shirt-demo"
BUNDLE = SOURCE / "reproduction/scene527_55k_73s"
LINKS = tuple(f"{side}_link{prefix}{i}" for side, prefix in (("left", 1), ("right", 2)) for i in (5, 6, 7, 8))


def require_asset(path):
    path = Path(path)
    with path.open("rb") as stream:
        if stream.read(80).startswith(b"version https://git-lfs"):
            raise ValueError(f"Git LFS data has not downloaded: {path}")
    return path


def read_obj(path):
    vertices, faces = [], []
    for line in require_asset(path).read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if not fields:
            continue
        if fields[0] == "v":
            vertices.append([float(v) for v in fields[1:4]])
        elif fields[0] == "f":
            ids = [int(v.split("/")[0]) for v in fields[1:]]
            ids = [v-1 if v > 0 else len(vertices)+v for v in ids]
            if len(ids) != 3:
                raise ValueError("Expected the reference triangular OBJ")
            faces.append(ids)
    v, f = np.asarray(vertices), np.asarray(faces, dtype=np.int32)
    if not len(v) or not len(f) or f.min() < 0 or f.max() >= len(v):
        raise ValueError("Invalid OBJ topology")
    return v, f


def subdivide(v, f):
    edges, inverse = np.unique(np.sort(f[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1), axis=0, return_inverse=True)
    mids = inverse.reshape(-1, 3) + len(v)
    a, b, c = f.T
    ab, bc, ca = mids.T
    fine_faces = np.vstack([np.column_stack(t) for t in ((a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca))]).astype(np.int32)
    rows = np.concatenate((np.arange(len(v)), np.repeat(np.arange(len(edges))+len(v), 2)))
    cols = np.concatenate((np.arange(len(v)), edges.ravel()))
    weights = np.concatenate((np.ones(len(v)), np.full(2*len(edges), .5)))
    p = csr_matrix((weights, (rows, cols)), shape=(len(v)+len(edges), len(v)))
    return p @ v, fine_faces, p


def prepare(output, coarse_faces=500, levels=3):
    import fast_simplification
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = BUNDLE / "assets/cloth/short-shirt-8000f.obj"
    v, f = read_obj(source)
    expected = json.loads((source.parent / "mesh_variants.json").read_text())["variants"][0]
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != expected["sha256"]:
        raise ValueError("Reference asset SHA256 mismatch")
    v, f = fast_simplification.simplify(v, f, target_count=coarse_faces, agg=5)
    # Match the repository's -90 degree X rotation, 0.001 scale and initial pose.
    v = v[:, [0, 2, 1]] * [.001, .001, -.001] + [.667, .015, .93]
    meta = {"source_commit": "c29cc92f954fa4b803fb5cfe3155d6d9dbd5b088", "source_sha256": digest,
            "source_faces": 8000, "generation": "quadric simplification, then nested midpoint subdivision",
            "original_8k_topology_preserved": False, "levels": []}
    for level in range(levels):
        if level:
            v, f, p = subdivide(v, f)
            save_npz(output / f"map_{level-1}_{level}.npz", p)
        np.savez_compressed(output / f"mesh_{level}.npz", vertices=v, faces=f)
        meta["levels"].append({"vertices": len(v), "faces": len(f)})
    (output / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def origin(element):
    t = np.eye(4)
    if element is not None:
        t[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        t[:3, :3] = Rotation.from_euler("xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    return t


class Robot:
    def __init__(self, trajectory=None):
        self.path = BUNDLE / "assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf"
        self.xml = ET.parse(self.path).getroot()
        with np.load(require_asset(BUNDLE / "controls/trajectory.npz"), allow_pickle=False) as data:
            self.q = data["joint_q"].copy()
            self.names = list(data["joint_names"])
            self.openness = data["openness"].copy()
        if trajectory is not None:
            original_initial = self.q[0].copy()
            with np.load(require_asset(trajectory), allow_pickle=False) as data:
                if list(data['joint_names']) != self.names:
                    raise ValueError('Custom trajectory joint order mismatch')
                self.q = np.asarray(data['joint_q'], dtype=float).copy()
                self.openness = np.asarray(data['openness'], dtype=float).copy()
            if self.q.ndim != 2 or self.q.shape[1] != len(self.names) or self.openness.shape != (len(self.q), 2):
                raise ValueError('Invalid custom trajectory dimensions')
            if not np.isfinite(self.q).all() or not np.isfinite(self.openness).all():
                raise ValueError('Nonfinite custom trajectory')
            np.testing.assert_array_equal(self.q[0], original_initial)
        self.joints = []
        for joint in self.xml.findall("joint"):
            axis = joint.find("axis")
            self.joints.append((joint.get("name"), joint.get("type"), joint.find("parent").get("link"),
                                joint.find("child").get("link"), origin(joint.find("origin")),
                                np.fromstring(axis.get("xyz") if axis is not None else "1 0 0", sep=" ")))

    def poses(self, q):
        values = dict(zip(self.names, q))
        poses = {"base_link": np.eye(4)}
        poses["base_link"][:3, 3] = [0, 0, .17]
        pending = self.joints.copy()
        while pending:
            remaining = []
            for name, kind, parent, child, offset, axis in pending:
                if parent not in poses:
                    remaining.append((name, kind, parent, child, offset, axis))
                    continue
                motion = np.eye(4)
                if kind in ("revolute", "continuous"):
                    motion[:3, :3] = Rotation.from_rotvec(axis / np.linalg.norm(axis) * values[name]).as_matrix()
                elif kind == "prismatic":
                    motion[:3, 3] = axis * values[name]
                elif kind != "fixed":
                    raise ValueError(f"Unsupported joint: {kind}")
                poses[child] = poses[parent] @ offset @ motion
            if len(remaining) == len(pending):
                raise ValueError("Invalid URDF tree")
            pending = remaining
        return poses

    def collisions(self):
        import trimesh
        result = []
        for name in LINKS:
            link = self.xml.find(f"link[@name='{name}']")
            vertices, faces = [], []
            for collision in link.findall("collision"):
                mesh = collision.find("geometry/mesh")
                path = require_asset((self.path.parent / mesh.get("filename")).resolve())
                loaded = trimesh.load_mesh(path, process=True)
                p = loaded.vertices * np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                t = origin(collision.find("origin"))
                p = p @ t[:3, :3].T + t[:3, 3]
                faces.append(loaded.faces + sum(len(a) for a in vertices))
                vertices.append(p)
            result.append((name, np.concatenate(vertices), np.concatenate(faces).astype(np.int32)))
        return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="output/shirt_assets")
    parser.add_argument("--coarse-faces", type=int, default=500)
    parser.add_argument("--levels", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output, args.coarse_faces, args.levels), indent=2))
