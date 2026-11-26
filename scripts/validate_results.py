"""Check stored physical invariants without rerunning simulations."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cloth.simulation import ClothSystem, METHODS


def validate(directory):
    directory = Path(directory)
    c = json.loads((directory / "config.json").read_text())
    logs = json.loads((directory / "solver.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    system = ClothSystem(c)
    with np.load(directory / "trajectories.npz") as a:
        methods = [m for m in METHODS if f"{m}_0" in a]
        coarse = a[f"{methods[0]}_0"]
        for method in methods:
            np.testing.assert_array_equal(a[f"{method}_0"], coarse)
            for k, l in enumerate(system.levels):
                x = a[f"{method}_{k}"]
                assert x.shape == (c["steps"]+1, len(l.mass), 3)
                assert np.isfinite(x).all()
                np.testing.assert_array_equal(x[0], l.initial)
                if l.pins.any():
                    np.testing.assert_array_equal(x[:, l.pins], np.broadcast_to(l.initial[l.pins], x[:, l.pins].shape))
                if c.get("floor"):
                    assert x[:, :, 2].min() > c["floor"]["height"]
                assert all(r["finite"] and r["success"] for r in logs[method][k])
            if method == "independent":
                for level in metrics[method]["levels"]:
                    ratios = [q for q in level["literal_squared_ratio"] if q is not None]
                    np.testing.assert_allclose(ratios, 0.25, atol=1e-7, rtol=1e-7)
    contact = ", feasible ground" if c.get("floor") else ""
    print(f"PASS {directory}: finite trajectories, fixed pins{contact}, shared preview, solver status, metric control")


if __name__ == "__main__":
    for path in sys.argv[1:]:
        validate(path)
