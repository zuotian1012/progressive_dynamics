"""Compare coarse starts against the same 96x96 direct IPC trajectory."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.assets.generate_pd_square_cloth import prolongation
from tools.pd.report_velpro_drape import fit, load_case, resolutions, rest_weights, spatial_rms
from tools.pd.run_integrator_benchmark import lift, read_mesh, read_p


def measure(reference, root, modes):
    cells = resolutions(root)
    meshes = root / "hierarchy"
    last = len(cells) - 1
    if cells[-1] != 96:
        raise ValueError("Comparisons require the same 96×96 finest grid")
    mass = rest_weights(*read_mesh(meshes, last))
    bulk_map = prolongation(16, cells[-1])
    bulk_weights = rest_weights(*read_mesh(reference["hierarchy"], 0))
    expected_bulk = fit(bulk_map, reference["positions"])
    preview, _, _ = load_case(root / "preview_L0")
    reference_rest, reference_tri = read_mesh(reference["hierarchy"], 3)
    rest, tri = read_mesh(meshes, last)
    np.testing.assert_array_equal(rest, reference_rest)
    np.testing.assert_array_equal(tri, reference_tri)
    preview_finest = lift(read_p(meshes, "direct", 0, last), preview)

    def rms(sequence):
        return float(np.sqrt(np.mean(sequence**2)))

    def compare(x):
        if x.shape != reference["positions"].shape:
            raise ValueError(f"{root}: common trajectory dimensions do not match")
        full = 1000 * spatial_rms(x - reference["positions"], mass)
        bulk = 1000 * spatial_rms(fit(bulk_map, x) - expected_bulk, bulk_weights)
        preview_distance = 1000 * spatial_rms(x - preview_finest, mass)
        return {"full_error_rms_mm": rms(full), "bulk_error_rms_mm": rms(bulk),
                "own_preview_distance_rms_mm": rms(preview_distance)}

    output = {"cells": list(cells), "preview": compare(preview_finest)}
    cases = ["preview_L0"]
    for mode in modes:
        name = f"{mode}_L{last}"
        x, v, _ = load_case(root / name)
        output[mode] = compare(x)
        output[mode]["second_velocity_rms_m_s"] = rms(spatial_rms(np.diff(v, n=2, axis=0), mass))
        cases.extend(f"{mode}_L{i}" for i in range(1, last + 1))
    if (root / "direct_fine/state.npz").exists():
        cases.append("direct_fine")
    timings = {}
    for name in cases:
        case = root / name
        _, _, timing = load_case(case)
        timing["case_elapsed_s"] = (case.joinpath("settings.json").stat().st_mtime_ns -
                                    case.joinpath("command.json").stat().st_mtime_ns) / 1e9
        if timing["case_elapsed_s"] < timing["solver_wall_s"]:
            raise ValueError(f"{case}: elapsed time derived from files is inconsistent")
        timings[name] = timing
    output["timings"] = timings
    for mode in modes:
        chain = [timings["preview_L0"]] + [timings[f"{mode}_L{i}"] for i in range(1, last + 1)]
        output[mode]["chain_solver_s"] = sum(case["solver_wall_s"] for case in chain)
        output[mode]["chain_case_elapsed_s"] = sum(case["case_elapsed_s"] for case in chain)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--six", type=Path, required=True)
    parser.add_argument("--four", type=Path, required=True)
    parser.add_argument("--four-control", type=Path, required=True)
    args = parser.parse_args()
    y, _, _ = load_case(args.four / "direct_fine")
    reference = {"positions": y, "hierarchy": args.four / "hierarchy"}
    summary = {
        "six_start_3": measure(reference, args.six, ("paper_eq5", "velpro")),
        "four_start_3": measure(reference, args.four_control, ("paper_eq5",)),
        "four_start_16": measure(reference, args.four, ("paper_eq5", "velpro")),
    }
    old_direct, _, _ = load_case(args.six / "direct_fine")
    p = prolongation(16, 96)
    weights = rest_weights(*read_mesh(reference["hierarchy"], 0))
    repeat_error = spatial_rms(fit(p, old_direct) - fit(p, y), weights)
    summary["old_vs_new_direct_bulk_rms_mm"] = float(1000 * np.sqrt(np.mean(repeat_error**2)))
    output = args.four / "coarse_start_comparison.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    for name in ("six_start_3", "four_start_3", "four_start_16"):
        case = summary[name]
        print(name, case["preview"]["bulk_error_rms_mm"], case["paper_eq5"]["bulk_error_rms_mm"])


if __name__ == "__main__":
    main()
