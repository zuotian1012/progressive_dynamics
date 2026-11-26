"""Report the C bending ablation and synchronized D assembly profile."""
import argparse
import json
from pathlib import Path

import numpy as np

from tools.pd.audit_drape_contact import gear_clip_geometry, intersecting_frames
from tools.pd.report_gear_drape_50k_sweep import load_gear, series, side_coverage
from tools.pd.report_velpro_drape import render_comparison, rest_weights, spatial_rms
from tools.pd.run_integrator_benchmark import lift, read_mesh, read_p


def read_json(path):
    return json.loads(path.read_text())


def load_x(run):
    with np.load(run / "state.npz", allow_pickle=False) as data:
        return data["x"]


def render_stills(root, sweep):
    gear, *_ = load_gear(sweep)
    reference = load_x(sweep / "A/direct_fine")
    fine_faces = read_mesh(sweep / "A/hierarchy", 2)[1]
    for pair in ("AB", "CD"):
        xs, faces, titles = [], [], []
        for name in pair:
            group = sweep / name
            coarse_faces = read_mesh(group / "hierarchy", 0)[1]
            xs.extend([load_x(group / case) for case in ("direct_coarse", "proxy_coarse", "velpro_fine")])
            xs.append(reference)
            faces.extend((coarse_faces, coarse_faces, fine_faces, fine_faces))
            titles.extend((f"{name} Direct Coarse", f"{name} Proxy Coarse", f"{name} VelPro Fine", "Direct Fine"))
        zoom = dict(camera_angles=(16., -42.), focus=((.075, -.13, .22), .41))
        for name, camera in (("overview", {}), ("zoom", zoom)):
            render_comparison(gear, xs, faces, titles, ("#987468", "#788dad", "#097c7b", "#326c99") * 2,
                              root / f"{pair}_{name}.png", rows=2, figsize=(16, 9),
                              keyframes=(46, 120), stills_only=True, **camera)


def analyze_ablation(root, sweep):
    coarse_rest, coarse_faces = read_mesh(sweep / "C/hierarchy", 0)
    fine_rest, fine_faces = read_mesh(sweep / "A/hierarchy", 2)
    reference, reference_stats = series(sweep / "A/direct_fine", len(fine_rest), 120)
    prolongation = read_p(sweep / "C/hierarchy", "direct", 0, 2)
    weights = rest_weights(fine_rest, fine_faces)
    _, outline, bottom, top = load_gear(sweep)
    clip = gear_clip_geometry(sweep / "C/gear.npz")
    uncapped = np.arange(121) >= 17
    uncapped[reference_stats["capped_frames"]] = False
    standard = read_json(root / "C_full_proxy/settings.json")
    alternative = read_json(root / "C_low_bending/settings.json")
    assert standard["config"] == alternative["config"] and standard["proxy"] == alternative["proxy"]
    adjusted = standard["effective_material"] | {"kappa": alternative["effective_material"]["kappa"]}
    assert adjusted == alternative["effective_material"]
    calibration = read_json(root / "ablation_setup.json")
    assert np.isclose(alternative["effective_material"]["kappa"], calibration["kappa"], rtol=1e-14)
    curves, rows = {}, {}
    for name, directory in (
        ("native_saved", sweep / "C/direct_coarse"), ("proxy_saved", sweep / "C/proxy_coarse"),
        ("full_proxy", root / "C_full_proxy"), ("low_bending", root / "C_low_bending"),
    ):
        x, timing = series(directory, len(coarse_rest), 120)
        fine = lift(prolongation, x)
        rms = 1000 * spatial_rms(fine - reference, weights)
        coverage = 100 * side_coverage(fine, weights, outline, bottom, top, .010)
        curves[name] = rms
        uncapped[timing["capped_frames"]] = False
        rows[name] = dict(rms_mm=float(np.sqrt(np.mean(rms[1:]**2))), last_mm=float(rms[-1]),
                          rms_at_046_mm=float(rms[46]), coverage_046_pct=float(coverage[46]),
                          coverage_final_pct=float(coverage[-1]), caps=timing["newton_cap_steps"],
                          max_cap_translation_mm=1000 * max(
                              (row["residual_m"] for row in timing["capped_translation"].values()), default=0.),
                          intersecting_frames=list(intersecting_frames(x, coarse_faces, clip)))
    if np.any(uncapped):
        for name, row in rows.items():
            row["common_uncapped_rms_mm"] = float(np.sqrt(np.mean(curves[name][uncapped]**2)))
    fresh = load_x(root / "C_full_proxy")
    coarse_weights = rest_weights(coarse_rest, coarse_faces)
    paired = {}
    for label, path in (("old_full", sweep / "C/proxy_coarse"), ("low_kappa", root / "C_low_bending")):
        difference = spatial_rms(load_x(path) - fresh, coarse_weights)
        paired[label] = 1000 * float(np.sqrt(np.mean(difference[1:]**2)))
    np.savetxt(root / "C_rms_curves.csv", np.column_stack([np.arange(121) * .01, *curves.values()]),
               delimiter=",", header="t_s," + ",".join(curves), comments="")
    return dict(rows=rows, common_uncapped_frames=int(uncapped.sum()), calibration=calibration,
                same_coarse_grid_rms_vs_fresh_full_mm=paired)


def analyze_profile(root):
    names = {
        "Build Linear System", "Assemble Subsystems", "Convert To BCOO",
        "Compute DyTopo Effect", "Assemble Dytopo Effect", "Convert Dytopo Matrix",
        "Distribute Dytopo Effect", "Assemble PD Proxy Elasticity", "Assemble IPC Contact Reporters",
        "Scatter FEM DyTopo Blocks",
    }
    result = {}
    for case in ("proxy_coarse", "velpro_middle"):
        timers = {}
        def visit(node):
            if node["name"] in names and "Newton Iteration" in node["parent"].split("/"):
                assert node["name"] not in timers, node["name"]
                timers[node["name"]] = {"seconds": node["duration"], "count": node["count"]}
            for child in node["children"]:
                visit(child)
        visit(read_json(root / f"D_{case}/profile.json"))
        assert names <= timers.keys(), (case, timers.keys())
        result[case] = timers
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stills", action="store_true")
    args = parser.parse_args()
    root = args.output.resolve()
    sweep = root.parent / "pd_gear_drape_50k_sweep"
    if args.stills:
        render_stills(root, sweep)
        return
    energies = read_json(sweep / "same_pose_energy.json")
    ratios = {name: [row["proxy_bending_J"] / row["native_bending_J"] for row in energies[name]["poses"]]
              for name in ("B", "C")}
    np.testing.assert_allclose(ratios["B"], 8, rtol=2e-14, atol=0)
    result = dict(bending_ratio=ratios, ablation=analyze_ablation(root, sweep), profiling=analyze_profile(root))
    (root / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
