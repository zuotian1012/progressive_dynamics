"""Compare fixed-endpoint 4- and 10-level flag experiments, including solver validity."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from cloth.report import COLORS, LABELS


def load(directory):
    directory = Path(directory)
    result = {key: json.loads((directory/f"{key}.json").read_text()) for key in ["config", "summary", "metrics", "solver"]}
    with np.load(directory/"trajectories.npz") as f:
        result["data"] = dict(f)
    return result


def compare(paths, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cases = [load(p) for p in paths]
    methods = ["semi", "full", "velpro"]
    base = cases[0]["config"]
    coarse = cases[0]["data"]["semi_0"]
    for case in cases:
        config = case["config"]
        assert config["segments"][0] == base["segments"][0] and config["segments"][-1] == base["segments"][-1]
        assert {k:v for k,v in config.items() if k != "segments"} == {k:v for k,v in base.items() if k != "segments"}
        for method in methods:
            np.testing.assert_array_equal(case["data"][f"{method}_0"], coarse)
    table = []
    for case in cases:
        for method in methods:
            summary = case["summary"]["methods"][method]
            warns = [(k, t+1, r["message"], r["gradient_inf"]) for k, rs in enumerate(case["solver"][method])
                     for t, r in enumerate(rs) if not r["success"]]
            table.append({"levels": len(case["config"]["segments"]), "method": method, **summary,
                          "first_solver_warning": warns[0] if warns else None,
                          "peak_solve_residual_rms_m": max(max(l["solve_residual_rms_m"]) for l in case["metrics"][method]["levels"])})
    (output/"comparison.json").write_text(json.dumps({"endpoint_meshes_fixed":True, "coarse_tracks_identical":True,
                                                   "rows":table}, indent=2))
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for case in cases:
        count = len(case["config"]["segments"])
        for method in methods:
            m = case["metrics"][method]
            levels = m["levels"]
            style = {"color":COLORS[method], "linestyle":"--" if count == 4 else "-", "marker":"o",
                     "markersize":3, "label":f"{method}, {count} levels"}
            xs = np.arange(count)
            axes[0,0].semilogy(xs, [max(l["continuity_rms_m"]) for l in levels], **style)
            axes[0,1].plot(xs, [max(l["max_speed_m_s"]) for l in levels], **style)
            axes[1,0].plot(xs, [max(l["max_principal_stretch"]) for l in levels], **style)
            axes[1,1].semilogy(xs, [max(l["solve_residual_rms_m"]) for l in levels], **style)
    axes[1,0].axhline(1.1, color="#555", linestyle=":", label="Paper strain limit (not enforced here)")
    titles = ["Peak continuity residual", "Peak vertex speed", "Peak principal stretch", "Peak optimizer position residual"]
    units = ["Mass-normalized RMS (m)", "Speed (m/s)", "Stretch ratio", "Mass-normalized RMS (m)"]
    for ax, title, unit in zip(axes.flat, titles, units):
        ax.set(title=title, xlabel="Level index (coarse to fine)", ylabel=unit)
        ax.grid(alpha=0.2)
    axes[0,0].legend(fontsize=8)
    axes[1,0].legend(fontsize=8)
    fig.suptitle("Flag hierarchy-depth experiment | same coarse and fine meshes", fontsize=14)
    fig.savefig(output/"hierarchy_metrics.png", dpi=150)
    plt.close(fig)

    # Animation: identical camera and world bounds across all six panels.
    tracks = [[case["data"][f"{m}_{len(case['config']['segments'])-1}"] for m in methods] for case in cases]
    xyz = np.concatenate([x.reshape(-1,3) for row in tracks for x in row])
    lo, hi = xyz.min(axis=0)-0.05, xyz.max(axis=0)+0.05
    center, radius = (lo+hi)/2, np.max(hi-lo)/2
    faces = cases[0]["data"][f"faces_{len(base['segments'])-1}"]
    fig = plt.figure(figsize=(14,9), facecolor="#fafafa")
    surfaces, titles = [], []
    for i, case in enumerate(cases):
        for j, method in enumerate(methods):
            ax = fig.add_subplot(2,3,i*3+j+1, projection="3d")
            ax.set(xlim=(center[0]-radius,center[0]+radius), ylim=(center[1]-radius,center[1]+radius),
                   zlim=(center[2]-radius,center[2]+radius))
            ax.view_init(elev=15, azim=-65)
            ax.set_box_aspect((1,1,1))
            ax.set_axis_off()
            p = Poly3DCollection(tracks[i][j][0][faces], facecolors=COLORS[method], edgecolors="#44525a", linewidths=0.16, alpha=0.96)
            ax.add_collection3d(p)
            ax.plot([-0.5,-0.5],[0,0],[base["initial_height"]-0.1,base["initial_height"]+1.1], color="#303845", lw=3)
            surfaces.append(p)
            titles.append(ax.set_title(f"{LABELS[method]} | {len(case['config']['segments'])} levels", y=0.96, fontsize=11))
    fig.subplots_adjust(left=0.01,right=0.99,bottom=0.03,top=0.84,wspace=0.01,hspace=0.12)
    header = fig.suptitle("Waving flag: 4 vs 10 levels", fontsize=17, y=0.98)
    fig.text(0.5,0.93,"Reduced experiment inspired by PD++ Figures 5-6; linear prolongation, no self-contact or strain limit", ha="center",fontsize=10)
    def update(t):
        for i,row in enumerate(tracks):
            for j,x in enumerate(row):
                surfaces[i*3+j].set_verts(x[t][faces])
        header.set_text(f"Waving flag: 4 vs 10 levels | t = {base['dt']*t:.2f} s")
        return surfaces+[header]
    # Select a diagnostic frame by peak 10-level semi continuity, not by claiming an instability onset.
    selected = int(np.argmax(cases[-1]["metrics"]["semi"]["levels"][-1]["continuity_rms_m"]))+1
    update(selected)
    fig.savefig(output/"flag_comparison.png", dpi=130)
    frames = np.unique(np.linspace(0,base["steps"],81).astype(int))
    FuncAnimation(fig,update,frames=frames,blit=False).save(output/"flag_comparison.gif",writer=PillowWriter(fps=20),dpi=80)
    plt.close(fig)
    (output/"diagnostic_frame.json").write_text(json.dumps({"frame":selected,"time_s":selected*base["dt"],
        "selection":"maximum fine-level continuity residual of 10-level semi, not asserted instability onset"},indent=2))
    print(json.dumps(table,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--four",default="output/flag_4levels")
    parser.add_argument("--ten",default="output/flag_10levels")
    parser.add_argument("--output",default="output/flag_study")
    args = parser.parse_args()
    compare([args.four,args.ten],args.output)
