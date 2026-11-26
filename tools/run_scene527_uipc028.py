"""Run the Scene527 checkpoint chain using the local libuipc 0.0.28 baseline."""

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from run_scene527_reproduction import replace_frames, verify


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "reproduction/scene527_55k_73s"
DEMO = ROOT / "dependencies/scene527-0.0.28"
SOURCE = DEMO / "src/run_genesis_ipc.py"
WORKER = DEMO / "tools/diagnose_first_fold_solver.py"
ANNOTATE = DEMO / "tools/annotate_genesis_video.py"
NATIVE_MATERIAL = {"stretch_E": 9800.0, "shear_E": 4.0, "uniform_kappa": 1e-7}
NEWTON_MAX = 100


def execute(command, log):
    with log.open("a") as stream:
        stream.write(json.dumps(command) + "\n")
        stream.flush()
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)


def render(base, replay, video, phase, fps, log):
    if video.with_suffix(".mp4.frames.json").is_file():
        return
    if not video.is_file():
        execute([
            *base, "--replay-states", str(replay), "--output", str(video),
            "--camera-view", "oblique", "--record-fps", str(fps), "--settle-frames", "0",
        ], log)
    rate_arg = "--physics-hz" if phase == "settle" else "--action-fps"
    execute([
        sys.executable, str(ANNOTATE), "--video", str(video), "--replay", str(replay),
        "--phase", phase, rate_arg, str(fps),
    ], log)


def validate_stage(stage_dir, first, last):
    config = json.loads((stage_dir / "effective_ipc_config.json").read_text())
    if (
        config["contact"]["constitution"] != "ipc"
        or config["contact"]["d_hat"] != 0.0015
        or not math.isclose(config["dt"], 1 / 120)
        or config["newton"]["max_iter"] != NEWTON_MAX
        or config["newton"]["semi_implicit"]["enable"]
    ):
        raise ValueError(f"Unexpected IPC solver configuration in {stage_dir}")
    native = json.loads((stage_dir / "native_material.json").read_text())
    if native["pyuipc_version"] != "0.0.28" or len(native["meshes"]) != 1:
        raise ValueError(f"Expected one native cloth mesh in {stage_dir}")
    mesh = native["meshes"][0]
    if (
        native["parameterization"] != "native"
        or mesh["legacy_adapter_applied"]
        or any(mesh[key] != value for key, value in NATIVE_MATERIAL.items())
        or mesh["lambda"]["count"] != 55068
        or mesh["kappa"]["min"] != NATIVE_MATERIAL["uniform_kappa"]
        or mesh["kappa"]["max"] != NATIVE_MATERIAL["uniform_kappa"]
    ):
        raise ValueError(f"Unexpected cloth material in {stage_dir}")
    checkpoint = stage_dir / "checkpoint.meta.json"
    if json.loads(checkpoint.read_text())["source_frame"] != last:
        raise ValueError(f"Wrong checkpoint frame in {stage_dir}")
    for name in ("checkpoint.pkl", "checkpoint.ipc_state.npz"):
        if not (stage_dir / name).is_file():
            raise FileNotFoundError(stage_dir / name)
    with np.load(stage_dir / "stage_physics.replay_states.npz", allow_pickle=False) as replay:
        if not np.array_equal(replay["source_frames"], np.arange(first, last + 1)):
            raise ValueError(f"Unexpected replay frames in {stage_dir}")
        if not np.isfinite(replay["cloth_pos"]).all():
            raise ValueError(f"Nonfinite cloth positions in {stage_dir}")


def update_index(output, stages, status):
    lines = [
        "# Scene527 55k · libuipc 0.0.28",
        "",
        f"状态：{status}。普通 IPC，60 Hz 动作 / 120 Hz 物理，原生材料，Newton 上限 {NEWTON_MAX}。",
        "",
        "[运行配置与版本](manifest.json)",
    ]
    for name, path in (
        ("完整 Genesis hero", "genesis_hero.mp4"),
        ("初始沉降 Genesis hero", "01_first_fold/settle_genesis_hero.mp4"),
        ("合成视频检查", "hero_validation.json"),
    ):
        if (output / path).is_file():
            lines.append(f"- [{name}]({path})")
    for stage in stages:
        stage_id = stage["id"]
        first, last = stage["frame_range"]
        links = []
        for name, file in (
            ("命令", "command.json"),
            ("实际 IPC 参数", "effective_ipc_config.json"),
            ("实际材料与后端", "native_material.json"),
            ("回放", "stage_physics.replay_states.npz"),
            ("Checkpoint", "checkpoint.pkl"),
            ("TCP", "stage_physics_tcp_trajectory.csv"),
            ("物理日志", "physics.log"),
            ("收敛统计", "solver_summary.json"),
            ("Timer", "timers.jsonl"),
            ("Genesis hero", "genesis_hero.mp4"),
        ):
            if (output / stage_id / file).is_file():
                links.append(f"[{name}]({stage_id}/{file})")
        if links:
            lines.append(f"- {stage_id}（{first}–{last}） · " + " · ".join(links))
    (output / "INDEX.md").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=args.resume)

    import genesis
    import uipc
    import imageio_ffmpeg

    overlay = ROOT / "dependencies/pyuipc-v0.0.28"
    if uipc.__version__ != "0.0.28" or not Path(uipc.__file__).resolve().is_relative_to(overlay):
        raise RuntimeError(f"Expected the local pyuipc 0.0.28 overlay, got {uipc.__file__}")
    if os.environ.get("GENESIS_IPC_CLOTH_PARAMETERIZATION") != "native":
        raise RuntimeError("Use scripts/run_scene527_uipc028.sh to select native materials")
    profile = json.loads((BUNDLE / "bundle.json").read_text())
    if not args.resume:
        verify(BUNDLE, profile)
    stages = profile["stages"]
    preset = [
        line.strip()
        for line in (BUNDLE / profile["preset"]).read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    common = [
        "--sim1-root", str(BUNDLE), "--trajectory", str(BUNDLE / profile["trajectory"]),
        "--shirt-obj", str(BUNDLE / profile["cloth"]), "--robot-urdf",
        str(BUNDLE / profile["robot_urdf"]), *preset,
        "--cloth-stretch-E", str(NATIVE_MATERIAL["stretch_E"]),
        "--cloth-shear-E", str(NATIVE_MATERIAL["shear_E"]),
        "--cloth-kappa", str(NATIVE_MATERIAL["uniform_kappa"]),
        "--cloth-strain-rate", "100",
    ]
    versioned = [
        Path(__file__), SOURCE, WORKER, ANNOTATE,
        Path(genesis.__file__).parent / "engine/couplers/ipc_coupler/coupler.py",
        Path(genesis.__file__).parent / "engine/couplers/ipc_coupler/legacy_cloth.py",
        Path(uipc.__file__).parent / "_native/libuipc_backend_cuda.so",
        BUNDLE / "bundle.json", BUNDLE / profile["preset"],
    ]
    manifest = {
        "pyuipc": uipc.__version__,
        "genesis": genesis.__file__,
        "libuipc_source": subprocess.check_output(
            ["git", "-C", str(ROOT / "dependencies/libuipc"), "rev-parse", "HEAD"], text=True
        ).strip(),
        "frames": [0, 4372],
        "action_fps": 60,
        "substeps": 2,
        "parameterization": "native",
        "native_material": NATIVE_MATERIAL,
        "newton_max": NEWTON_MAX,
        "detailed_timers": os.environ.get("GENESIS_IPC_DETAILED_TIMERS") == "1",
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in versioned},
    }
    manifest_path = output / "manifest.json"
    if args.resume:
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Run configuration or source changed since launch")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    lock_path = f"/tmp/genesis_ipc_shirt_demo_{os.getuid()}.lock"
    with open(lock_path, "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            run_chain(output, stages, common, args.headless, imageio_ffmpeg.get_ffmpeg_exe())
        except Exception:
            update_index(output, stages, "运行中断，详见各段日志")
            raise


def run_chain(output, stages, common, headless, ffmpeg):
    original = SOURCE.read_text()
    baseline_newton = "        newton_max_iterations = 50\n"
    if original.count(baseline_newton) != 1:
        raise RuntimeError("Could not locate the normal IPC Newton limit")
    physics_source = original.replace(
        baseline_newton, f"        newton_max_iterations = {NEWTON_MAX}\n"
    )
    previous = None
    render_jobs = []
    for stage in stages:
        stage_id = stage["id"]
        first, last = stage["frame_range"]
        stage_dir = output / stage_id
        marker = stage_dir / "physics_complete"
        base = [
            sys.executable, str(SOURCE), *common, *replace_frames(stage["args"], last + 1),
            "--no-keyframe-images", "--no-record",
        ]
        if not marker.exists():
            if stage_dir.exists():
                for attempt in range(1, 1000):
                    archived = stage_dir.with_name(f"{stage_id}.attempt-{attempt}")
                    if not archived.exists():
                        stage_dir.rename(archived)
                        break
            stage_dir.mkdir()
            (stage_dir / "physics_source_snapshot.py").write_text(physics_source)
            command = [
                sys.executable, str(WORKER), "--output", str(stage_dir),
                "--geometry-guard-interval", "60", "--worker",
            ]
            if previous is None:
                command.append("--capture-settle-replay")
            command += [
                "--", *base[2:], "--log-level", "info",
                "--output", str(stage_dir / "stage_physics.mp4"),
                "--dump-replay-states", str(stage_dir / "stage_physics.replay_states.npz"),
                "--save-checkpoint-source-frame", str(last),
                "--save-third-fold-checkpoint", str(stage_dir / "checkpoint.pkl"),
            ]
            if previous is not None:
                command += ["--load-third-fold-checkpoint", str(previous)]
            if not headless:
                command.append("--viewer")
            (stage_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            update_index(output, stages, f"正在模拟 {stage_id}")
            print(f"Physics {stage_id}: {first}–{last}", flush=True)
            execute(command, stage_dir / "physics.log")
            validate_stage(stage_dir, first, last)
            execute([
                sys.executable, str(WORKER), "--output", str(stage_dir), "--summarize-only",
            ], stage_dir / "summary.log")
            marker.touch()
        else:
            validate_stage(stage_dir, first, last)
        previous = stage_dir / "checkpoint.pkl"
        render_jobs.append((stage_dir, base))

    videos = []
    for stage_dir, base in render_jobs:
        update_index(output, stages, f"正在渲染 {stage_dir.name}")
        print(f"Render {stage_dir.name}", flush=True)
        video = stage_dir / "genesis_hero.mp4"
        render(base, stage_dir / "stage_physics.replay_states.npz", video,
               "trajectory", 60, stage_dir / "genesis_render.log")
        videos.append(video)
        settle_replay = stage_dir / "settle_native.replay_states.npz"
        if settle_replay.exists():
            render(base, settle_replay, stage_dir / "settle_genesis_hero.mp4",
                   "settle", 120, stage_dir / "settle_render.log")

    concat = output / "hero_concat.txt"
    concat.write_text("".join(f"file '{video.relative_to(output)}'\n" for video in videos))
    hero = output / "genesis_hero.mp4"
    execute([
        ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
        "-c", "copy", "-movflags", "+faststart", str(hero),
    ], output / "hero_concat.log")
    from imageio_ffmpeg import count_frames_and_secs

    frames, seconds = count_frames_and_secs(str(hero))
    if frames != 4373:
        raise ValueError(f"Full hero has {frames} frames, expected 4373")
    (output / "hero_validation.json").write_text(json.dumps({
        "frames": frames, "duration_seconds": seconds, "checkpoint_seams": [
            stage["frame_range"][1] for stage in stages[:-1]
        ],
    }, indent=2) + "\n")
    update_index(output, stages, "七段物理、离线渲染和合成视频已完成")
    print(f"Complete: {hero}", flush=True)


if __name__ == "__main__":
    main()
