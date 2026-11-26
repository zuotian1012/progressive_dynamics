"""Label saved Genesis frames without running or changing simulation.

The original video is retained as ``<stem>_unlabelled.mp4``. A successful
annotation is recorded in ``<video>.frames.json``; identical repeat requests
are skipped. Run this with the Genesis environment (OpenCV/NumPy installed).
"""

import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def integer_frames(values, name, count=None):
    values = np.asarray(values)
    if values.ndim != 1 or not len(values):
        raise ValueError(f"{name} must be a nonempty one-dimensional frame array")
    if count is not None and len(values) != count:
        raise ValueError(f"{name} length differs from source_frames")
    if not np.issubdtype(values.dtype, np.number):
        raise ValueError(f"{name} must contain numeric frame identifiers")
    if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
        raise ValueError(f"{name} must contain finite integer frame identifiers")
    return values.astype(np.int64)


def scalar_value(replay, name):
    if name not in replay:
        return None
    value = np.asarray(replay[name])
    return float(value.reshape(-1)[0]) if value.size == 1 else None


def load_mapping(path, phase, requested_hz):
    with np.load(path, allow_pickle=False) as replay:
        source_frames = integer_frames(replay["source_frames"], "source_frames")
        native_frames = None
        native_key = None
        if phase == "settle":
            for key in ("native_frames", "native_frame", "frame", "frames"):
                if key in replay and np.asarray(replay[key]).ndim == 1:
                    native_frames = integer_frames(replay[key], key, len(source_frames))
                    native_key = key
                    break
        physics_hz = requested_hz
        hz_source = "argument" if requested_hz is not None else None
        if physics_hz is None and phase == "settle":
            physics_hz = scalar_value(replay, "physics_hz")
            hz_source = "replay.physics_hz" if physics_hz is not None else None
            if physics_hz is None:
                dt = scalar_value(replay, "physical_dt")
                if dt is not None and math.isfinite(dt) and dt > 0:
                    physics_hz, hz_source = 1.0 / dt, "replay.physical_dt"
        if phase == "settle" and (physics_hz is None or not math.isfinite(physics_hz) or physics_hz <= 0):
            raise ValueError("Settling labels require --physics-hz or a positive replay physical_dt/physics_hz")
    return source_frames, native_frames, native_key, physics_hz, hz_source


def video_metadata(path):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not math.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"Invalid video dimensions/FPS: {path}")
        return fps, width, height
    finally:
        capture.release()


def verify_decode(path, expected_count, expected_fps, expected_size):
    fps, width, height = video_metadata(path)
    if not math.isclose(fps, expected_fps, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(f"Encoded FPS changed: {expected_fps} -> {fps}")
    if (width, height) != expected_size:
        raise RuntimeError(f"Encoded resolution changed: {expected_size} -> {(width, height)}")
    capture = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame.shape[:2] != (height, width):
                raise RuntimeError("Decoded frame has inconsistent dimensions")
            decoded += 1
    finally:
        capture.release()
    if decoded != expected_count:
        raise RuntimeError(f"Decoded {decoded} video frames; replay has {expected_count}")
    return dict(full_decode_pass=True, decoded_frames=decoded, fps=fps, width=width, height=height)


def ffmpeg_path():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def draw_label(frame, lines):
    height, width = frame.shape[:2]
    scale = min(1.05, max(0.4, height / 850.0))
    thickness = max(1, round(scale * 2))
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = max(8, round(height / 70))
    while max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + 4 * pad > width and scale > 0.2:
        scale *= 0.9
    line_height = cv2.getTextSize("Ag", font, scale, thickness)[0][1] + pad
    box_width = min(width, max(cv2.getTextSize(line, font, scale, thickness)[0][0] for line in lines) + 3 * pad)
    box_height = min(height, len(lines) * line_height + 2 * pad)
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad // 2, pad // 2), (box_width, box_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0.0, frame)
    for index, line in enumerate(lines):
        cv2.putText(frame, line, (pad, pad + (index + 1) * line_height), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)


def encode_labels(source, target, source_frames, native_frames, phase, physics_hz, action_fps, fps, size):
    width, height = size
    rate = Fraction(fps).limit_denominator(1_000_000)
    pixel_format = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
    command = [ffmpeg_path(), "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo",
               "-pixel_format", "bgr24", "-video_size", f"{width}x{height}", "-framerate",
               f"{rate.numerator}/{rate.denominator}", "-i", "pipe:0", "-an", "-c:v", "libx264",
               "-preset", "fast", "-crf", "18", "-threads", "2", "-pix_fmt", pixel_format,
               "-movflags", "+faststart", str(target)]
    capture = cv2.VideoCapture(str(source))
    count = 0
    with tempfile.TemporaryFile() as errors:
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if count >= len(source_frames):
                    raise RuntimeError("Video has more frames than replay")
                if phase == "trajectory":
                    lines = [f"source_frame: {source_frames[count]} | t: {source_frames[count] / action_fps:.3f} s",
                             f"trajectory | video_frame: {count + 1}/{len(source_frames)}"]
                else:
                    lines = [f"settle_step: {count + 1}/{len(source_frames)} | t: {(count + 1) / physics_hz:.3f} s"]
                    if native_frames is not None:
                        lines.append(f"native_frame: {native_frames[count]}")
                draw_label(frame, lines)
                encoder.stdin.write(frame.tobytes())
                count += 1
            if count != len(source_frames):
                raise RuntimeError(f"Video has {count} frames; replay has {len(source_frames)}")
            encoder.stdin.close()
            if encoder.wait() != 0:
                errors.seek(0)
                raise RuntimeError("FFmpeg annotation failed: " + errors.read().decode(errors="replace"))
        except BaseException:
            if encoder.stdin and not encoder.stdin.closed:
                encoder.stdin.close()
            if encoder.poll() is None:
                encoder.terminate()
            encoder.wait()
            raise
        finally:
            capture.release()
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--phase", choices=("trajectory", "settle"), required=True)
    parser.add_argument("--physics-hz", type=float)
    parser.add_argument("--action-fps", type=float, default=60.0)
    args = parser.parse_args()
    if not math.isfinite(args.action_fps) or args.action_fps <= 0:
        parser.error("--action-fps must be finite and positive")
    if args.physics_hz is not None and (not math.isfinite(args.physics_hz) or args.physics_hz <= 0):
        parser.error("--physics-hz must be finite and positive")
    video, replay = args.video.resolve(strict=True), args.replay.resolve(strict=True)
    backup = video.with_name(video.stem + "_unlabelled" + video.suffix)
    sidecar = video.with_suffix(video.suffix + ".frames.json")
    source_frames, native_frames, native_key, physics_hz, hz_source = load_mapping(replay, args.phase, args.physics_hz)
    signature = dict(code_sha256=file_hash(Path(__file__)), replay_sha256=file_hash(replay),
                     phase=args.phase, physics_hz=physics_hz, action_fps=args.action_fps)
    if sidecar.exists():
        prior = json.loads(sidecar.read_text())
        if prior.get("request") == signature and prior.get("video_sha256") == file_hash(video) and backup.exists():
            print(json.dumps(dict(status="already_annotated", video=str(video), manifest=str(sidecar))))
            return
    source = backup if backup.exists() else video
    fps, width, height = video_metadata(source)
    with tempfile.NamedTemporaryFile(prefix=video.stem + "_label_", suffix=video.suffix,
                                     dir=video.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        encode_labels(source, temporary, source_frames, native_frames, args.phase, physics_hz,
                      args.action_fps, fps, (width, height))
        validation = verify_decode(temporary, len(source_frames), fps, (width, height))
        source_hash, encoded_hash = file_hash(source), file_hash(temporary)
        if not backup.exists():
            video.replace(backup)
        try:
            temporary.replace(video)
        except BaseException:
            if not video.exists():
                shutil.copy2(backup, video)
            raise
        mapping = dict(replay=str(replay), frames=len(source_frames),
                       source_frame_first=int(source_frames[0]), source_frame_last=int(source_frames[-1]),
                       source_frames_sha256=hashlib.sha256(source_frames.tobytes()).hexdigest(),
                       native_frames_key=native_key,
                       native_frame_first=int(native_frames[0]) if native_frames is not None else None,
                       native_frame_last=int(native_frames[-1]) if native_frames is not None else None,
                       time_formula="source_frame/action_fps" if args.phase == "trajectory" else "(video_index+1)/physics_hz",
                       physics_hz_source=hz_source)
        manifest = dict(request=signature, video=str(video), unlabelled_video=str(backup),
                        unlabelled_sha256=source_hash, video_sha256=encoded_hash, phase=args.phase,
                        physics_advanced=False, fps=fps, mapping=mapping, validation=validation)
        sidecar.write_text(json.dumps(manifest, indent=2) + "\n")
        print(json.dumps(dict(status="annotated", video=str(video), manifest=str(sidecar), frames=len(source_frames), fps=fps)))
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
