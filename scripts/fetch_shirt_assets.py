"""Fetch only Scene527 cloth, command and IPC collision assets; verify SHA256.

Run after cloning the reference branch with Git LFS smudging disabled.
This avoids downloading the large replay/checkpoint/rendering bundles.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sys
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shirt_demo.assets import SOURCE, BUNDLE, LINKS


def fetch(path):
    prefix = path.read_bytes()[:250]
    if not prefix.startswith(b"version https://git-lfs"):
        return f"Present: {path.name}"
    fields = dict(line.split(" ", 1) for line in prefix.decode().splitlines())
    expected = fields["oid"].split(":", 1)[1]
    relative = path.relative_to(SOURCE).as_posix()
    url = "https://media.githubusercontent.com/media/happy1041/genesis-ipc-shirt-demo/c29cc92f954fa4b803fb5cfe3155d6d9dbd5b088/" + relative
    with urllib.request.urlopen(url, timeout=45) as response:
        payload = response.read()
    if len(payload) != int(fields["size"]) or hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError(f"Failed size/hash check: {relative}")
    temporary = path.with_suffix(path.suffix + ".download")
    temporary.write_bytes(payload)
    temporary.replace(path)
    return f"Verified: {path.name} ({len(payload)} bytes)"


def main():
    urdf = BUNDLE / "assets/robots/dual_x5_2025_ipc_v1/dual_x5_2025_ipc.urdf"
    xml = ET.parse(urdf).getroot()
    files = list((BUNDLE / "assets/cloth").glob("*.obj")) + [BUNDLE / "controls/trajectory.npz"]
    for name in LINKS:
        for mesh in xml.findall(f"link[@name='{name}']/collision/geometry/mesh"):
            files.append((urdf.parent / mesh.get("filename")).resolve())
    with ThreadPoolExecutor(max_workers=4) as pool:
        for message in pool.map(fetch, sorted(set(files))):
            print(message, flush=True)


if __name__ == "__main__":
    main()
