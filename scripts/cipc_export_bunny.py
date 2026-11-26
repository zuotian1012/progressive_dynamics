"""Lossless OBJ export of the existing finest cloth and fixed bunny."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def export(assets, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {'source': str(assets.resolve()), 'meshes': {}, 'coordinate_transform': 'identity; Z up'}
    for source, name in [('mesh_7.npz', 'cloth'), ('bunny.npz', 'bunny')]:
        path = assets/source
        data = np.load(path)
        x, f = data['vertices'], data['faces']
        dest = output/(name+'.obj')
        with dest.open('w', newline='\n') as stream:
            np.savetxt(stream, x, fmt='v %.17g %.17g %.17g')
            np.savetxt(stream, f+1, fmt='f %d %d %d')
        manifest['meshes'][name] = {'source_file': source, 'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'obj_sha256': hashlib.sha256(dest.read_bytes()).hexdigest(), 'vertices': len(x), 'faces': len(f),
            'bounds': [x.min(axis=0).tolist(), x.max(axis=0).tolist()]}
    if manifest['meshes']['cloth']['vertices'] != 90000:
        raise ValueError('Expected the formal 90K mesh')
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--assets', type=Path, default=Path('output/bunny_paper_assets'))
    p.add_argument('--output', type=Path, default=Path('output/cipc_bunny_inputs'))
    a = p.parse_args()
    export(a.assets, a.output)
