"""Level bookkeeping only; no prolongation or solver mathematics."""
import json
from pathlib import Path


def finest_level(assets):
    assets = Path(assets)
    manifest = json.loads((assets / 'manifest.json').read_text())
    levels = manifest['levels']
    if not levels or [r.get('level', i) for i, r in enumerate(levels)] != list(range(len(levels))):
        raise ValueError('Hierarchy levels must be contiguous from zero')
    found = sorted(int(p.stem.split('_')[-1]) for p in assets.glob('mesh_*.npz'))
    if found != list(range(len(levels))):
        raise ValueError('Hierarchy manifest and mesh files disagree')
    return len(levels) - 1


def validate_level(assets, level, identity=None):
    last = finest_level(assets)
    if not 0 <= level <= last:
        raise ValueError('Requested level is outside the asset hierarchy')
    if identity is not None and bool(identity) != (level == last):
        raise ValueError('Identity bundle must correspond exactly to the finest asset level')
    return last
