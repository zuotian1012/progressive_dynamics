"""Locate the corrected bunny CUDA runtime or the pinned historical control."""
import hashlib
import json
from pathlib import Path

BUILD_ROOT = Path(__file__).resolve().parents[1] / 'output/flag_L3_fix_20261118'
DEFAULT_PACKAGE = BUILD_ROOT / 'bunny_frozen_runtime/src'
FROZEN_PACKAGE = BUILD_ROOT / 'bunny_precision_runtime/src'
ORDERED_PACKAGE = BUILD_ROOT / 'bunny_ordered_fem_runtime/src'
# The reconstructed L5 cloth has a 0.811 mm point-to-opposite-edge gap at
# rest. A 0.700 mm activation band plus 0.070 mm shell clearance fits it.
BUNNY_CONTACT_DISTANCE = .0007


def continuation_package(parent):
    if parent['level'] or parent.get('energy_checked_convergence'):
        return Path(parent['native_package'])
    return DEFAULT_PACKAGE


def audit_bunny_contact(run, meta):
    friction = meta['contact_friction']
    mu = friction['cloth_bunny']
    if mu not in (0., .3) or friction != dict(cloth_bunny=mu, cloth_self=mu, cloth_ground=0.):
        raise ValueError(f'Unexpected bunny contact coefficients: {friction}')
    contact = meta['config']['contact']
    if contact['enable'] != 1 or contact['friction']['enable'] != int(mu > 0):
        raise ValueError('Native contact settings disagree with the recorded friction experiment')
    systems = json.loads((run / 'backend/systems.json').read_text())['sim_systems']
    names = {system['name'].split('::')[-1] for system in systems if system['valid']}
    for prefix in ('IPCSimplex', 'IPCVertexHalfPlane'):
        if prefix + 'NormalContact' not in names or (prefix + 'FrictionalContact' in names) != (mu > 0):
            raise ValueError(f'Unexpected active CUDA contact systems for {prefix}, mu={mu}')
    constrained = 'SoftPositionConstraint' in names
    if constrained != (meta.get('consistency_weight', 0.) > 0):
        raise ValueError('Native soft position constraint disagrees with the requested consistency weight')
    return dict(cloth_bunny=mu, cloth_self=mu, cloth_ground=0., normal_contact_active=True,
                friction_systems_active=mu > 0, soft_position_constraint_active=constrained)


def check_run_sources(run, hashes):
    snapshots = {'run_bunny_ground_fine.py': 'source_snapshot.py', 'run_bunny_ground_l0.py': 'source_snapshot.py',
                 'paper_solver_settings.py': 'solver_settings_snapshot.py'}
    for source, expected in hashes.items():
        path = Path(source)
        if path.name in snapshots:
            path = run / snapshots[path.name]
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Source or asset differs from the simulated version: {path}')


def runtime_provenance(package):
    library = Path(package) / 'uipc/_native/uipc_backend_cuda.dll'
    if not library.is_file():
        raise ValueError(f'Missing CUDA library at {library}. See the build commands in docs/BUNNY_REPRODUCTION.md')
    with library.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    # The release runtime carries its own provenance, independent of historical
    # experiment directories. Never silently accept an unrecognized DLL.
    release_record = Path(package).resolve().parent / 'provenance.json'
    if release_record.is_file():
        provenance = json.loads(release_record.read_text(encoding='utf-8'))
        if provenance.get('release_source_build'):
            if provenance.get('dll_sha256') != digest:
                raise ValueError('Release CUDA library differs from its build provenance')
            return provenance
    for variant in ('bunny', 'rescue', 'bunny_control','bunny_stable','bunny_energy','bunny_rms','bunny_search','bunny_backtrack','bunny_frozen','bunny_paper','bunny_precision','bunny_direct','bunny_ordered','bunny_ordered_fem','bunny_ordered_pcg','bunny_ordered_mkl','bunny_ordered_cached','bunny_event_fallback'):
        path = BUILD_ROOT / f'{variant}_build/provenance.json'
        if path.is_file():
            provenance = json.loads(path.read_text())
            if provenance['dll_sha256'] == digest:
                return provenance
    raise ValueError(f'Unknown bunny CUDA build at {library}')
