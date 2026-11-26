"""Summarize failure-only native PSQ probes; never treats probes as trajectory states."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def read_rows(log, marker):
    rows = []
    for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
        if marker + ' ' in line:
            rows.append({k: float(v) for k, v in re.findall(
                r'([a-z_]+)=([-+\d.eE]+)', line.split(marker, 1)[1])})
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--diagnostic', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    baseline = json.loads((a.baseline/'run.json').read_text())
    diagnostic = json.loads((a.diagnostic/'run.json').read_text())
    log = a.diagnostic.with_suffix('.log')
    full = read_rows(log, 'PSQ_DIRECTION_AUDIT')
    frozen = read_rows(log, 'PSQ_FROZEN_AUDIT')
    caps = read_rows(log, 'PSQ_AUDIT_CAPS')
    if not full or len(full) != len(frozen) or not caps:
        raise ValueError('Missing paired failure diagnostics')
    for left, right in zip(full, frozen):
        if any(left[k] != right[k] for k in ['frame', 'newton', 'alpha', 'predicted_slope']):
            raise ValueError('Unmatched directional probes')
    x = np.load(a.baseline/'accepted_positions.npy')
    y = np.load(a.diagnostic/'accepted_positions.npy')
    common = min(len(x), len(y))
    same_inputs = all(baseline[k] == diagnostic[k] for k in
                      ['assets', 'bundle_sha256', 'config', 'material', 'initial_stretch'])
    result = dict(diagnostic_only=True, probe_states_accepted=False,
                  physical_model_or_acceptance_changed=False, same_inputs=same_inputs,
                  baseline_accepted_steps=baseline['completed_steps'],
                  diagnostic_accepted_steps=diagnostic['completed_steps'],
                  common_prefix_max_position_difference=float(np.max(np.abs(x[:common]-y[:common]))),
                  full_energy_probes=full, frozen_offset_probes=frozen, step_caps=caps,
                  all_full_slopes_positive=all(r['actual_slope'] > 0 for r in full),
                  all_frozen_slopes_negative=all(r['actual_slope'] < 0 for r in frozen),
                  frozen_relative_slope_errors=[abs(r['actual_slope']/r['predicted_slope']-1) for r in frozen],
                  scope='A single failed Bunny L0 search direction; not proof of whole solver correctness',
                  source_hashes={})
    root = Path(__file__).resolve().parents[1]
    for name in ['psq_proxy.cu', 'psq_proxy.h', 'psq_direction_audit.inl']:
        result['source_hashes'][name] = hashlib.sha256((root/'native/libuipc'/name).read_bytes()).hexdigest()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
