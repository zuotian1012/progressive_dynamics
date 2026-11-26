"""Audit explicit rescue energy descent, dynamic response, and shirt isolation."""
from pathlib import Path
import argparse,hashlib,json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'output/flag_L0_shirt_aligned_20261118'
WORK=ROOT/'output/flag_L0_repair_20261118'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('run',type=Path);a=parser.parse_args()
    m=json.loads((a.run/'run.json').read_text());old=json.loads((BASE/'run.json').read_text())
    assert m['completed'] and m['initial_non_descent_rescue'] and not m['exact_offset_gradient']
    for k in ['config','material','bundle_sha256','assets','vertices','faces','pins','dt','steps']:
        assert m[k]==old[k],k
    triggers={};accepts={}
    for line in a.run.with_suffix('.log').read_text(encoding='utf-8-sig',errors='replace').splitlines():
        if 'FLAG_PSQ_RESCUE frame=' in line:
            d=dict(re.findall(r'(\w+)=([^\s]+)',line));triggers[int(d['frame'])]=d
        if 'FLAG_PSQ_RESCUE_ACCEPT frame=' in line:
            d={k:float(v) for k,v in re.findall(r'(\w+)=([^\s]+)',line)};accepts[int(d['frame'])]=d
            assert d['E1']<d['E0'] and 0<d['alpha']<=1 and 0<d['ccd']<=1 and d['newton']==1
    assert triggers.keys()==accepts.keys() and triggers
    audit=json.loads((a.run/'audit.json').read_text());assert audit['accepted_prefix_checks_passed']
    preview=json.loads((a.run/'report/non_descent_audit.json').read_text());assert preview['all_accepted_preview_evidence_verified']
    assert not preview['preview_at_first_newton_frames']
    x=np.load(a.run/'positions.npy');motion=np.linalg.norm(np.diff(x,axis=0),axis=2).max(axis=1)
    assert np.all(motion>1e-10*m['dt'])
    protected=json.loads((WORK/'protected_manifest.json').read_text())
    changes=[p for p,h in protected.items() if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changes,changes
    windows=[]
    for start in range(0,m['steps'],25):
        v=motion[start:start+25]/m['dt'];windows.append(dict(start=start*m['dt'],minimum_speed=float(v.min()),maximum_speed=float(v.max())))
    result=dict(passed=True,physical_seconds=m['steps']*m['dt'],rescue_steps=len(triggers),
        all_rescues_strictly_decreased_full_energy=True,zero_motion_steps=0,
        minimum_max_vertex_speed=float(motion.min()/m['dt']),last_second_max_vertex_speed=float((motion[-25:]/m['dt']).max()),
        baseline_config_mesh_material_force_and_dt_preserved=True,protected_files_checked=len(protected),protected_files_changed=changes,
        one_second_response=windows,
        scope='Flag L0 repair extension only; neither unchanged paper algorithm nor multilevel VelPro validation')
    (a.run/'repair_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))

if __name__=='__main__':main()
