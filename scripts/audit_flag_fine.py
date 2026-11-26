"""Check continuation provenance, isolated files, and every exact-gradient retry."""
from pathlib import Path
import argparse, hashlib, json, re, subprocess, sys
ROOT=Path(__file__).resolve().parents[1]

def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args()
    m=json.loads((a.run/'run.json').read_text())
    assert m['completed'] and m['completed_steps']==500
    if m.get('lower_run'):
        lower=Path(m['lower_run']);old=json.loads((lower/'run.json').read_text())
        assert old['completed'] and m['level']==old['level']+1 and m['method']=='velpro'
        assert m['lower_sha256']==digest(lower/'positions.npy')
        assert m['map_sha256']==digest(Path(m['assets'])/f'map_{old["level"]}_{m["level"]}.npz')
        assert m['lower_backend_sha256']==old['library_sha256']['uipc_backend_cuda.dll']
        if m.get('per_iteration_acceptance_required'):
            assert old.get('per_iteration_acceptance_required')
            assert json.loads((lower/'continuation_audit.json').read_text())['passed']
    else:
        assert m['level']==0
        old=json.loads((ROOT/'output/flag_L0_repair_20261118/rescue_full/run.json').read_text())
    cfg=json.loads(json.dumps(old['config']))
    cfg['newton']['max_iter']=m['config']['newton']['max_iter']
    cfg['linear_system']['fem_preconditioner']=m['config']['linear_system']['fem_preconditioner']
    assert cfg==m['config']
    for k in ['assets','material','dt','steps']:assert m[k]==old[k],k
    assert m['adaptive_line_search'] and m['non_descent_probe_refinement'] and m['initial_non_descent_rescue']
    assert not m['exact_offset_gradient'] and not m['tiny_probe']
    prov=json.loads((ROOT/'output/flag_L3_fix_20261118/rescue_build/provenance.json').read_text())
    for source,h in prov['source_hashes'].items():assert digest(source)==h,source
    assert m['library_sha256']['uipc_backend_cuda.dll']==prov['dll_sha256']
    assert digest(Path(m['native_package'])/'uipc/_native/uipc_backend_cuda.dll')==prov['dll_sha256']
    protected={}
    for rel in ['flag_L0_repair_20261118/protected_manifest.json','flag_L1_fix_20261118/protected_additional.json','flag_L2_fix_20261118/protected_additional.json','flag_fine_20261118/protected_additional.json']:
        protected.update(json.loads((ROOT/'output'/rel).read_text()))
    changed=[p for p,h in protected.items() if not Path(p).is_file() or digest(p)!=h]
    assert not changed,changed
    triggers={};accepts={};restored_bases={};refined=False;search=False
    with a.run.with_suffix('.log').open(encoding='utf-8-sig',errors='replace') as log:
        for line in log:
            assert 'Line Search Exits with Max Iteration' not in line, line
            assert 'PSQ_PREVIEW_REJECT frame=' not in line, line
            refined |= 'FLAG_PROBE_REFINE ' in line
            search |= 'FLAG_SEARCH_TRIAL ' in line
            if 'FLAG_PSQ_' not in line:continue
            d=dict(re.findall(r'(\w+)=([^\s]+)',line))
            if 'FLAG_PSQ_UNRESOLVED_RESCUE ' in line:
                key=(int(d['frame']),int(d['retry']));assert key not in triggers
                assert int(d['retry'])==int(d['newton'])+1 and d['consistent']=='false'
                assert abs(float(d['E_retained'])-float(d['E_base']))<=float(d['noise'])
                triggers[key]='unresolved'
                restored_bases[key]=(float(d['E_base']),float(d['noise']))
            elif 'FLAG_PSQ_RESCUE frame=' in line:
                key=(int(d['frame']),1);assert key not in triggers;triggers[key]='first_iteration'
            elif 'FLAG_PSQ_RESCUE_ACCEPT ' in line:
                key=(int(d['frame']),int(d['newton']));assert key not in accepts
                assert float(d['E1'])<=float(d['E0']) and 0<float(d['alpha'])<=float(d['ccd'])<=1
                if key in restored_bases:
                    base,noise=restored_bases[key];assert abs(float(d['E0'])-base)<=noise
                accepts[key]=float(d['E1'])<float(d['E0'])
    assert triggers.keys()==accepts.keys()
    for script,needed in [('audit_flag_refined_probes.py',refined),('audit_flag_fine_search.py',search)]:
        if needed:subprocess.run([sys.executable,str(ROOT/'scripts'/script),str(a.run)],check=True)
    audit=json.loads((a.run/'audit.json').read_text())
    assert audit['accepted_prefix_checks_passed'] and audit['completed_requested_run'] and not audit['zero_motion_steps']
    assert audit['velocity_readback_error']==0 and audit['position_injection_error']==0
    if m['level']<9:
        preview=json.loads((a.run/'report/non_descent_audit.json').read_text())
        assert preview['all_accepted_preview_evidence_verified'] and not preview['preview_at_first_newton_frames']
    else:assert m['native_steps']==500 and m['preview_steps']==0
    result=dict(passed=True,level=m['level'],protected_files_checked=len(protected),protected_files_changed=changed,
        rescue_events=len(triggers),unresolved_rescues=sum(v=='unresolved' for v in triggers.values()),
        strict_energy_decreases=sum(accepts.values()),all_rescues_energy_nonincreasing=True,
        exact_velocity_transfer=True,positions_not_overwritten=True,same_physics_and_tolerances=True,no_unhandled_line_search_exhaustion=True)
    (a.run/'continuation_audit.json').write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
