"""Recompute interpolation proposals and check actual full-energy trial records."""
import argparse,json,re
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--check-converged-energy',action='store_true')
    p.add_argument('--accepted-prefix',action='store_true',help='Audit only Newton solves belonging to committed frames')
    a=p.parse_args()
    trials={};shrinks=[];restored=set()
    retries={}; rejected_probes=set(); analytic_stops=set(); rounded_stops=set(); retained_directions=set()
    meta=json.loads((a.run/'run.json').read_text())
    search_budget=int(meta['config']['line_search']['max_iter'])
    assert search_budget>0
    # Shirt event runs keep the native feature declarations in a separate manifest.
    provenance_path=a.run/'runtime_provenance.json'
    if meta.get('event_fallback_enabled') and provenance_path.exists():
        provenance=json.loads(provenance_path.read_text())
        assert provenance['dll_sha256']==meta['event_runtime_sha256']
        for field in ('checked_search_exhaustion','exhausted_search_acceptance'):
            meta.setdefault(field,provenance[field])
    rms_control=meta.get('bunny_newton_norm')=='free_vertex_rms_3'
    rms_counts={}
    direct_errors=[]
    solver_direction=None
    for line in a.run.with_suffix('.log').open(encoding='utf-8-sig',errors='replace'):
        if 'BUNNY_EVENT_TRIGGER ' in line:
            assert meta.get('event_fallback_enabled')
            event=dict(re.findall(r'(\w+)=([^\s]+)',line))
            key=(int(event['frame']),int(event['newton']))
            base,retained=float(event['E0']),float(event['retained'])
            assert np.isfinite([base,retained]).all()
            assert abs(base-retained)<=64*np.finfo(float).eps*max(abs(base),1e-30)
            retained_directions.add(key)
            if event['reason']=='representable':rounded_stops.add(key)
        if a.accepted_prefix:
            header=re.search(r'Frame (\d+) Newton Iteration|\bframe=(\d+)|>>> Begin Frame: (\d+)',line)
            if header and int(header[1] or header[2] or header[3])>meta['completed_steps']:
                break
        if 'BUNNY_DIRECT_SOLVE ' in line:
            solve=dict(re.findall(r'(\w+)=([^\s]+)',line))
            error=float(solve['backward_error'])
            assert int(solve['dofs'])>=3*meta['vertices'] and float(solve['min_pivot'])>0
            assert np.isfinite(error) and 0<=error<=1e-12
            direct_errors.append(error)
        if any(tag in line for tag in ('PSQ_PREVIEW_STOP frame=', 'FLAG_PSQ_RESCUE frame=', 'FLAG_PSQ_UNRESOLVED_RESCUE frame=')):
            data=dict(re.findall(r'(\w+)=([^\s]+)',line))
            retained_directions.add((int(data['frame']),int(data['newton'])))
        if 'BUNNY_PREVIEW_DERIVATIVE frame=' in line:
            data=dict(re.findall(r'(\w+)=([^\s]+)',line))
            assert meta.get('frozen_direction_derivative')
            if float(data['slope'])>float(data['bound']):
                analytic_stops.add((int(data['frame']),int(data['newton'])))
        if 'BUNNY_PREVIEW_STAGNATION frame=' in line:
            data=dict(re.findall(r'(\w+)=([^\s]+)',line))
            assert meta.get('representable_preview') and meta['level']<7
            rounded_stops.add((int(data['frame']),int(data['newton'])))
        if 'BUNNY_SEARCH_RETRY ' in line or 'BUNNY_FROZEN_SEARCH ' in line or 'PSQ_PREVIEW_REJECT frame=' in line:
            data=dict(re.findall(r'(\w+)=([^\s]+)',line))
            direction=(int(data['frame']),int(data['newton']))
            if 'BUNNY_SEARCH_RETRY ' in line or 'BUNNY_FROZEN_SEARCH ' in line:
                retries.setdefault(direction,[]).append(data)
            else:
                rejected_probes.add(direction)
        if rms_control and ' Newton Iteration ' in line:
            header=re.search(r'Frame (\d+) Newton Iteration (\d+)',line)
            if header:
                solver_direction=tuple(map(int,header.groups()))
        if rms_control and 'BUNNY_RMS frame=' in line:
            v=dict(re.findall(r'(\w+)=([^\s]+)',line))
            # The manager's report header carries the actual frame, including
            # in cached libuipc builds whose pre_newton() drops its frame argument.
            assert solver_direction is not None
            frame,iteration=solver_direction
            assert int(v['newton'])==iteration and int(v['frame']) in (0,frame)
            residual,tolerance=float(v['velocity']),float(v['tolerance'])
            assert np.isfinite(residual) and tolerance==meta['config']['newton']['velocity_tol']
            assert int(v['free'])==meta['vertices'] and (frame,iteration) not in rms_counts
            previous=rms_counts[(frame,iteration-1)] if iteration else 0
            count=min(3,previous+1) if residual<=tolerance else 0
            assert count==int(v['consecutive'])
            rms_counts[frame,iteration]=count
        if a.check_converged_energy and 'PSQ_CONVERGED_RESTORE ' in line:
            values=dict(re.findall(r'(\w+)=([^\s]+)',line))
            key=int(values['frame']),int(values['newton'])
            assert key not in restored
            retained,base,noise=map(float,(values['retained'],values['E0'],values['noise']))
            assert np.isfinite(retained) and abs(retained-base)<=noise
            restored.add(key)
        if 'FLAG_SEARCH_TRIAL ' not in line and 'FLAG_SEARCH_SHRINK ' not in line:continue
        raw=dict(re.findall(r'(\w+)=([^\s]+)',line));d={k:(v=='true' if v in ('true','false') else float(v)) for k,v in raw.items()}
        key=(int(d['frame']),int(d['newton']),int(d['search']))
        if 'FLAG_SEARCH_TRIAL ' in line:assert key not in trials;trials[key]=d
        else:shrinks.append((key,d))
    interpolated=[]
    for key,d in shrinks:
        t=trials[key];assert t['E']>t['E0'];np.testing.assert_allclose(d['delta'],t['E']-t['E0'],rtol=1e-12,atol=0)
        alpha=d['alpha'];expected=.5*alpha
        if d['previous_alpha']>alpha:
            previous=next(trials[(key[0],key[1],j)] for j in range(key[2]-1,-1,-1) if np.isfinite(trials[(key[0],key[1],j)]['E']))
            assert d['previous_alpha']==previous['alpha']
            np.testing.assert_allclose(d['previous_delta'],previous['E']-previous['E0'],rtol=1e-12,atol=0)
            q=(d['previous_delta']/d['previous_alpha']-d['delta']/alpha)/(d['previous_alpha']-alpha)
            g=d['delta']/alpha-q*alpha
            if np.isfinite(q) and np.isfinite(g) and q>0 and g<0:expected=np.clip(-g/(2*q),.1*alpha,.5*alpha)
        np.testing.assert_allclose(d['next'],expected,rtol=1e-12,atol=0)
        assert .1*alpha<=d['next']<=.5*alpha
        following=trials.get((key[0],key[1],key[2]+1))
        if following:
            assert following['alpha']==d['next']
            if d['next']<.5*alpha and following['E']<=following['E0']:
                interpolated.append(dict(frame=key[0],newton=key[1],search=key[2]+1,alpha=following['alpha'],energy_drop=following['E0']-following['E']))
    groups={}
    for key,t in trials.items():
        assert 0<=key[2]<search_budget and 0<t['alpha']<=t['cap']
        groups.setdefault(key[:2],[]).append(key[2])
    for searches in groups.values():assert sorted(searches)==list(range(len(searches))) and len(searches)<=search_budget
    if rms_control:
        assert rms_counts.keys()==groups.keys(), 'Missing RMS record for a Newton direction'
        for key in groups:
            if trials[(*key,0)]['converged']:
                assert rms_counts[key]==3, key
    if meta.get('checked_search_exhaustion'):
        assert rejected_probes <= (retries.keys() | analytic_stops), 'Rejected preview without a verified update or non-descent certificate'
    assert rounded_stops <= retries.keys()
    for key, attempts in retries.items():
        last=trials[(*key,max(groups[key]))]
        assert not last['converged'] and last['E']>last['E0'] and len(groups[key])==search_budget
        previous=last['alpha']
        for i,record in enumerate(attempts):
            alpha,cap,base,energy=map(float,(record['alpha'],record['cap'],record['E0'],record['E']))
            assert i==int(record['retry']) and 0<alpha<previous<=cap==last['cap'] and base==last['E0']
            if i:
                assert alpha==previous/2
            noise=64*np.finfo(float).eps*max(abs(base),1e-30)
            identical=record.get('identical')=='true'
            if 'identical' in record:
                assert meta.get('representable_preview') and meta['level']<7
                assert identical==(key in rounded_stops and i==len(attempts)-1)
            else:
                assert i<24 and noise==float(record['noise'])
            required_drop=0. if meta.get('exhausted_search_acceptance')=='strict_decrease' else noise
            accepted=np.isfinite(energy) and base-energy>required_drop and not identical
            assert accepted==(record['accepted']=='true')==(key not in rounded_stops and i==len(attempts)-1)
            previous=alpha
    checked=0
    if a.check_converged_energy:
        handled=retained_directions | restored | analytic_stops | rounded_stops | retries.keys()
        for key, searches in groups.items():
            trial=trials[(*key,max(searches))]
            assert (np.isfinite(trial['E']) and trial['E']<=trial['E0']) or key in handled, (
                'Newton continued after an unchecked trial',key)
            if trial['converged']:
                checked+=1
                assert (np.isfinite(trial['E']) and trial['E']<=trial['E0']) or key in restored, key
        records=json.loads((a.run/'steps.json').read_text())
        expected_trials=sum(row['native']['line_search_trials'] for row in records)
        checked_trials=len(trials)+sum(map(len,retries.values()))
        assert checked_trials==expected_trials, (checked_trials,expected_trials)
        for row in records:
            native=row['native']
            if native['converged']:
                final=(native['frame'],native['newton_iterations']-1)
                assert final in groups and trials[(*final,max(groups[final]))]['converged'], final
    assert trials
    result=dict(passed=True,newton_searches=len(groups),maximum_trials=max(map(len,groups.values())),
        total_trials=len(trials),proposals_verified=len(shrinks),interpolated_following_trials_with_energy_decrease=len(interpolated),examples=interpolated[:10],
        converged_energy_checks=checked,restored_converged_iterates=len(restored),
        recovered_exhausted_searches=len(retries)-len(rounded_stops),representable_search_stops=len(rounded_stops),
        extra_energy_trials=sum(map(len,retries.values())))
    if rms_control:
        result.update(rms_decrements_checked=len(rms_counts),free_vertices=meta['vertices'])
    if meta.get('bunny_linear_solver')=='eigen_sparse_ldlt':
        assert len(direct_errors)==len(groups)
        result.update(direct_linear_solves=len(direct_errors),max_linear_backward_error=max(direct_errors))
    else:
        assert not direct_errors
    output='line_search_prefix_audit.json' if a.accepted_prefix else 'line_search_audit.json'
    (a.run/output).write_text(json.dumps(result,indent=2));print(result)

if __name__=='__main__':main()
