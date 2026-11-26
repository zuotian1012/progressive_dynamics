"""Replay recorded binding-face directions, preserving native SVD inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.check_psq_step_cap import evaluate_recorded


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('audit',type=Path)
    p.add_argument('--replay-dll',type=Path,required=True)
    p.add_argument('--candidate-dll',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    records=[json.loads(line) for line in a.audit.read_text().splitlines()]
    samples=[dict(frame=r['frame'],iteration=r['iteration'],face=f['face'],values=f['values'])
             for r in records for f in r['faces']]
    values=np.array([r['values'] for r in samples])
    old=evaluate_recorded(values,a.replay_dll.resolve())
    np.testing.assert_allclose(old,values[:,:2],rtol=2e-12,atol=1e-30)
    margin=1.1-64*np.finfo(float).eps*1.1
    near=values[:,2]>=margin
    unique_near={(r['frame'],r['iteration'],r['face']):r for r,b in zip(samples,near) if b}
    # The second face in each record is the minimum of all production
    # directional face caps, including when it duplicates the legacy minimum.
    binding=[r['faces'][1]['face'] for r in records if r['directional_min']<1e-6]
    result=dict(records=len(records),recorded_face_cases=len(samples),
        exact_native_inputs_replayed=True,near_boundary_cases=int(near.sum()),
        unique_near_boundary_directions=len(unique_near),
        tiny_global_cap_binding_faces=dict(Counter(binding)),
        first_near_boundary_sample=next((r for r,b in zip(samples,near) if b),None),
        last_sample=samples[-1],
        limitation='Only the legacy and directional global binding faces are recorded per sample; candidate caps of these faces alone do not predict a new global cap or a converged solve.')
    if a.candidate_dll:
        new=evaluate_recorded(values,a.candidate_dll.resolve())
        assert np.isfinite(new).all() and (new[:,1]>0).all() and (new[:,1]<=1).all()
        changed=new[:,1]!=old[:,1]
        assert np.all(near[changed]),'Unrelated non-boundary cases changed'
        assert np.all(new[changed,1]>old[changed,1])
        f=values[changed,5:11].reshape(-1,2,3).transpose(0,2,1)
        d=values[changed,11:17].reshape(-1,2,3).transpose(0,2,1)
        caps=new[changed,1]
        path_max=0.;path_min=float('inf');roundoff_at_start=0
        if len(f):
            endpoint=np.linalg.svd(f+caps[:,None,None]*d,compute_uv=False)
            assert (endpoint[:,0]<1.1).all() and (endpoint[:,1]>1e-12).all()
            for t in np.linspace(0,1,201):
                s=np.linalg.svd(f+t*caps[:,None,None]*d,compute_uv=False)
                # Native and NumPy SVD can disagree by a few ulps at the
                # already accepted initial boundary. Report this explicitly.
                if t==0:roundoff_at_start=int((s[:,0]>=1.1).sum())
                assert np.all(s[:,0]<=1.1+8*np.spacing(1.1)) and np.all(s[:,1]>1e-12)
                path_max=max(path_max,float(s[:,0].max()));path_min=min(path_min,float(s[:,1].min()))
        selected=np.flatnonzero(changed)
        examples=[]
        for i in sorted(selected,key=lambda k:new[k,1]/old[k,1],reverse=True)[:8]:
            r=samples[i]
            examples.append(dict(frame=r['frame'],iteration=r['iteration'],face=r['face'],
                                 old_cap=float(old[i,1]),candidate_cap=float(new[i,1]),
                                 native_maximum_stretch=float(values[i,2]),direction_norm=float(values[i,4])))
        result['candidate']=dict(improved_cases=int(changed.sum()),unchanged_other_cases=True,
            path_samples_per_case=201,maximum_path_stretch=path_max,minimum_path_sigma=path_min if len(f) else None,
            initial_numpy_native_boundary_disagreements=roundoff_at_start,
            endpoint_strictly_below_limit=True,examples=examples,
            path_check_note='Initial native feasibility retained; NumPy path check allows 8 ulps solely for SVD/roundoff comparison. This is not a change to the production 1.1 limit.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ('first_near_boundary_sample','last_sample')},indent=2))


if __name__=='__main__':main()
