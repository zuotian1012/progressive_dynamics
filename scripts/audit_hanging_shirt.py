"""Audit accepted hanging-shirt states, with sampled self-intersection checks."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import ipctk


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();root=a.run
    meta=json.loads((root/'run.json').read_text());mesh=np.load(root/'mesh.npz')
    rest,faces,pins=mesh['vertices'],mesh['faces'],mesh['pins']
    x=np.load(root/'positions.npy',mmap_mode='r')[:meta['completed_steps']+1]
    records=json.loads((root/'steps.json').read_text())
    edges=np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    cm=ipctk.CollisionMesh(rest,edges,faces.astype(np.int32))
    samples=[]
    for k in sorted(set(range(0,len(x),60))|{len(x)-1}):
        samples.append(dict(step=k,self_intersection=bool(ipctk.has_intersections(cm,np.asarray(x[k])))))
    source=Path(meta['source_operator'])
    with np.load(source) as original,np.load(root/'operator.npz') as transformed:
        R=np.array(meta['rotation']);o=np.array(meta['origin']);s=np.array(meta['translation'])
        transform_error=max(float(np.max(np.abs((original[key]-o)@R.T+s-transformed[key]))) for key in ['coarse_rest','fine_rest'])
        unchanged_maps=all(np.array_equal(original[k],transformed[k]) for k in ['coarse_faces','fine_faces','U_indptr','U_indices','U_data','anchors'])
    audit=dict(completed=meta['completed'],finite_accepted_states=bool(np.isfinite(x).all()),
        pinned_vertices=int(pins.sum()),max_pin_error_m=float(np.max(np.abs(x[:,pins]-rest[pins]))),
        source_operator_hash_unchanged=hashlib.sha256(source.read_bytes()).hexdigest()==meta['source_operator_sha256'],
        rigid_transform_error_m=transform_error,topology_maps_and_anchors_unchanged=unchanged_maps,
        accepted_step_count_matches=len(records)==len(x)-1==meta['completed_steps'],
        accepted_newton_limit_steps=sum(bool(r['native']['hit_newton_limit']) for r in records),
        accepted_unverified_search_limit_steps=sum(bool(r['native']['hit_line_search_limit']) and not r['native']['preview_non_descent'] for r in records),
        maximum_stretch=max(r['maximum_stretch'] for r in records),
        self_intersection_samples=samples,
        semi_implicit_preview=meta.get('semi_implicit_preview',False),
        interpretation='Native convergence flags include semi-implicit stopping when enabled; no fine level dynamics or continuous collision-freedom certification.')
    audit['basic_checks_passed']=bool(audit['finite_accepted_states'] and audit['max_pin_error_m']<1e-10
        and audit['source_operator_hash_unchanged'] and unchanged_maps and transform_error<1e-12
        and audit['accepted_step_count_matches'] and audit['accepted_newton_limit_steps']==0
        and audit['accepted_unverified_search_limit_steps']==0 and audit['maximum_stretch']<=1.1+1e-9
        and not any(s['self_intersection'] for s in samples))
    (root/'audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
