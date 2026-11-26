"""Measure overhang and sample cloth/table intersections for a saved shirt run."""
import argparse
import json
from pathlib import Path
import numpy as np
import ipctk
import trimesh


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment',type=Path)
    args=parser.parse_args();root=args.experiment/'run'
    meta=json.loads((root/'run.json').read_text());assert meta['status']!='running'
    end=meta['completed_steps'];assert end>0
    x=np.load(root/'positions.npy',mmap_mode='r');f=np.load(root/'faces.npy');n=x.shape[1]
    plan=json.loads((args.experiment/'plan.json').read_text());s=plan['stages_physical_steps']
    box=trimesh.creation.box(extents=[1,2,.8]);v=box.vertices+[.65,0,.4]
    faces=np.concatenate([f,np.asarray(box.faces)+n]).astype(np.int32)
    edges=np.unique(np.sort(faces[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0).astype(np.int32)
    cm=ipctk.CollisionMesh(np.concatenate([x[0],v]),edges,faces)
    cm.can_collide=ipctk.make_vertex_patches_filter(np.r_[np.zeros(n,dtype=np.int32),np.ones(len(v),dtype=np.int32)])
    first=None;max_inside=0
    for begin in range(0,end+1,64):
        q=x[begin:min(begin+64,end+1)];below=q[:,:,2]<.799
        where=np.flatnonzero(below.any(axis=1))
        if first is None and len(where):first=begin+int(where[0])
        inside=below&(q[:,:,2]>.001)&(q[:,:,0]>.151)&(q[:,:,0]<1.149)&(np.abs(q[:,:,1])<.999)
        max_inside=max(max_inside,int(inside.sum(axis=1).max()))
    samples=[]
    for k in sorted({min(60,end),min(s['transfer_hold_end'],end),min((s['transfer_hold_end']+s['fold_arc_end'])//2,end),
                     min(s['fold_arc_end'],end),min(s['placement_end'],end),min(s['opened'],end),min(s['retreat_end'],end),end}):
        samples.append(dict(step=k,cloth_table_intersection=bool(ipctk.has_intersections(cm,np.concatenate([x[k],v]))),
            vertices_below_top=int((x[k,:,2]<.799).sum()),minimum_z_m=float(x[k,:,2].min())))
    final=x[end];below=final[:,2]<.799
    result=dict(table_bounds_m=[[.15,1.15],[-1,1],[0,.8]],below_threshold_m=.799,
        completed_steps=end,completed_requested_motion=meta['completed'],first_below_top_step=first,
        first_below_top_time_s=first*meta['dt'] if first is not None else None,
        final_below_top_vertex_count=int(below.sum()),total_vertices=n,final_below_top_vertex_fraction=float(below.mean()),
        all_final_below_top_vertices_beyond_right_edge=bool(np.all(final[below,0]>1.15)),
        final_minimum_z_m=float(final[:,2].min()),maximum_vertex_inside_box_count_with_1mm_margin=max_inside,
        intersection_samples=samples,limitations=['Vertex fractions are not cloth area fractions.',
            'Vertex screening and sampled intersections do not establish continuous collision freedom.',
            'Heights below the finite table outside its footprint are overhang, not fold flatness.'])
    (root/'report').mkdir(exist_ok=True)
    (root/'report/table_extent_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
