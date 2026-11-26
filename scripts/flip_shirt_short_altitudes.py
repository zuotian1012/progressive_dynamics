"""Improve local triangle altitudes with quality-gated interior edge flips."""
import argparse,json
from pathlib import Path
import numpy as np
import ipctk
from improve_shirt_finest_mesh import SurfaceProjector,metrics


def altitude(tri):
    area2=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)
    return area2/np.linalg.norm(tri[:,[1,2,0]]-tri,axis=2).max(1)


def improve(v,f,reference,passes=12):
    f=f.copy();records=[]
    for iteration in range(passes):
        directed=f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2)
        sorted_edges=np.sort(directed,axis=1)
        edges,inv,counts=np.unique(sorted_edges,axis=0,return_inverse=True,return_counts=True)
        order=np.argsort(inv,kind='stable');starts=np.r_[0,np.cumsum(counts)[:-1]]
        interior=np.flatnonzero(counts==2);i0=order[starts[interior]];i1=order[starts[interior]+1]
        fi=i0//3;fj=i1//3;a=directed[i0,0];b=directed[i0,1]
        c=f[fi,(i0%3+2)%3];d=f[fj,(i1%3+2)%3]
        assert np.all(directed[i1,0]==b) and np.all(directed[i1,1]==a)
        nf=np.stack([c,d,b],axis=1);ng=np.stack([d,c,a],axis=1)
        old=np.minimum(altitude(v[f[fi]]),altitude(v[f[fj]]))
        new=np.minimum(altitude(v[nf]),altitude(v[ng]))
        t0=v[f[fi]];t1=v[f[fj]];tn=v[nf];tg=v[ng]
        norm=np.cross(t0[:,1]-t0[:,0],t0[:,2]-t0[:,0])+np.cross(t1[:,1]-t1[:,0],t1[:,2]-t1[:,0])
        nn=np.cross(tn[:,1]-tn[:,0],tn[:,2]-tn[:,0]);ngn=np.cross(tg[:,1]-tg[:,0],tg[:,2]-tg[:,0])
        valid=(old<.0022)&(new>old*1.05)&(np.sum(nn*norm,axis=1)>0)&(np.sum(ngn*norm,axis=1)>0)&(c!=d)
        ids=np.flatnonzero(valid)
        if not len(ids):break
        # Bound the sampled deviation of new faces from the original surface.
        samples=np.concatenate([tn[ids].mean(1),tg[ids].mean(1),(v[c[ids]]+v[d[ids]])*.5])
        error=np.linalg.norm(samples-reference.project(samples),axis=1).reshape(3,-1).max(0)
        ids=ids[error<.0005]
        ids=ids[np.argsort(-(np.minimum(new[ids],.0022)-old[ids]))]
        present={tuple(e) for e in edges};used=set();changed=0
        for k in ids:
            if int(fi[k]) in used or int(fj[k]) in used:continue
            edge=tuple(sorted([int(c[k]),int(d[k])]))
            if edge in present:continue
            f[fi[k]]=nf[k];f[fj[k]]=ng[k]
            present.add(edge);present.discard(tuple(sorted([int(a[k]),int(b[k])])))
            used.update([int(fi[k]),int(fj[k])]);changed+=1
        records.append(dict(pass_index=iteration,flips=changed));print(records[-1],flush=True)
        if not changed:break
    return f,records


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    with np.load(a.source) as m:v=m['vertices'];f0=m['faces'].astype(np.int32)
    with np.load(a.reference) as m:rv=m['vertices'];rf=m['faces'].astype(np.int32)
    project=SurfaceProjector(rv,rf);f,history=improve(v,f0,project)
    edges,counts=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    oldedges,oldcounts=np.unique(np.sort(f0[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
    assert counts.max()==2 and len(edges)==len(oldedges)
    np.testing.assert_array_equal(edges[counts==1],oldedges[oldcounts==1])
    assert len(np.unique(np.sort(f,axis=1),axis=0))==len(f)
    cm=ipctk.CollisionMesh(v,edges.astype(np.int32),f)
    report=metrics(v,f,edges,cm,rv,rf);report.update(flips=history,boundary_edges_preserved=True,
        boundary_positions_preserved=bool(np.array_equal(v[np.unique(edges[counts==1])],rv[np.unique(edges[counts==1])])) )
    np.savez_compressed(a.output/'mesh.npz',vertices=v,faces=f)
    with (a.output/'mesh.obj').open('w') as stream:
        for xyz in v:stream.write('v '+' '.join(f'{x:.17g}' for x in xyz)+'\n')
        for face in f:stream.write('f '+' '.join(str(int(i)+1) for i in face)+'\n')
    (a.output/'metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
