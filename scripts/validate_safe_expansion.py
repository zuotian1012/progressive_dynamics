"""Reproducible safe refinement validation, with saved failures and geometry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import load_npz
import ipctk
from progressive.expansion_history import ExpansionHistory
from progressive.prolongation import ShellProlongation
from progressive.safe_expansion import initialize
from progressive.refinement_contact import ExpansionOptions,RefinementFailure,collision_mesh,minimum_gap
from progressive.runtime import numerical_threads


def validate(assets,output,case='rest'):
    if output.exists() and any(output.iterdir()):raise ValueError('Refusing nonempty validation output')
    output.mkdir(parents=True,exist_ok=True)
    a=np.load(assets/'mesh_0.npz');b=np.load(assets/'mesh_1.npz')
    coarse,cf=a['vertices'],a['faces'];fine,ff=b['vertices'],b['faces']
    P=ShellProlongation(coarse,fine,cf,load_npz(assets/'map_0_1.npz'),np.load(assets/'anchors_0_1.npy'))
    record=json.loads((assets/'expansion_0_1.json').read_text())
    if case=='self-contact':
        # Two components of the same deformable collision mesh. The lower
        # component is close only in the deformed state, not in its rest mesh.
        plane=np.array([[-2.,-1.,-.2],[2.,-1.,-.2],[2.,2.,-.2],[-2.,2.,-.2]])
        pf=np.array([[0,1,2],[0,2,3]]);nf,nc=len(fine),len(coarse)
        record['fine_vertex_count']+=4;record['coarse_ids']+=list(range(nf,nf+4))
        fine=np.vstack((fine,plane));ff=np.vstack((ff,pf+nf))
        coarse=np.vstack((coarse,plane));cf=np.vstack((cf,pf+nc))
        x=coarse.copy();x[:nc,2]=0;x[nc:,2]=-.002
        naive=np.vstack((P.position(x[:nc]),x[nc:]))
    else:x=coarse.copy();naive=P.position(x)
    history=ExpansionHistory(record,coarse,cf,fine,ff)
    metadata={'case':case,'assets':str(assets.resolve()),'accepted':False,
              'history_sha256':hashlib.sha256((assets/'expansion_0_1.json').read_bytes()).hexdigest(),
              'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('progressive').glob('*.py')}}
    mesh=collision_mesh(fine,ff)
    metadata['naive_intersections']=bool(ipctk.has_intersections(mesh,naive))
    metadata['naive_gap_lower_bound']=minimum_gap(mesh,naive,.001)
    try:
        result,report=initialize(history,x,naive,on_batch=lambda r:print(
            f"color {r['color']+1}/{len(history.colors)}: {r['splits']} splits, gap={r['gap_after']:.6g}, relaxation stages={len(r['relaxations'])}",flush=True))
        metadata.update(accepted=True,report=report,safe_intersections=bool(ipctk.has_intersections(mesh,result)),
                        safe_gap_lower_bound=minimum_gap(mesh,result,.001))
        assert not metadata['safe_intersections'] and metadata['safe_gap_lower_bound']>.0002
        if case=='self-contact':assert metadata['naive_intersections']
        np.savez_compressed(output/'states.npz',coarse=x,coarse_faces=cf,naive=naive,safe=result,fine_faces=ff)
        make_plot(x,cf,naive,result,ff,output/'comparison.png')
    except RefinementFailure as error:
        metadata['report']=error.report
        if hasattr(error,'failed_state'):
            np.savez_compressed(output/'failure_state.npz',positions=error.failed_state,rest=error.failed_rest,
                                faces=error.failed_faces,cloth_vertices=error.failed_cloth_vertices)
        raise
    finally:
        (output/'validation.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def make_plot(coarse,cf,naive,safe,ff,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained',sharex=True,sharey=True)
    all_positions=np.vstack((coarse,naive,safe))
    lo,hi=all_positions.min(axis=0),all_positions.max(axis=0)
    contact_fixture=hi[0]-lo[0]>3
    if contact_fixture:
        lo=np.array([-.6,-.1,min(-.025,lo[2]-.005)])
        hi=np.array([.6,1.1,max(.025,hi[2]+.005)])
    else:
        margin=np.maximum((hi-lo)*.05,.001);lo-=margin;hi+=margin
    for i,(x,f,title) in enumerate(((coarse,cf,'Coarse input'),(naive,ff,'Direct prolongation'),(safe,ff,'Safe refinement'))):
        ax=axes[i];edges=np.unique(np.sort(f[:,[[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0)
        segments=x[edges][:,:,[0,2]].copy();segments[:,:,1]*=1000
        ax.add_collection(LineCollection(segments,colors='#226b9e',linewidths=.7))
        if contact_fixture:ax.axhline(-2,color='#b74930',ls='--',label='Lower cloth: z = -2 mm');ax.legend(fontsize=8,loc='lower right')
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[2]*1000,hi[2]*1000),title=title,xlabel='x (m)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('z (mm)');fig.suptitle('Side projection (y suppressed); vertical scale exaggerated')
    fig.savefig(path,dpi=150);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--case',choices=('rest','self-contact'),default='rest')
    a=p.parse_args()
    with numerical_threads(1):validate(a.assets,a.output,a.case)
