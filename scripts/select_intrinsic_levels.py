"""Select valid levels without remeshing, composing U and reversible histories."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import load_npz,save_npz,eye
from progressive.expansion_history import ExpansionHistory


def select(source,output,levels):
    manifest=json.loads((source/'manifest.json').read_text())
    assert levels==sorted(set(levels)) and len(levels)>=2
    assert levels[-1]==len(manifest['levels'])-1, 'Retain original finest mesh'
    assert all(manifest['levels'][k]['rest_thickness_feasible'] for k in levels)
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    geometries=[]
    for i,k in enumerate(levels):
        shutil.copyfile(source/f'mesh_{k}.npz',output/f'mesh_{i}.npz')
        with np.load(output/f'mesh_{i}.npz') as d:geometries.append((d['vertices'],d['faces']))
    histories=[]
    for i,(a,b) in enumerate(zip(levels,levels[1:])):
        U=eye(len(geometries[i][0]),format='csr')
        for k in range(a,b):U=load_npz(source/f'map_{k}_{k+1}.npz')@U
        save_npz(output/f'map_{i}_{i+1}.npz',U)
        # Native export includes adjacent and global-finest anchor triangles.
        shutil.copyfile(source/f'anchors_{a}_{b}.npy',output/f'anchors_{i}_{i+1}.npy')
        record=json.loads((source/f'expansion_{b-1}_{b}.json').read_text())
        for k in reversed(range(a,b-1)):
            lower=json.loads((source/f'expansion_{k}_{k+1}.json').read_text())
            ids=np.asarray(record['coarse_ids'],int)
            for patch in lower['collapses']:
                p=copy.deepcopy(patch)
                for field in ('survivor','removed'):p[field]=int(ids[p[field]])
                for field in ('ids','pre_faces','post_faces'):p[field]=ids[np.asarray(p[field],int)].tolist()
                record['collapses'].append(p)
            record['coarse_ids']=ids[lower['coarse_ids']].tolist()
        name=f'expansion_{i}_{i+1}.json'
        (output/name).write_text(json.dumps(record,separators=(',',':')))
        history=ExpansionHistory(record,*geometries[i],*geometries[i+1])
        histories.append(dict(history.summary(),file=name,sha256=hashlib.sha256((output/name).read_bytes()).hexdigest()))
    for i,k in enumerate(levels[:-1]):
        shutil.copyfile(source/f'anchors_{k}_{levels[-1]}.npy',output/f'anchors_{i}_{len(levels)-1}.npy')
    result=copy.deepcopy(manifest)
    result.update(levels=[dict(manifest['levels'][k],level=i) for i,k in enumerate(levels)],
                  targets_fine_to_coarse=[manifest['levels'][k]['faces'] for k in reversed(levels[:-1])],
                  expansion_histories=histories,nonintersecting_rest_hierarchy=True,
                  thickness_feasible_rest_hierarchy=True,
                  selection={'source':str(source),'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
                             'retained_source_levels':levels,'reason':'Exclude statically infeasible simulation levels; compose intrinsic U and preserve every collapse patch',
                             'intermediate_expansion_states_collision_safe':False},
                  limitation='Static selected levels validated; merged expansion histories reconstruct topology/rest, but dynamic safe expansion remains untested')
    (output/'manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--levels',type=int,nargs='+',required=True)
    a=p.parse_args();select(a.source,a.output,a.levels)
