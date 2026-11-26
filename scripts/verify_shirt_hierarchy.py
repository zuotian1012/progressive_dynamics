"""Recover and verify the reference 13k -> 55k subdivision correspondence."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import save_npz
from scipy.spatial import cKDTree
from shirt_demo.assets import BUNDLE, read_obj, subdivide


def canonical(faces):
    # Preserve face winding while allowing cyclic permutations and reordered faces.
    m=np.argmin(faces,axis=1)
    faces=np.take_along_axis(faces,(m[:,None]+np.arange(3))%3,axis=1)
    return faces[np.lexsort(faces.T[::-1])]


def main():
    source=BUNDLE/'assets/cloth'
    a,f=read_obj(source/'short-shirt-13767f.obj')
    b,g=read_obj(source/'short-shirt-55068f.obj')
    generated,fg,p=subdivide(a,f)
    distance,fine_to_generated=cKDTree(generated).query(b)
    if len(b)!=len(generated) or len(np.unique(fine_to_generated))!=len(b) or distance.max()>1e-7:
        raise ValueError('Fine asset is not an exact midpoint refinement')
    generated_to_fine=np.argsort(fine_to_generated)
    if not np.array_equal(canonical(g),canonical(generated_to_fine[fg])):
        raise ValueError('Subdivision face connectivity/winding mismatch')
    p=p[fine_to_generated]
    output=Path('output/shirt_reference_hierarchy')
    output.mkdir(parents=True,exist_ok=True)
    save_npz(output/'map_0_1.npz',p)
    for i,(v,faces) in enumerate(((a,f),(b,g))):
        world=v[:,[0,2,1]]*[.001,.001,-.001]+[.667,.015,.93]
        np.savez_compressed(output/f'mesh_{i}.npz',vertices=world,faces=faces)
    record={'passed':True,'vertices':[len(a),len(b)],'faces':[len(f),len(g)],
            'max_rest_error_mm':float(np.max(np.abs(p@a-b))),
            'topology_and_winding_match':True,'partition_of_unity_error':float(np.max(np.abs(np.asarray(p.sum(1)).ravel()-1))),
            'source_sha256':{name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in ('short-shirt-13767f.obj','short-shirt-55068f.obj')},
            'scope':'Exact linear subdivision geometry; not a proof of dynamic equivalence or PSQ.'}
    (output/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
