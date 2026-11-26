"""Audit prepared Scene527 assets, export direct PSQ operators, and plot meshes.

This is static asset verification, not a dynamics or safe-expansion test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import load_npz, save_npz, coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.transform import Rotation
from progressive.prolongation import Hierarchy, frames
from shirt_demo.assets import read_obj


def audit(root):
    manifest = json.loads((root/'manifest.json').read_text())
    source = Path(manifest['source'])
    original, original_faces = read_obj(source)
    transform = manifest['coordinate_transform']
    custom_world_source=transform.get('kind')=='identity'
    expected = original.copy() if custom_world_source else original[:, transform['axis_order']]*transform['axis_scale']+transform['translation_m']
    rests, faces = [], []
    for k in range(len(manifest['levels'])):
        with np.load(root/f'mesh_{k}.npz') as data:
            rests.append(data['vertices']); faces.append(data['faces'])
    last = len(rests)-1
    serialization_error = float(np.max(np.abs(rests[-1]-expected)))
    np.testing.assert_allclose(rests[-1],expected,atol=1e-14,rtol=0)
    # Restore exact source float64 values after harmless native OBJ rounding.
    rests[-1] = expected.copy()
    np.savez_compressed(root/f'mesh_{last}.npz',vertices=expected,faces=faces[-1])
    h = Hierarchy(rests, faces, [load_npz(root/f'map_{k}_{k+1}.npz') for k in range(last)],
                  [np.load(root/f'anchors_{k}_{k+1}.npy') for k in range(last)],
                  [np.load(root/f'anchors_{k}_{last}.npy') for k in range(last)]+[None])
    np.testing.assert_array_equal(faces[-1], original_faces)
    np.testing.assert_array_equal(rests[-1], expected)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['source_sha256']
    # Verify against the source repository's frozen bundle, not just our own hash.
    source_hash = manifest['source_sha256']
    if not custom_world_source:
        bundle = json.loads((source.parents[2]/'bundle.json').read_text())
        assert source_hash == bundle['files'][str(source.relative_to(source.parents[2])).replace('\\','/')]['sha256']
    topology = []
    import ipctk
    contact_checks = []
    for k, (v, f) in enumerate(zip(rests, faces)):
        assert np.isfinite(v).all() and f.min() >= 0 and f.max() < len(v)
        edges, counts = np.unique(np.sort(f[:, [[0,1],[1,2],[2,0]]].reshape(-1,2),axis=1),axis=0,return_counts=True)
        assert counts.max() <= 2, 'Nonmanifold edge'
        assert len(np.unique(np.sort(f,axis=1),axis=0)) == len(f), 'Duplicate triangle'
        area = frames(v,f)[3]
        graph = coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(v),len(v)))
        components = int(connected_components(graph,directed=False,return_labels=False))
        record = dict(level=k, vertices=len(v), faces=len(f), connected_components=components,
                      boundary_edges=int(np.sum(counts==1)), euler_characteristic=int(len(v)-len(edges)+len(f)),
                      minimum_triangle_area_m2=float(area.min()), total_area_m2=float(area.sum()))
        topology.append(record)
        cm=ipctk.CollisionMesh(v,edges.astype(np.int32),f.astype(np.int32))
        intersection=bool(ipctk.has_intersections(cm,v))
        clearance=manifest['levels'][k]['contact_clearance_m']
        contacts=ipctk.NormalCollisions();contacts.build(cm,v,clearance)
        distance=None if contacts.empty() else float(np.sqrt(contacts.compute_minimum_distance(cm,v)))
        contact_checks.append(dict(level=k,self_intersections=intersection,clearance_m=clearance,
                                   minimum_distance_below_clearance_m=distance,
                                   passed=not intersection and (distance is None or distance>clearance)))
        with (root/f'mesh_{k}.obj').open('w',encoding='ascii') as out:
            for p in v: out.write('v '+' '.join(format(float(a),'.17g') for a in p)+'\n')
            for t in f: out.write('f '+' '.join(str(int(a)+1) for a in t)+'\n')
    assert len({(t['connected_components'],t['euler_characteristic']) for t in topology}) == 1
    Q = Rotation.from_rotvec([.4,-.2,.8]).as_matrix()
    translation = np.array([.17,-.23,.31])
    operators = []
    pairs = {(k,k+1):p for k,p in enumerate(h.adjacent)}
    pairs.update({(k,last):p for k,p in enumerate(h.to_finest[:-1])})
    for (a,b), p in sorted(pairs.items()):
        U = p.U
        assert np.isfinite(U.data).all() and U.data.min() >= -1e-12
        unity = float(np.max(np.abs(np.asarray(U.sum(1)).ravel()-1)))
        rest_error = float(np.max(np.abs(p.position(rests[a])-rests[b])))
        rigid_error = float(np.max(np.abs(p.position(rests[a]@Q.T+translation)-(rests[b]@Q.T+translation))))
        assert max(unity,rest_error,rigid_error) < 1e-11
        save_npz(root/f'map_{a}_{b}.npz',U)
        np.savez_compressed(root/f'psq_{a}_{b}.npz',anchors=p.anchors,gamma=p.gamma,tangent=p.tangent,inv_dm=p.inv_dm)
        operators.append(dict(source_level=a,target_level=b,shape=list(U.shape),nnz=U.nnz,
                              partition_of_unity_error=unity,rest_error_m=rest_error,rigid_motion_error_m=rigid_error,
                              max_offset_m=float(np.linalg.norm(p.offset(rests[a]),axis=1).max())))
    report = dict(source_sha256=source_hash,source_matches_frozen_bundle=not custom_world_source,
                  source_matches_declared_input=True,custom_world_source=custom_world_source,
                  native_text_roundtrip_error_m_restored_to_exact_source=serialization_error,
                  finest_vertex_order_faces_winding_and_world_positions_exact=True,
                  topology=topology,operators=operators,
                  mapping_checks_passed=True,
                  rest_contact_checks=contact_checks,
                  rest_self_contact_passed=all(c['passed'] for c in contact_checks),
                  passed=all(c['passed'] for c in contact_checks),
                  scope='Static assets only; no dynamic collision safety, strain correction, gripper preflight or simulation claimed')
    (root/'audit.json').write_text(json.dumps(report,indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,len(rests),figsize=(4*len(rests),5),squeeze=False)
    lo,hi = rests[-1].min(0),rests[-1].max(0)
    for k,(v,f,ax) in enumerate(zip(rests,faces,axes[0])):
        ax.triplot(v[:,0],v[:,1],f,lw=.12 if len(f)>10000 else .25,color='#195680')
        ax.set(xlim=(lo[0]-.03,hi[0]+.03),ylim=(lo[1]-.03,hi[1]+.03),aspect='equal',
               title=f'L{k}: {len(v):,} vertices\n{len(f):,} triangles',xlabel='x (m)',ylabel='y (m)')
    fig.suptitle(f'Scene527 | {len(faces[-1]):,}-face declared finest mesh | intrinsic PSQ hierarchy')
    fig.tight_layout();fig.savefig(root/'hierarchy.png',dpi=180);plt.close(fig)
    hashes = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(root.iterdir()) if p.is_file() and p.name!='checksums.json'}
    (root/'checksums.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('assets',type=Path)
    audit(parser.parse_args().assets)
