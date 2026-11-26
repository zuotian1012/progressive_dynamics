"""Export audited PSQ operator data for a future native libuipc constitution.

This does not install a backend or advance a simulation. Proxies are energy
evaluation buffers, never additional dynamic/contact vertices.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy.sparse import load_npz, eye, csr_matrix
from progressive.prolongation import ShellProlongation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--level', type=int, required=True)
    p.add_argument('--assets', type=Path, default=Path('output/bunny_paper_assets'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--fine-level',type=int,default=7)
    a = p.parse_args()
    if not 0 <= a.level <= a.fine_level:
        p.error('Require 0 <= level <= fine-level')
    a.output.mkdir(parents=True, exist_ok=False)
    fine = np.load(a.assets/f'mesh_{a.fine_level}.npz')
    coarse = np.load(a.assets/f'mesh_{a.level}.npz')
    U = eye(len(fine['vertices']),format='csr')
    for k in range(a.fine_level-1,a.level-1,-1): U = U @ load_npz(a.assets/f'map_{k}_{k+1}.npz')
    if a.level==a.fine_level:
        anchors=np.full(len(fine['vertices']),-1,dtype=np.int32)
        for i,face in enumerate(fine['faces']):anchors[face]=i
        if np.any(anchors<0):raise ValueError('Isolated finest vertex')
    else:anchors=np.load(a.assets/f'anchors_{a.level}_{a.fine_level}.npy')
    transfer = ShellProlongation(coarse['vertices'],fine['vertices'],coarse['faces'],U,anchors)
    U = transfer.U
    payload = dict(coarse_rest=coarse['vertices'], coarse_faces=coarse['faces'].astype(np.int32),
        fine_rest=fine['vertices'], fine_faces=fine['faces'].astype(np.int32),
        U_indptr=U.indptr.astype(np.int32), U_indices=U.indices.astype(np.int32), U_data=U.data,
        U_shape=np.asarray(U.shape,np.int32), anchors=transfer.anchors.astype(np.int32),
        gamma=transfer.gamma,tangent=transfer.tangent,inv_dm=transfer.inv_dm)
    bundle = a.output/'operator.npz'
    np.savez(bundle,**payload)
    # One round-trip check, no Newton/time stepping, using an already accepted
    # short native state where available; this checks transport, not physics.
    state = coarse['vertices']
    if a.level == 0 and a.assets.resolve()==Path('output/bunny_paper_assets').resolve():
        path = Path('output/libuipc_bunny_L0_contact_v2/accepted_positions.npy')
        if path.exists(): state = np.load(path)[-1]
    with np.load(bundle) as d:
        V = csr_matrix((d['U_data'],d['U_indices'],d['U_indptr']),shape=tuple(d['U_shape']))
        tri = state[d['coarse_faces']]
        F = np.stack((tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=2) @ d['inv_dm']
        q,_,vt = np.linalg.svd(F,full_matrices=False)
        n = np.cross(F[:,:,0],F[:,:,1]); n /= np.linalg.norm(n,axis=1)[:,None]
        ids = d['anchors']
        y = V@state + n[ids]*d['gamma'][:,None] + np.einsum('nij,nj->ni',(q@vt)[ids],d['tangent'])
    error = float(np.max(np.abs(y-transfer.position(state))))
    if error>1e-12: raise RuntimeError(f'Export mismatch: {error}')
    info = dict(level=a.level, fine_level=a.fine_level, identity=a.level==a.fine_level,current_vertices=U.shape[1], proxy_vertices=U.shape[0],
        nnz=U.nnz, sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(), roundtrip_max_error_m=error,
        native_psq_backend_implemented=False,
        contract=dict(dtype='float64', indices='int32', layout='C row-major', units='SI',
            proxy_role='elastic evaluation only: no mass, no collision, no solver DOFs',
            gradient='dt^2 * U.T * fine_elastic_gradient; offsets frozen in derivatives',
            hessian='dt^2 * current-level elastic Hessian for paper coarse preconditioner',
            energy='dt^2 * fine_elastic_energy(P(x)); recompute P for every trial x',
            current_level_terms='native mass/inertia, contact/friction, additional C-IPC strain barrier',
            native_energy_scaling='libuipc uses dt^2 E + inertial energy; do not add a second native coarse elastic gradient'))
    (a.output/'manifest.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))


if __name__=='__main__': main()
