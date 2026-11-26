"""Reversible intrinsic collapse patches and dependency-respecting coloring.

PSQ 2023 Algorithms 6--7. IDs remain in the finer mesh's original numbering;
only collision/elastic queries use compact numbering. Coloring is conservative:
entire patch vertex sets must be disjoint, not just the split edges.
"""
import json
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from .prolongation import ShellProlongation


def face_key(face):
    return tuple(sorted(map(int, face)))


def oriented_face_key(face):
    face=tuple(map(int,face))
    return min(face,face[1:]+face[:1],face[2:]+face[:2])


class ExpansionPatch:
    def __init__(self, record):
        self.survivor, self.removed = int(record['survivor']), int(record['removed'])
        self.ids = np.asarray(record['ids'], int)
        self.pre_rest = np.asarray(record['pre_rest'], float)
        self.post_rest = np.asarray(record['post_rest'], float)
        self.pre_faces = np.asarray(record['pre_faces'], int).reshape(-1, 3)
        self.post_faces = np.asarray(record['post_faces'], int).reshape(-1, 3)
        if len(set(self.ids)) != len(self.ids) or not {self.survivor,self.removed} <= set(self.ids):
            raise ValueError('Invalid split vertex identities')
        index = {int(v):i for i,v in enumerate(self.ids)}
        self.active = np.unique(self.post_faces)
        if self.removed in self.active or self.survivor not in self.active:
            raise ValueError('Invalid post-collapse patch')
        self.split = np.array([self.survivor,self.removed])
        local = {int(v):i for i,v in enumerate(self.active)}
        coarse_faces = np.array([[local[int(v)] for v in f] for f in self.post_faces])
        uv = np.asarray(record['post_uv'], float)
        before_uv = np.asarray(record['pre_uv'], float)
        self.pre_uv,self.post_uv=before_uv,uv
        # Match the author's joint UV containment and boundary clipping rule.
        triangles = uv[[[index[int(v)] for v in f] for f in self.post_faces]]
        U = np.zeros((2,len(self.active))); anchors=[]
        for row,v in enumerate(self.split):
            t = before_uv[index[int(v)]]
            edges = np.stack((triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=2)
            ab = np.linalg.solve(edges,(t-triangles[:,0])[...,None])[...,0]
            bary = np.column_stack((1-ab.sum(axis=1),ab))
            face = int(np.argmax(bary.min(axis=1)))
            weights = np.maximum(bary[face],0);weights/=weights.sum()
            U[row,coarse_faces[face]]=weights;anchors.append(face)
        self.transfer = ShellProlongation(
            self.post_rest[[index[int(v)] for v in self.active]],
            self.pre_rest[[index[int(v)] for v in self.split]],
            coarse_faces,csr_matrix(U),np.array(anchors))

    def targets(self, x):
        return self.transfer.position(x[self.active])


class ExpansionHistory:
    def __init__(self, record, coarse_rest, coarse_faces, fine_rest, fine_faces):
        if record['schema'] != 1:raise ValueError('Unsupported expansion history schema')
        self.coarse_ids = np.asarray(record['coarse_ids'],int)
        self.coarse_rest,self.fine_rest = np.asarray(coarse_rest,float),np.asarray(fine_rest,float)
        self.coarse_faces,self.fine_faces = np.asarray(coarse_faces,int),np.asarray(fine_faces,int)
        self.n = len(fine_rest)
        if record['fine_vertex_count'] != self.n or len(self.coarse_ids)!=len(coarse_rest):
            raise ValueError('Expansion history does not match hierarchy')
        if len(set(self.coarse_ids))!=len(self.coarse_ids) or np.any(self.coarse_ids<0) or np.any(self.coarse_ids>=self.n):
            raise ValueError('Invalid coarse-to-original identities')
        self.patches = [ExpansionPatch(p) for p in reversed(record['collapses'])]
        # Collapse code reuses a parent's array slot for one child. That slot
        # is NOT a persistent vertex identity: a reverse split retires the
        # parent and creates both endpoints (PSQ Sec. 4.4.1). Only coarse
        # vertices never split survive as inherited vertices for Sec. 4.4.5.
        split_ids={int(v) for p in self.patches for v in p.split}
        self.unsplit_coarse_ids=np.array([v for v in self.coarse_ids if int(v) not in split_ids],dtype=int)
        # Preserve the order of all overlapping patches, including dependence
        # through a vertex which is absent from the initial coarse topology.
        predecessors=[];last={}
        for i,p in enumerate(self.patches):
            predecessors.append({last[int(v)] for v in p.ids if int(v) in last})
            for v in p.ids:last[int(v)]=i
        pending=set(range(len(self.patches)));done=set();self.colors=[]
        active=set(map(int,self.coarse_ids))
        while pending:
            used=set();batch=[]
            for i in sorted(pending):
                p=self.patches[i];vertices=set(map(int,p.ids))
                if predecessors[i] <= done and set(p.active)<=active and p.removed not in active and not vertices & used:
                    batch.append(i);used.update(vertices)
            if not batch:raise ValueError('Cyclic or inconsistent expansion history')
            self.colors.append(batch)
            for i in batch:active.add(self.patches[i].removed)
            done.update(batch);pending.difference_update(batch)
        self.validate_reconstruction()

    @classmethod
    def load(cls, path, *geometry):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')),*geometry)

    def initial(self, x):
        if np.asarray(x).shape != self.coarse_rest.shape or not np.isfinite(x).all():
            raise ValueError('Invalid coarse expansion state')
        positions=np.full((self.n,3),np.nan);positions[self.coarse_ids]=x
        rest=np.full((self.n,3),np.nan);rest[self.coarse_ids]=self.coarse_rest
        faces={face_key(f):f.copy() for f in self.coarse_ids[self.coarse_faces]}
        return positions,rest,faces

    def update_topology(self, x, rest, faces, color):
        for i in color:
            p=self.patches[i]
            for f in p.post_faces:
                if face_key(f) not in faces:raise ValueError('Missing pre-expansion face')
                del faces[face_key(f)]
            for f in p.pre_faces:
                if face_key(f) in faces:raise ValueError('Duplicate expanded face')
                faces[face_key(f)]=f.copy()
            x[p.removed]=x[p.survivor]
            rest[p.ids]=p.pre_rest

    def validate_reconstruction(self):
        x,rest,faces=self.initial(self.coarse_rest)
        for color in self.colors:
            for i in color:
                p=self.patches[i]
                lookup={int(v):k for k,v in enumerate(p.ids)}
                np.testing.assert_allclose(rest[p.active],p.post_rest[[lookup[int(v)] for v in p.active]],rtol=0,atol=1e-12)
                np.testing.assert_allclose(p.targets(rest),p.pre_rest[[lookup[int(v)] for v in p.split]],rtol=0,atol=1e-12)
            self.update_topology(x,rest,faces,color)
        if {oriented_face_key(f) for f in faces.values()} != {oriented_face_key(f) for f in self.fine_faces}:
            raise ValueError('History does not reconstruct oriented finest topology')
        np.testing.assert_allclose(rest,self.fine_rest,rtol=0,atol=1e-12)

    def summary(self):
        return {'collapses':len(self.patches),'colors':len(self.colors),
                'largest_batch':max(map(len,self.colors),default=0),
                'coarse_storage_slots':len(self.coarse_ids),'unsplit_inherited_vertices':len(self.unsplit_coarse_ids),
                'disjoint_patch_coloring':True,'rest_and_topology_reconstruction_verified':True}

    def dynamic_prolongation(self, rest, face_dict, expanded):
        """Replay intrinsic point queries through the remaining collapses.

        This follows query_fine_to_coarse's joint-UV containment, rather than
        replacing it by Euclidean nearest points or composing nonlinear P.
        Queries are indexed by their current face to avoid all-points scans.
        """
        vertices=np.unique(list(face_dict.values()));index=np.full(self.n,-1,int)
        index[vertices]=np.arange(len(vertices))
        faces=np.array(list(face_dict.values()),int)
        face_indices={face_key(f):i for i,f in enumerate(faces)}
        bf=np.zeros((self.n,3),int);bc=np.zeros((self.n,3));assigned=np.zeros(self.n,bool);buckets={}
        for face in self.fine_faces:
            key=face_key(face)
            for j,v in enumerate(face):
                if not assigned[v]:
                    assigned[v]=True;bf[v]=face;bc[v,j]=1;buckets.setdefault(key,[]).append(int(v))
        for i in reversed(range(len(self.patches))):
            if i in expanded:continue
            p=self.patches[i];lookup={int(v):j for j,v in enumerate(p.ids)}
            queries=[]
            for face in p.pre_faces:queries.extend(buckets.pop(face_key(face),[]))
            if not queries:continue
            q=np.asarray(queries,int)
            local=np.array([[lookup[int(v)] for v in face] for face in bf[q]])
            point=np.einsum('ni,nij->nj',bc[q],p.pre_uv[local])
            triangles=p.post_uv[[[lookup[int(v)] for v in f] for f in p.post_faces]]
            edges=np.stack((triangles[:,1]-triangles[:,0],triangles[:,2]-triangles[:,0]),axis=2)
            ab=np.linalg.solve(edges[None,:,:,:],(point[:,None,:]-triangles[None,:,0,:])[...,None])[...,0]
            weights=np.concatenate((1-ab.sum(axis=2,keepdims=True),ab),axis=2)
            selected=np.argmax(weights.min(axis=2),axis=1)
            chosen=np.maximum(weights[np.arange(len(q)),selected],0);chosen/=chosen.sum(axis=1)[:,None]
            bc[q]=chosen;bf[q]=p.post_faces[selected]
            for face_index in np.unique(selected):
                buckets.setdefault(face_key(p.post_faces[face_index]),[]).extend(q[selected==face_index].tolist())
        if np.any(index[bf]<0):raise ValueError('Intrinsic query references an inactive vertex')
        anchors=np.array([face_indices[face_key(f)] for f in bf],int)
        U=csr_matrix((bc.ravel(),(np.repeat(np.arange(self.n),3),index[bf].ravel())),shape=(self.n,len(vertices)))
        return ShellProlongation(rest[vertices],self.fine_rest,index[faces],U,anchors)
