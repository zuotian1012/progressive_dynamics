"""Driven physical gripper surfaces reduced to 12 affine DOFs per body.

All cloth/body coupling is IPC contact/friction. The cloth has no handle,
pin, stitch, robot target, or prescribed grasp vertices. Body orthogonality
penalties approximate rigid bodies, as in affine-body dynamics.
"""
import numpy as np
from scipy.sparse import coo_matrix, eye, kron, diags
from .shell import assemble_hessian


class AffineObstacles:
    def __init__(self,cloth_rest,cloth_faces,bodies=(),fixed=(),drive_translation=1e6,drive_affine=1e3,rigidity=1e4):
        self.n=len(cloth_rest)
        self.bodies=list(bodies)
        self.rigidity=rigidity
        self.weights=np.tile([drive_translation,drive_affine,drive_affine,drive_affine],len(bodies))
        self.weights=np.repeat(self.weights,3)
        self.q0=np.zeros((self.n+4*len(bodies),3))
        self.q0[:self.n]=cloth_rest
        self.names=[]
        faces=[cloth_faces]
        rows=list(range(self.n)); cols=list(range(self.n)); values=[1.]*self.n
        offsets=[np.zeros_like(cloth_rest)]
        count=self.n
        for i,(name,v,f,transform) in enumerate(bodies):
            self.names.append(name)
            self.q0[self.n+4*i:self.n+4*i+4]=np.vstack((transform[:3,3],transform[:3,:3].T))
            coeff=np.column_stack((np.ones(len(v)),v))
            rows.extend(np.repeat(np.arange(count,count+len(v)),4))
            cols.extend(np.tile(np.arange(self.n+4*i,self.n+4*i+4),len(v)))
            values.extend(coeff.ravel())
            offsets.append(np.zeros_like(v))
            faces.append(f+count)
            count+=len(v)
        for v,f in fixed:
            offsets.append(v)
            faces.append(f+count)
            count+=len(v)
        self.B=coo_matrix((values,(rows,cols)),shape=(count,len(self.q0))).tocsr()
        self.B3=kron(self.B,eye(3),format="csr")
        self.offset=np.vstack(offsets)
        self.faces=np.vstack(faces)
        self.target=self.q0[self.n:].copy()
        self.indices=np.arange(self.n,len(self.q0)).reshape(-1,4)

    def positions(self,q):
        return self.B @ q+self.offset

    def set_targets(self,transforms):
        for i,name in enumerate(self.names):
            t=transforms[name]
            self.target[4*i:4*i+4]=np.vstack((t[:3,3],t[:3,:3].T))

    def rigidity_local(self,p):
        A=p[:,1:,:].transpose(0,2,1)
        C=A.transpose(0,2,1) @ A-np.eye(3)
        value=.25*self.rigidity*np.sum(C*C,axis=(1,2))
        gradient=np.zeros_like(p)
        gradient[:,1:,:]=(self.rigidity*A @ C).transpose(0,2,1)
        return value,gradient

    def evaluate(self,q,hessian=False):
        gradient=np.zeros_like(q)
        delta=(q[self.n:]-self.target).ravel()
        value=.5*np.sum(self.weights*delta**2)
        gradient[self.n:]=(self.weights*delta).reshape(-1,3)
        p=q[self.indices]
        e,g=self.rigidity_local(p)
        value+=e.sum()
        gradient[self.n:]+=g.reshape(-1,3)
        H=None
        if hessian:
            H=diags(np.concatenate((np.zeros(self.n*3),self.weights)),format="csr")
            if len(p):
                A=p[:,1:,:].transpose(0,2,1)
                C=A.transpose(0,2,1) @ A-np.eye(3)
                blocks=np.zeros((len(p),12,12))
                for i in range(3,12):
                    delta=np.zeros_like(p);delta[:,i//3,i%3]=1.
                    dA=delta[:,1:,:].transpose(0,2,1)
                    dC=dA.transpose(0,2,1) @ A+A.transpose(0,2,1) @ dA
                    blocks[:,3:,i]=(self.rigidity*(dA @ C+A @ dC)).transpose(0,2,1).reshape(-1,9)
                H+=assemble_hessian(self.indices,blocks,len(q))
        return float(value),gradient,H

    def transforms(self,q):
        result=np.tile(np.eye(4),(len(self.names),1,1))
        for i in range(len(self.names)):
            p=q[self.n+4*i:self.n+4*i+4]
            result[i,:3,3]=p[0]
            result[i,:3,:3]=p[1:].T
        return result
