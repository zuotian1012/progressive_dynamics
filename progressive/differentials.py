"""Batched exact forward differentials for the discrete hinge gradient.

This evaluates chain rules in floating point, without perturbed geometry or a
finite-difference step. The last axis stores derivatives w.r.t. local DOFs.
"""
import numpy as np


class Differential:
    def __init__(self,value,derivative):
        self.v=np.asarray(value)
        self.d=np.asarray(derivative)

    def __add__(self,other):
        if isinstance(other,Differential):return Differential(self.v+other.v,self.d+other.d)
        return Differential(self.v+other,self.d)
    __radd__=__add__

    def __neg__(self):return Differential(-self.v,-self.d)
    def __sub__(self,other):return self+-other
    def __rsub__(self,other):return -self+other

    def __mul__(self,other):
        if isinstance(other,Differential):
            return Differential(self.v*other.v,self.d*other.v[...,None]+self.v[...,None]*other.d)
        return Differential(self.v*other,self.d*np.asarray(other)[...,None])
    __rmul__=__mul__

    def __truediv__(self,other):
        if isinstance(other,Differential):return self*other.power(-1)
        return self*(1/np.asarray(other))

    def power(self,p):return Differential(self.v**p,self.d*(p*self.v**(p-1))[...,None])

    def sum(self):return Differential(self.v.sum(axis=1,keepdims=True),self.d.sum(axis=1,keepdims=True))

    def component(self,i):return Differential(self.v[:,i:i+1],self.d[:,i:i+1])


def cross(a,b):
    c=[a.component(i)*b.component(j)-a.component(j)*b.component(i) for i,j in ((1,2),(2,0),(0,1))]
    return Differential(np.concatenate([v.v for v in c],axis=1),np.concatenate([v.d for v in c],axis=1))


def atan2(y,x):
    return Differential(np.arctan2(y.v,x.v),
                        (x.v[...,None]*y.d-y.v[...,None]*x.d)/(x.v*x.v+y.v*y.v)[...,None])


def hinge_gradient_hessian(p,rest_angles,coeff,periodic=False):
    """Derivative of the signed-angle hinge energy gradient, in local order."""
    n=len(p)
    basis=np.broadcast_to(np.eye(12).reshape(4,3,12),(n,4,3,12))
    a,b,c,d=[Differential(p[:,i],basis[:,i]) for i in range(4)]
    e,u,v=b-a,c-a,d-a
    le=(e*e).sum().power(.5);eh=e/le
    r1,r2=cross(e,u),cross(v,e)
    l1,l2=(r1*r1).sum().power(.5),(r2*r2).sum().power(.5)
    n1,n2=r1/l1,r2/l2
    s,co=(eh*cross(n1,n2)).sum(),(n1*n2).sum()
    theta=atan2(s,co)
    delta=theta-rest_angles[:,None]
    # Wrapping changes only the value on a fixed branch, not its derivative.
    if periodic:delta.v=np.arctan2(np.sin(delta.v),np.cos(delta.v))
    scale=delta*coeff[:,None]/(s*s+co*co)
    gs,gc=scale*co,-scale*s
    geh=gs*cross(n1,n2)
    gn1,gn2=gs*cross(n2,eh)+gc*n2,gs*cross(eh,n1)+gc*n1
    gr1=(gn1-n1*(n1*gn1).sum())/l1
    gr2=(gn2-n2*(n2*gn2).sum())/l2
    ge=(geh-eh*(eh*geh).sum())/le+cross(u,gr1)+cross(gr2,v)
    gu,gv=cross(gr1,e),cross(e,gr2)
    return np.concatenate([g.d for g in (-ge-gu-gv,ge,gu,gv)],axis=1)
