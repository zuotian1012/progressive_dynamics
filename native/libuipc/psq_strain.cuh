#pragma once
#include <type_define.h>
#include <cuda_tool/cuda_tool.h>
#include <cmath>
#include "psq_step_cap.h"

namespace uipc::backend::cuda::psq_native
{
// C-IPC singular-value barrier. Full material thickness is supplied by caller.
// The spectral Hessian is evaluated analytically; it is projected by the native
// element assembler together with the current-level elastic Hessian.
struct StrainParameters { Float limit, onset, stiffness; };
using Mat32 = Eigen::Matrix<Float,3,2>;

// Explicit round-to-nearest operations keep the energy, derivative and cap
// kernels on the same arithmetic path despite different compiler contexts.
UIPC_DEVICE inline Float dot3(const Vector3& a,const Vector3& b)
{
    return __dadd_rn(__dadd_rn(__dmul_rn(a(0),b(0)),__dmul_rn(a(1),b(1))),
                     __dmul_rn(a(2),b(2)));
}

UIPC_DEVICE inline Matrix2x2 inverse_rest_frame(const Matrix2x2& inverse_metric)
{
    // Native NH stores (Dm^T Dm)^-1, not Dm^-1. Its positive square root
    // gives an equivalent orthonormal rest frame for singular-value barriers.
    Float determinant=__dsub_rn(__dmul_rn(inverse_metric(0,0),inverse_metric(1,1)),
                                __dmul_rn(inverse_metric(0,1),inverse_metric(1,0)));
    Float root=sqrt(fmax(determinant,0.0));
    Float denominator=sqrt(__dadd_rn(__dadd_rn(inverse_metric(0,0),inverse_metric(1,1)),__dmul_rn(2.,root)));
    Matrix2x2 frame;
    for(int i=0;i<2;++i)for(int j=0;j<2;++j)
        frame(i,j)=__ddiv_rn(i==j?__dadd_rn(inverse_metric(i,j),root):inverse_metric(i,j),denominator);
    return frame;
}

UIPC_DEVICE inline void svd32(const Mat32& F, Matrix3x3& U, Matrix2x2& V, Vector2& s)
{
    Float a=dot3(F.col(0),F.col(0)), b=dot3(F.col(0),F.col(1)), c=dot3(F.col(1),F.col(1));
    Float angle=0.5*atan2(2*b,a-c), co=cos(angle), si=sin(angle);
    V << co,-si,si,co;
    s(0)=sqrt(fmax(0.0,__dmul_rn(.5,__dadd_rn(__dadd_rn(a,c),hypot(__dsub_rn(a,c),__dmul_rn(2.,b))))));
    Vector3 cross;
    for(int j=0;j<3;++j)cross(j)=__dsub_rn(__dmul_rn(F((j+1)%3,0),F((j+2)%3,1)),
                                           __dmul_rn(F((j+2)%3,0),F((j+1)%3,1)));
    s(1)=__ddiv_rn(sqrt(dot3(cross,cross)),fmax(s(0),1e-30));
    U.col(0)=F*V.col(0)/fmax(s(0),1e-30);
    U.col(1)=F*V.col(1)/fmax(s(1),1e-30);
    U.col(2)=U.col(0).cross(U.col(1));
}

UIPC_DEVICE inline Mat32 deformation(const Vector9& X, const Matrix2x2& IB)
{
    Mat32 F;
    for(int j=0;j<3;++j) {
        Float a=__dsub_rn(X(3+j),X(j)),b=__dsub_rn(X(6+j),X(j));
        for(int k=0;k<2;++k)F(j,k)=__dadd_rn(__dmul_rn(a,IB(0,k)),__dmul_rn(b,IB(1,k)));
    }
    return F;
}

UIPC_DEVICE inline bool feasible(const Mat32& F,const Vector2& s,Float limit)
{
    if(!(s(1)>1e-12) || !(s(0)<limit))return false;
    if(s(0)>=limit-64*DBL_EPSILON*fmax(1.,limit)) {
        const double zero[6]={};
        if(!psq_step_cap::inside_stretch(F.data(),zero,0.,limit))return false;
    }
    return true;
}

UIPC_DEVICE inline Float coordinate_step(const Vector9& X,const Vector9& D,
    const Matrix2x2& IB,Float limit,const Mat32& F,const Mat32& dF,const Vector2& s,Float legacy)
{
    if(!feasible(F,s,limit))return 0.;
    Float coordinates=0.,inverse_norm=0.;
    for(int j=0;j<9;++j)coordinates=fmax(coordinates,fabs(X(j))+fabs(D(j)));
    for(int k=0;k<2;++k)inverse_norm=fmax(inverse_norm,fabs(IB(0,k))+fabs(IB(1,k)));
    // Account for storing world-space vertices and subtracting short edges.
    // This only moves the cap target inward; the physical barrier is unchanged.
    Float margin=256*DBL_EPSILON*fmax(1.,coordinates*inverse_norm)*fmax(1.,limit);
    Float alpha=psq_step_cap::directional(F.data(),dF.data(),limit,s(0),s(1),dF.norm(),legacy,margin);
    // Replay rounded coordinates as well as the linearized deformation path.
    // Test fused and unfused updates to avoid relying on compiler contraction.
    for(int attempt=0;attempt<64 && alpha>0;++attempt,alpha*=.5) {
        bool valid=true;
        for(int fused=0;fused<2;++fused) {
            Vector9 trial;for(int j=0;j<9;++j)
                trial(j)=fused?__fma_rn(alpha,D(j),X(j)):__dadd_rn(X(j),__dmul_rn(alpha,D(j)));
            Mat32 next=deformation(trial,IB);Matrix3x3 u;Matrix2x2 v;Vector2 sigma;
            svd32(next,u,v,sigma);valid=valid && feasible(next,sigma,limit);
        }
        if(valid)return alpha;
    }
    return 0.;
}

UIPC_DEVICE inline Vector9 pullback(const Mat32& stress, const Matrix2x2& IB)
{
    Mat32 gd=stress*IB.transpose(); Vector9 g;
    g.segment<3>(0)=-gd.col(0)-gd.col(1);
    g.segment<3>(3)=gd.col(0); g.segment<3>(6)=gd.col(1);
    return g;
}

UIPC_DEVICE inline Float strain(const Vector9& X, const Matrix2x2& inverse_metric,
                               StrainParameters p, Vector9* g=nullptr, Matrix9x9* h=nullptr)
{
    if(g) g->setZero(); if(h) h->setZero();
    Matrix2x2 IB=inverse_rest_frame(inverse_metric);
    Matrix3x3 U; Matrix2x2 V; Vector2 s;
    Mat32 F=deformation(X,IB);
    svd32(F,U,V,s);
    if(!feasible(F,s,p.limit))return INFINITY;
    if(s(0)<=p.onset) return 0;
    Float width=p.limit-p.onset, scale=p.stiffness/(width*width), E=0;
    Vector2 first=Vector2::Zero(), second=Vector2::Zero();
    for(int j=0;j<2;++j) if(s(j)>p.onset) {
        Float gap=p.limit-s(j), d=gap-width, l=log(gap/width);
        E-=scale*d*d*l;
        first(j)=scale*(2*d*l+d*d/gap);
        second(j)=scale*(-2*l-4*d/gap+d*d/(gap*gap));
    }
    if(g) *g=pullback(U.leftCols<2>()*first.asDiagonal()*V.transpose(),IB);
    if(h) {
        Float sym=fabs(s(0)-s(1))<1e-10 ? 0.5*(second(0)+second(1))
                    : (first(0)-first(1))/(s(0)-s(1));
        Float skew=(first(0)+first(1))/(s(0)+s(1));
        for(int k=0;k<9;++k) {
            Vector9 basis=Vector9::Zero(); basis(k)=1;
            Mat32 D=U.transpose()*deformation(basis,IB)*V, R=Mat32::Zero();
            R(0,0)=second(0)*D(0,0); R(1,1)=second(1)*D(1,1);
            Float a=0.5*sym*(D(0,1)+D(1,0)), b=0.5*skew*(D(0,1)-D(1,0));
            R(0,1)=a+b; R(1,0)=a-b;
            R(2,0)=first(0)/s(0)*D(2,0); R(2,1)=first(1)/s(1)*D(2,1);
            h->col(k)=pullback(U*R*V.transpose(),IB);
        }
        *h=0.5*(*h+h->transpose()).eval();
    }
    return E;
}
}
