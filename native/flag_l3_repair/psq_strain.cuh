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

UIPC_DEVICE inline Matrix2x2 inverse_rest_frame(const Matrix2x2& inverse_metric)
{
    // Native NH stores (Dm^T Dm)^-1, not Dm^-1. Its positive square root
    // gives an equivalent orthonormal rest frame for singular-value barriers.
    Float determinant=inverse_metric(0,0)*inverse_metric(1,1)-inverse_metric(0,1)*inverse_metric(1,0);
    Float root=sqrt(fmax(determinant,0.0));
    return (inverse_metric+root*Matrix2x2::Identity())/sqrt(inverse_metric.trace()+2*root);
}

UIPC_DEVICE inline void svd32(const Mat32& F, Matrix3x3& U, Matrix2x2& V, Vector2& s)
{
    Float a=F.col(0).squaredNorm(), b=F.col(0).dot(F.col(1)), c=F.col(1).squaredNorm();
    Float angle=0.5*atan2(2*b,a-c), co=cos(angle), si=sin(angle);
    V << co,-si,si,co;
    s(0)=sqrt(fmax(0.0,0.5*(a+c+hypot(a-c,2*b))));
    s(1)=F.col(0).cross(F.col(1)).norm()/fmax(s(0),1e-30);
    U.col(0)=F*V.col(0)/fmax(s(0),1e-30);
    U.col(1)=F*V.col(1)/fmax(s(1),1e-30);
    U.col(2)=U.col(0).cross(U.col(1));
}

UIPC_DEVICE inline Mat32 deformation(const Vector9& X, const Matrix2x2& IB)
{
    Mat32 ds;
    ds.col(0)=X.segment<3>(3)-X.segment<3>(0);
    ds.col(1)=X.segment<3>(6)-X.segment<3>(0);
    return ds*IB;
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
    if(!(s(1)>1e-12) || !(s(0)<p.limit)) return INFINITY;
    // Use the same compensated boundary predicate as the step cap. Otherwise
    // an SVD rounding down could admit a trial that the next cap must reject.
    if(s(0)>=p.limit-64*DBL_EPSILON*fmax(1.,p.limit)) {
        const double zero[6]={0,0,0,0,0,0};
        if(!psq_step_cap::inside_stretch(F.data(),zero,0.,p.limit)) return INFINITY;
    }
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
