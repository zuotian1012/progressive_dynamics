#pragma once
#include <cuda_runtime.h>
#include <cmath>

namespace psq {
// All buffers are device-resident, float64 / int32, row-major. U is CSR.
// Proxy vertices have no independent mass, collision mesh, or solver DOFs.
__global__ void prolong(int nf, const int* row, const int* col, const double* val,
                       const int* faces, const int* anchors, const double* inv_dm,
                       const double* gamma, const double* tangent,
                       const double* x, double* y, int* invalid) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nf) return;
    int f = anchors[i];
    int a=faces[3*f], b=faces[3*f+1], c=faces[3*f+2];
    double F[6];
    for(int j=0;j<3;++j) {
        double e=x[3*b+j]-x[3*a+j], h=x[3*c+j]-x[3*a+j];
        F[2*j]=e*inv_dm[4*f]+h*inv_dm[4*f+2];
        F[2*j+1]=e*inv_dm[4*f+1]+h*inv_dm[4*f+3];
    }
    double A=0,B=0,C=0;
    for(int j=0;j<3;++j) { A+=F[2*j]*F[2*j]; B+=F[2*j]*F[2*j+1]; C+=F[2*j+1]*F[2*j+1]; }
    double n[3]={F[2]*F[5]-F[4]*F[3],F[4]*F[1]-F[0]*F[5],F[0]*F[3]-F[2]*F[1]};
    // s=sqrt(det(F^T F)); cross-product form avoids determinant cancellation.
    double s=sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
    double tr=A+C, disc=hypot(A-C,2*B), largest=.5*(tr+disc);
    double smallest=largest>0 ? s*s/largest : 0;
    if(!isfinite(s) || !(smallest>=1e-24)) {
        atomicExch(invalid,1); for(int j=0;j<3;++j) y[3*i+j]=nan(""); return;
    }
    // Closed-form SPD 2x2 inverse square root; R=F(F^T F)^(-1/2).
    double scale=sqrt(tr+2*s)/(s*(tr+2*s));
    double i00=(C+s)*scale, i01=-B*scale, i11=(A+s)*scale;
    for(int j=0;j<3;++j) {
        double linear=0;
        for(int k=row[i]; k<row[i+1]; ++k) linear+=val[k]*x[3*col[k]+j];
        double r0=F[2*j]*i00+F[2*j+1]*i01;
        double r1=F[2*j]*i01+F[2*j+1]*i11;
        y[3*i+j]=linear+n[j]/s*gamma[i]+r0*tangent[2*i]+r1*tangent[2*i+1];
    }
}

// Frozen-offset derivative only: g_coarse = U^T g_proxy.
// Caller zeros coarse_g; multiply by dt^2 when assembling libuipc energy.
__global__ void restrict_gradient(int nf,const int* row,const int* col,const double* val,
                                  const double* fine_g,double* coarse_g) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=nf) return;
    for(int k=row[i];k<row[i+1];++k)
        for(int j=0;j<3;++j) atomicAdd(coarse_g+3*col[k]+j,val[k]*fine_g[3*i+j]);
}
} // namespace psq
