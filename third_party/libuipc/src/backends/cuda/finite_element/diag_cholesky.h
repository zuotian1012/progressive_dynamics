#pragma once
#include <cmath>
#include <cfloat>
#ifdef __CUDACC__
#define UIPC_DIAG_HD __host__ __device__
#else
#define UIPC_DIAG_HD
#endif
namespace uipc_diag {
struct Factor { double lower[9]; double scale[3]; };
// Returns 0 for scaled Cholesky, 1 for positive diagonal Jacobi fallback,
// -1 for an invalid diagonal. Only the preconditioner changes, never A.
UIPC_DIAG_HD inline int factor(const double* a, Factor& f) {
    for(int i=0;i<9;++i) f.lower[i]=0;
    for(int i=0;i<3;++i) {
        if(!(a[3*i+i]>0 && a[3*i+i]<=DBL_MAX)) return -1;
        f.scale[i]=sqrt(a[3*i+i]);
    }
    for(int i=0;i<3;++i) for(int j=0;j<=i;++j) {
        double v=.5*(a[3*i+j]/f.scale[i]/f.scale[j]+a[3*j+i]/f.scale[j]/f.scale[i]);
        for(int k=0;k<j;++k) v=fma(-f.lower[3*i+k],f.lower[3*j+k],v);
        if(!(v>=-DBL_MAX && v<=DBL_MAX) || (i==j && !(v>0))) {
            // Rounding may destroy a tiny positive pivot. A diagonal positive
            // preconditioner remains valid; do not invert an indefinite block.
            for(int k=0;k<9;++k) f.lower[k]=0;
            for(int k=0;k<3;++k) f.lower[3*k+k]=1;
            return 1;
        }
        f.lower[3*i+j]=i==j?sqrt(v):v/f.lower[3*j+j];
    }
    return 0;
}
UIPC_DIAG_HD inline void solve(const Factor& f,const double* r,double* z) {
    double y[3],w[3];
    for(int i=0;i<3;++i) {
        double v=r[i]/f.scale[i];
        for(int j=0;j<i;++j) v=fma(-f.lower[3*i+j],y[j],v);
        y[i]=v/f.lower[3*i+i];
    }
    for(int i=2;i>=0;--i) {
        double v=y[i];
        for(int j=i+1;j<3;++j) v=fma(-f.lower[3*j+i],w[j],v);
        w[i]=v/f.lower[3*i+i]; z[i]=w[i]/f.scale[i];
    }
}
}
#undef UIPC_DIAG_HD
