#pragma once
#include <cmath>
#include <cfloat>
#ifdef __CUDACC__
#define PSQ_CAP_HD __host__ __device__
#else
#define PSQ_CAP_HD
#endif
namespace psq_step_cap {
// Compensated arithmetic for the near-boundary 2x2 Gram test. Directly
// subtracting double-precision singular values loses the available gap here.
struct Pair { double hi,lo; };
PSQ_CAP_HD inline Pair pair_sum(double a,double b) {
    double s=a+b,bb=s-a;
    return {s,(a-(s-bb))+(b-bb)};
}
PSQ_CAP_HD inline Pair add(Pair a,Pair b) {
    Pair s=pair_sum(a.hi,b.hi);
    return pair_sum(s.hi,s.lo+a.lo+b.lo);
}
PSQ_CAP_HD inline Pair sub(Pair a,Pair b) { return add(a,{-b.hi,-b.lo}); }
PSQ_CAP_HD inline Pair mul(Pair a,Pair b) {
    double p=a.hi*b.hi;
    double e=fma(a.hi,b.hi,-p)+a.hi*b.lo+a.lo*b.hi+a.lo*b.lo;
    return pair_sum(p,e);
}
PSQ_CAP_HD inline bool positive(Pair a) {
    return a.hi>0 || (a.hi==0 && a.lo>0);
}
// ||F+tD||_2 < limit iff limit^2 I - (F+tD)^T(F+tD) is positive
// definite. Evaluate both leading minors without cancellation at double
// precision. The O(eps^2) guard is in Gram units and keeps the test away from
// double-double roundoff; it does not increase the physical stretch limit.
PSQ_CAP_HD inline bool inside_stretch(const double* F,const double* D,double t,double limit) {
    Pair ll=mul({limit,0},{limit,0}),a=ll,c=ll,b={0,0};
    for(int j=0;j<3;++j) {
        Pair x=add({F[j],0},mul({t,0},{D[j],0}));
        Pair y=add({F[j+3],0},mul({t,0},{D[j+3],0}));
        a=sub(a,mul(x,x));c=sub(c,mul(y,y));b=add(b,mul(x,y));
    }
    double scale=fmax(1.,limit*limit);
    double guard=4096*DBL_EPSILON*DBL_EPSILON*scale;
    a=sub(a,{guard,0});c=sub(c,{guard,0});
    Pair det=sub(mul(a,c),mul(b,b));
    return positive(a) && positive(c) && positive(sub(det,{guard*scale,0}));
}
PSQ_CAP_HD inline double boundary_directional(const double* F,const double* D,double limit,double hi) {
    // A lo=0 bisection requires a strictly feasible initial point. Do not
    // silently allow an infeasible state because the native SVD rounded down.
    if(!inside_stretch(F,D,0.,limit)) return 0.;
    if(inside_stretch(F,D,hi,limit)) return hi;
    double lo=0.;
    for(int k=0;k<96;++k) {
        double mid=.5*(lo+hi);
        if(mid==lo || mid==hi) break;
        if(inside_stretch(F,D,mid,limit)) lo=mid; else hi=mid;
    }
    // Convexity gives upper-stretch feasibility for the entire segment; hi
    // supplied by the caller also retains the original Weyl rank protection.
    double result=.9*lo;
    return inside_stretch(F,D,result,limit)?result:0.;
}
// Column-major 3x2 matrices. Shared by production CUDA and the standalone check.
PSQ_CAP_HD inline double maximum(const double* F, const double* D, double t) {
    double a=0,b=0,c=0;
    for(int j=0;j<3;++j) { double x=F[j]+t*D[j], y=F[j+3]+t*D[j+3]; a+=x*x; b+=x*y; c+=y*y; }
    return sqrt(fmax(0.,.5*(a+c+hypot(a-c,2*b))));
}
PSQ_CAP_HD inline double directional(const double* F,const double* D,double limit,
                                    double smax,double smin,double norm,double legacy) {
    if(!(smax<limit) || !(smin>1e-12) || !(norm>=0)) return 0.;
    // Weyl rank protection is unchanged. In particular, an endpoint with full
    // rank alone would not exclude an intermediate rank collapse.
    double hi=norm==0 ? 1. : fmin(1.,.9*(smin-1e-12)/norm);
    double target=limit-64*DBL_EPSILON*fmax(1.,limit);
    if(!(smax<target)) {
        return boundary_directional(F,D,limit,hi);
    }
    if(norm==0) return 1.;
    // The spectral norm is convex along a linear segment: feasible endpoints
    // certify the entire segment for the upper stretch constraint.
    if(maximum(F,D,hi)<target) return hi;
    double lo=0.;
    for(int k=0;k<48;++k) {
        double mid=.5*(lo+hi);
        if(maximum(F,D,mid)<target) lo=mid; else hi=mid;
    }
    return .9*lo;
}
}
#undef PSQ_CAP_HD
