#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
namespace psq_preview {
// PSQ Algorithm 1, lines 14-15. Only explicit non-identity coarse previews
// may use this rule; gamma is a declared local choice, not a recovered value.
inline bool small_step(bool enabled,bool coarse,bool converged,double alpha,double gamma) {
    return enabled && coarse && !converged && std::isfinite(alpha) && alpha>=0
        && std::isfinite(gamma) && gamma>0 && gamma<1 && alpha<gamma;
}
struct Decision { bool resolved=false,consistent=false; double noise=0,slopes[3]={}; };
// Flat/noisy trials are not evidence of non-descent.
inline Decision assess(double base,double last,const double* h,const double* e) {
    Decision d;
    if(!std::isfinite(base) || !std::isfinite(last))return d;
    double scale=std::max(std::abs(base),1e-30);
    for(int i=0;i<3;++i) {
        if(!(h[i]>0) || !std::isfinite(h[i]) || !std::isfinite(e[i]))return d;
        scale=std::max(scale,std::abs(e[i]));
    }
    if(!(h[0]>h[1] && h[1]>h[2]))return d;
    d.noise=64*std::numeric_limits<double>::epsilon()*scale;
    d.resolved=true;
    for(int i=0;i<3;++i) {
        d.slopes[i]=(e[i]-base)/h[i];
        d.resolved=d.resolved && std::isfinite(d.slopes[i]) && e[i]-base>d.noise;
    }
    if(!d.resolved || last<base-d.noise)return d;
    double lo=*std::min_element(d.slopes,d.slopes+3),hi=*std::max_element(d.slopes,d.slopes+3);
    d.consistent=hi-lo<=.25*hi;
    return d;
}
}
