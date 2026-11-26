#pragma once
#include <algorithm>
#include <cmath>
namespace flag_line_search {
// Fit delta E(a)=g*a+q*a*a to two rejected finite trials and E(0).
// An interpolation is only a proposal: the ordinary full-energy test decides.
inline double next(double a,double delta,double previous_a,double previous_delta) {
    double candidate=.5*a;
    if(previous_a>a && std::isfinite(delta) && std::isfinite(previous_delta)) {
        const double q=(previous_delta/previous_a-delta/a)/(previous_a-a);
        const double g=delta/a-q*a;
        if(std::isfinite(q) && std::isfinite(g) && q>0 && g<0)
            candidate=std::clamp(-g/(2*q),.1*a,.5*a);
    }
    return candidate;
}
}
