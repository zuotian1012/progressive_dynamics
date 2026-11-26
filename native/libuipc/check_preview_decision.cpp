#include "psq_preview_decision.h"
#include <cstdio>
#include <cstdlib>
int main() {
    using psq_preview::small_step;
    if(!small_step(true,true,false,1e-12,1e-8)
        || small_step(true,false,false,1e-12,1e-8)
        || small_step(false,true,false,1e-12,1e-8)
        || small_step(true,true,true,1e-12,1e-8)
        || small_step(true,true,false,1e-8,1e-8)
        || small_step(true,true,false,NAN,1e-8)
        || small_step(true,true,false,-1,1e-8)
        || small_step(true,true,false,0,0))return 1;
    double h[3]={1e-3,1e-4,1e-5};int count=0;
    auto check=[&](const char* name,double base,double last,const double* steps,const double* energies,bool expected) {
        auto d=psq_preview::assess(base,last,steps,energies);
        if(d.consistent!=expected) {std::fprintf(stderr,"FAILED %s\n",name);std::exit(1);}++count;
    };
    double up[3]={1.001,1.0001,1.00001};check("ascending",1,1.1,h,up,true);
    double down[3]={.999,.9999,.99999};check("descending",1,1.1,h,down,false);
    check("earlier decrease must be preserved",1,.9,h,up,false);
    double flat[3]={1,1,1};check("flat",1,1,h,flat,false);
    double noise[3]={1+2e-16,1+2e-16,1+2e-16};check("unresolved",1,1,h,noise,false);
    double curve[3]={1.001,1.001,1.001};check("inconsistent slopes",1,1.1,h,curve,false);
    double mixed[3]={1.001,1.0001,.99999};check("fine-scale descent",1,1.1,h,mixed,false);
    double tiny[3]={1e-5+1e-12,1e-5+1e-13,1e-5+1e-14};check("small resolved energy",1e-5,1.1e-5,h,tiny,true);
    double bad_h[3]={1e-3,0,1e-5};check("zero step",1,1,bad_h,up,false);
    double reverse_h[3]={1e-5,1e-4,1e-3};check("unordered scales",1,1,reverse_h,up,false);
    double inf[3]={INFINITY,1,1};check("nonfinite energy",1,1,h,inf,false);
    check("nonfinite base",NAN,1,h,up,false);
    check("nonfinite last trial",1,NAN,h,up,false);
    std::printf("%d non-descent and 8 small-step eligibility checks passed\n",count);
}
