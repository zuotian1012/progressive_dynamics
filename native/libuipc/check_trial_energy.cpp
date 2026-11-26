#include "psq_trial_energy.h"
#include <limits>
#include <cstdio>
#include <initializer_list>
int main() {
    const double inf=std::numeric_limits<double>::infinity();
    const double nan=std::numeric_limits<double>::quiet_NaN();
    using psq_trial_energy::rejectable_positive_infinity;
    int count=0;
    for(bool initial:{false,true})for(bool fem:{false,true})for(bool enabled:{false,true}) {
        bool expected=!initial && fem && enabled;
        if(rejectable_positive_infinity(inf,initial,fem,enabled)!=expected)return 1;
        ++count;
        for(double energy:{0.,1.,-1.,-inf,nan}) {
            if(rejectable_positive_infinity(energy,initial,fem,enabled))return 2;
            ++count;
        }
    }
    std::printf("%d trial-energy boundary checks passed\n",count);
}
