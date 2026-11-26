#pragma once
#include <cmath>
namespace psq_trial_energy {
inline bool rejectable_positive_infinity(double energy, bool initial,
                                        bool fem_reporter, bool enabled) {
    return enabled && fem_reporter && !initial && std::isinf(energy) && energy>0;
}
}
