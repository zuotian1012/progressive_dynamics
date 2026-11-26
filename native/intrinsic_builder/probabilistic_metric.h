// SPDX-License-Identifier: GPL-3.0-or-later
// Uses Trettner & Kobbelt's MIT-licensed reference implementation.
#pragma once
#include <minimal-math.hh>
#include <Eigen/Core>
#include <tuple>
#include <vector>
#include <stdexcept>

namespace pd_metric {
inline bool probabilistic=true;
inline bool lock_boundary=false;
inline double normal_stddev=.01;
inline double position_stddev_relative=.01;
using Quadric=std::tuple<Eigen::MatrixXd,Eigen::RowVectorXd,double>;

inline Quadric convert(const pq::quadric<pq::minimal_math<double>>& q) {
    Eigen::Matrix3d A;
    A<<q.A00,q.A01,q.A02,q.A01,q.A11,q.A12,q.A02,q.A12,q.A22;
    // pq uses x'Ax - 2b'x + c; libigl uses x'Ax + 2b'x + c.
    Eigen::RowVector3d b(-q.b0,-q.b1,-q.b2);
    return {A,b,q.c};
}

inline void add_uncertainty(const Eigen::MatrixXd& V,const Eigen::MatrixXi& F,
                            std::vector<Quadric>& quadrics) {
    if(!probabilistic)return;
    using Q=pq::quadric<pq::minimal_math<double>>;
    double sum=0.;size_t count=0;
    for(int f=0;f<F.rows();++f) {
        if(!V.row(F(f,0)).allFinite() || !V.row(F(f,1)).allFinite() || !V.row(F(f,2)).allFinite())continue;
        for(int j=0;j<3;++j){sum+=(V.row(F(f,j))-V.row(F(f,(j+1)%3))).norm();++count;}
    }
    if(!count)throw std::runtime_error("No finite triangles for probabilistic metric");
    double sp=position_stddev_relative*sum/count;
    for(int f=0;f<F.rows();++f) {
        Eigen::Vector3d a=V.row(F(f,0)),b=V.row(F(f,1)),c=V.row(F(f,2));
        if(!a.allFinite() || !b.allFinite() || !c.allFinite())continue;
        Eigen::Vector3d cross=(b-a).cross(c-a);
        double area2=cross.norm();if(area2<=0)throw std::runtime_error("Degenerate input face");
        Eigen::Vector3d n=cross/area2,p=(a+b+c)/3.;
        pq::dpos3 point{p[0],p[1],p[2]};pq::dvec3 normal{n[0],n[1],n[2]};
        // Keep libigl's classical plane and boundary terms, replacing each
        // finite face plane by the reference probabilistic plane quadric.
        auto uncertainty=convert(Q::probabilistic_plane_quadric(point,normal,sp,normal_stddev)-Q::plane_quadric(point,normal));
        for(int j=0;j<3;++j){
            auto& target=quadrics[F(f,j)];
            std::get<0>(target)+=area2*std::get<0>(uncertainty);
            std::get<1>(target)+=area2*std::get<1>(uncertainty);
            std::get<2>(target)+=area2*std::get<2>(uncertainty);
        }
    }
}
}
