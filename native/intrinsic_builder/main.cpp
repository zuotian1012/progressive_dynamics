// SPDX-License-Identifier: GPL-3.0-or-later
// Local command-line adapter for Liu et al.'s public intrinsic prolongation
// implementation (2021). Author code is kept intact under external/ with its
// LICENSE; see README.md. Added 2026-11-06 for this research project.
#include <SSP_decimate.h>
#include <query_fine_to_coarse.h>
#include <igl/readOBJ.h>
#include <igl/writeOBJ.h>
#include <Eigen/Core>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>
#include "probabilistic_metric.h"
#include "expansion_history.h"

using Eigen::MatrixXd;
using Eigen::MatrixXi;
using Eigen::VectorXi;

static void vertex_queries(const MatrixXi& faces, int vertices,
                           MatrixXd& bc, MatrixXi& bf, VectorXi& fi) {
    bc=MatrixXd::Zero(vertices,3);
    bf=MatrixXi::Zero(vertices,3);
    fi=VectorXi::Constant(vertices,-1);
    for(int f=0;f<faces.rows();++f) for(int j=0;j<3;++j) {
        int v=faces(f,j);
        if(fi(v)<0) { fi(v)=f; bf.row(v)=faces.row(f); bc(v,j)=1.; }
    }
    if((fi.array()<0).any()) throw std::runtime_error("Isolated input vertices");
}

int main(int argc,char** argv) {
    try {
        if(argc==2 && std::string(argv[1])=="--self-test") {
            using Q=pq::quadric<pq::minimal_math<double>>;
            auto q=Q::probabilistic_plane_quadric({1,2,3},{0,0,1},.2,.1);
            auto [A,b,c]=pd_metric::convert(q);
            Eigen::Vector3d x(.3,-.5,.7),p(1,2,3);
            double expected=std::pow(x[2]-3,2)+.01*(x-p).squaredNorm()+.04+3*.04*.01;
            if(std::abs(x.dot(A*x)+2*b.dot(x)+c-expected)>1e-12)throw std::runtime_error("Quadric convention test failed");
            auto zero=pd_metric::convert(Q::probabilistic_plane_quadric({1,2,3},{0,0,1},0.,0.));
            if(std::abs(x.dot(std::get<0>(zero)*x)+2*std::get<1>(zero).dot(x)+std::get<2>(zero)-std::pow(x[2]-3,2))>1e-12)
                throw std::runtime_error("Zero uncertainty limit test failed");
            std::cout<<"probabilistic quadric equation and coefficient convention: passed\n";return 0;
        }
        if(argc<4) throw std::runtime_error("Usage: intrinsic_builder input.obj empty_output_directory target_faces [target_faces ...]");
        std::vector<int> targets;
        for(int i=3;i<argc;++i) {
            std::string arg(argv[i]);
            if(arg=="--lock-boundary"){pd_metric::lock_boundary=true;continue;}
            if(arg=="--quadric" && i+1<argc){std::string mode(argv[++i]);if(mode!="probabilistic" && mode!="ordinary")throw std::runtime_error("Unknown quadric mode");pd_metric::probabilistic=mode=="probabilistic";}
            else if(arg=="--normal-stddev" && i+1<argc)pd_metric::normal_stddev=std::stod(argv[++i]);
            else if(arg=="--position-stddev-relative" && i+1<argc)pd_metric::position_stddev_relative=std::stod(argv[++i]);
            else targets.push_back(std::stoi(arg));
        }
        if(targets.empty() || !std::isfinite(pd_metric::normal_stddev) || !std::isfinite(pd_metric::position_stddev_relative) ||
           pd_metric::normal_stddev<0 || pd_metric::position_stddev_relative<0)throw std::runtime_error("Invalid quadric parameters");
        const std::filesystem::path output(argv[2]);
        if(std::filesystem::exists(output) && !std::filesystem::is_empty(output))
            throw std::runtime_error("Refusing nonempty output directory");
        std::filesystem::create_directories(output);
        std::ofstream settings(output/"metric.json");
        settings<<std::setprecision(17)<<"{\"quadric\":\""<<(pd_metric::probabilistic?"probabilistic":"ordinary")
                <<"\",\"normal_stddev\":"<<pd_metric::normal_stddev<<",\"position_stddev_relative\":"<<pd_metric::position_stddev_relative
                <<",\"lock_boundary\":"<<(pd_metric::lock_boundary?"true":"false")<<"}\n";
        MatrixXd V; MatrixXi F;
        if(!igl::readOBJ(argv[1],V,F)) throw std::runtime_error("Cannot read input mesh");
        if(!V.allFinite() || F.cols()!=3) throw std::runtime_error("Need finite triangle mesh");
        MatrixXd fine_bc; MatrixXi fine_bf; VectorXi fine_fi;
        vertex_queries(F,static_cast<int>(V.rows()),fine_bc,fine_bf,fine_fi);
        igl::writeOBJ((output/"mesh_0.obj").string(),V,F);
        for(int level=1;level<=targets.size();++level) {
            const int target=targets[level-1];
            if(target<4 || target>=F.rows()) throw std::runtime_error("Face targets must decrease");
            MatrixXd C,bc; MatrixXi G,bf; VectorXi J,IM,FIM,fi;
            std::vector<single_collapse_data> history;
            std::vector<std::vector<int>> face_history;
            vertex_queries(F,static_cast<int>(V.rows()),bc,bf,fi);
            bool reached=SSP_decimate(V,F,target,0,C,G,J,IM,history,face_history,FIM);
            if(C.rows()==0 || G.rows()>=F.rows()) throw std::runtime_error("Decimation made no progress");
            query_fine_to_coarse(history,IM,face_history,FIM,bc,bf,fi);
            // Propagate original finest vertex anchors through the SAME collapse
            // history. Direct U is composed separately by the Python loader.
            query_fine_to_coarse(history,IM,face_history,FIM,fine_bc,fine_bf,fine_fi);
            if(!bc.allFinite() || !fine_bc.allFinite()) throw std::runtime_error("Nonfinite intrinsic map");
            auto suffix=std::to_string(level);
            export_expansions(output/("expansion_"+suffix+".json"),static_cast<int>(V.rows()),IM,history);
            igl::writeOBJ((output/("mesh_"+suffix+".obj")).string(),C,G);
            std::ofstream map(output/("map_"+suffix+".txt"));
            map<<std::setprecision(17)<<V.rows()<<" "<<C.rows()<<" "<<3*bc.rows()<<"\n";
            for(int i=0;i<bc.rows();++i) for(int j=0;j<3;++j)
                map<<i<<" "<<bf(i,j)<<" "<<bc(i,j)<<"\n";
            std::ofstream anchor(output/("anchors_"+suffix+".txt")); anchor<<fi<<"\n";
            std::ofstream direct(output/("finest_anchors_"+suffix+".txt")); direct<<fine_fi<<"\n";
            std::ofstream info(output/("info_"+suffix+".txt"));
            info<<"target "<<target<<"\nactual "<<G.rows()<<"\nvertices "<<C.rows()<<"\nreached "<<reached<<"\n";
            std::cout<<"INTRINSIC_LEVEL "<<level<<" vertices "<<C.rows()<<" faces "<<G.rows()<<" target_reached "<<reached<<std::endl;
            V=std::move(C); F=std::move(G);
        }
        return 0;
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<std::endl; return 1;
    }
}
