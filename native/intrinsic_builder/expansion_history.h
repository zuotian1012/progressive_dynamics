// SPDX-License-Identifier: GPL-3.0-or-later
// Lossless finite patch records from the author's joint parameterization.
#pragma once
#include <single_collapse_data.h>
#include <fstream>
#include <iomanip>

template<class M> void json_matrix(std::ostream& out,const M& a) {
    out<<"[";
    for(int i=0;i<a.rows();++i){if(i)out<<",";out<<"[";
        for(int j=0;j<a.cols();++j){if(j)out<<",";out<<a(i,j);}out<<"]";}
    out<<"]";
}
inline void export_expansions(const std::filesystem::path& path,int vertex_count,
    const Eigen::VectorXi& coarse_ids,const std::vector<single_collapse_data>& history) {
    std::ofstream out(path);out<<std::setprecision(17);
    out<<"{\"schema\":1,\"fine_vertex_count\":"<<vertex_count<<",\"coarse_ids\":[";
    for(int i=0;i<coarse_ids.size();++i){if(i)out<<",";out<<coarse_ids(i);}out<<"],\"collapses\":[";
    for(size_t index=0;index<history.size();++index){
        if(index)out<<",";const auto& h=history[index];
        out<<"{\"survivor\":"<<h.subsetVIdx(h.b(0))<<",\"removed\":"<<h.subsetVIdx(h.b(1));
        std::vector<int> rows;
        for(int i=0;i<h.subsetVIdx.size();++i)if(h.subsetVIdx(i)<vertex_count)rows.push_back(i);
        out<<",\"ids\":[";for(size_t i=0;i<rows.size();++i){if(i)out<<",";out<<h.subsetVIdx(rows[i]);}out<<"]";
        auto field=[&](const char* name,const Eigen::MatrixXd& a){
            Eigen::MatrixXd finite(rows.size(),a.cols());
            for(size_t i=0;i<rows.size();++i)finite.row(i)=a.row(rows[i]);
            if(!finite.allFinite())throw std::runtime_error("Nonfinite expansion history");
            out<<",\""<<name<<"\":";json_matrix(out,finite);
        };
        field("pre_rest",h.V_pre);field("post_rest",h.V_post);
        field("pre_uv",h.UV_pre);field("post_uv",h.UV_post);
        auto faces=[&](const char* name,const Eigen::MatrixXi& a){
            out<<",\""<<name<<"\":[";bool first=true;
            for(int i=0;i<a.rows();++i){bool valid=true;
                for(int j=0;j<3;++j)valid&=h.subsetVIdx(a(i,j))<vertex_count;
                if(!valid)continue;if(!first)out<<",";first=false;out<<"[";
                for(int j=0;j<3;++j){if(j)out<<",";out<<h.subsetVIdx(a(i,j));}out<<"]";
            }out<<"]";
        };
        faces("pre_faces",h.FUV_pre);faces("post_faces",h.FUV_post);out<<"}";
    }out<<"]}\n";
}
