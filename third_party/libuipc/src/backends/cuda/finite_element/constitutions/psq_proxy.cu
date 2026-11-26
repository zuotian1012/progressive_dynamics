#include "psq_proxy.h"
#include "psq_kernels.cuh"
#include "psq_exact_offset.cuh"
#include "psq_step_cap.h"
#include <iomanip>
#include <cuda_tool/cub.h>
#include <finite_element/finite_element_method.h>
#include <finite_element/constitutions/neo_hookean_shell_2d_function.h>
#include <finite_element/constitutions/discrete_shell_bending_function.h>
#include <utils/matrix_assembler.h>
#include <uipc/common/log.h>
#include <fstream>
#include <cstdlib>
#include <map>
#include <mutex>
#include <stdexcept>
#include <cstring>

namespace uipc::backend::cuda
{
namespace {
namespace NH=sym::neo_hookean_shell_2d;
namespace DSB=sym::discrete_shell_bending;
std::mutex registry_mutex;
std::map<FiniteElementMethod*,PsqProxy*> registry;

void check(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
template<class T> vector<T> read(std::ifstream& f, SizeT n) {
    vector<T> v(n); f.read(reinterpret_cast<char*>(v.data()),n*sizeof(T));
    check(bool(f),"Truncated PSQ bundle"); return v;
}
template<class T> void upload(cuda_tool::DeviceBuffer<T>& d,const vector<T>& h) {
    d.resize(h.size()); if(!h.empty()) d.view().copy_from(h.data());
}

#include "bunny_gradient_gather.inl"

__global__ void proxy_membrane(int n, const Vector3i* faces, const Matrix2x2* IB,
    const Float* areas, const Vector3* x, Float* energies, Vector3* gradient,
    Float lambda,Float mu,Float thickness,Float dt,bool derivatives) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    Vector3i ids=faces[i]; Vector9 X;
    for(int j=0;j<3;++j) X.segment<3>(3*j)=x[ids(j)];
    // The generated native NH polynomial expands absolute coordinates. At 90K
    // resolution this causes cancellation even in rigid free fall. Translation
    // invariance permits an element-local origin without changing E or dE/dx.
    Vector3 origin=X.segment<3>(0);
    for(int j=0;j<3;++j) X.segment<3>(3*j)-=origin;
    Float scale=areas[i]*thickness*dt*dt;
    if(derivatives) {
        Vector9 G; NH::dEdX(G,lambda,mu,X,IB[i]); G*=scale;
        for(int j=0;j<3;++j) gradient[3*i+j] = G.segment<3>(3*j);
    } else {
        Float E; NH::E(E,lambda,mu,X,IB[i]);
        energies[i]=isfinite(E)? E*scale: INFINITY;
    }
}

__global__ void proxy_bending(int n,const Vector4i* ids,const Vector4* constants,
    const Vector3* x,Float* energies,Vector3* gradient,Float stiffness,Float dt,bool derivatives) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    Vector4i t=ids[i]; Vector4 c=constants[i];
    if(derivatives) {
        Vector12 G; DSB::dEdx(G,x[t(0)],x[t(1)],x[t(2)],x[t(3)],c(0),c(1),c(2),stiffness);
        G*=c(3)*dt*dt;
        for(int j=0;j<4;++j) gradient[4*i+j] = G.segment<3>(3*j);
    } else {
        Float E=DSB::E(x[t(0)],x[t(1)],x[t(2)],x[t(3)],c(0),c(1),c(2),stiffness);
        energies[i]=isfinite(E)? E*c(3)*dt*dt : INFINITY;
    }
}

__global__ void write_proxy_gradient(int n,int offset,const Vector3* g,cuda_tool::DoubletVectorView<Float,3> out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    DoubletVectorAssembler DVA{out}; DVA.segment<1>(i).write(offset+i,g[i]);
}

__global__ void diagnostic_frozen_prolong(int n,const int* row,const int* col,
    const Float* val,const Vector3* x,const Vector3* base_x,const Vector3* base_y,Vector3* y) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    Vector3 delta=Vector3::Zero();
    for(int k=row[i];k<row[i+1];++k) delta+=val[k]*(x[col[k]]-base_x[col[k]]);
    y[i]=base_y[i]+delta;
}

__global__ void strain_step(int n, const Vector3i* faces,const Matrix2x2* IB,
    cuda_tool::CBufferView<Vector3> x,cuda_tool::CBufferView<Vector3> dx,Float limit,Float* cap,
    bool directional,Float* audit) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    Vector9 X,D; for(int j=0;j<3;++j) { X.segment<3>(3*j)=x(faces[i](j)); D.segment<3>(3*j)=dx(faces[i](j)); }
    Matrix3x3 U; Matrix2x2 V; Vector2 s;
    Matrix2x2 inverse_frame=psq_native::inverse_rest_frame(IB[i]);
    psq_native::Mat32 F=psq_native::deformation(X,inverse_frame), dF=psq_native::deformation(D,inverse_frame);
    psq_native::svd32(F,U,V,s);
    Float norm=dF.norm();
    // Weyl's inequality: this conservative bound keeps both stretch and rank
    // feasible for the entire segment. Native CCD still handles intersections.
    Float gap=fmin(limit-s(0),s(1)-1e-12);
    Float old_cap=gap>0 ? (norm>0 ? fmin(1.0,0.9*gap/norm):1.0) : 0.0;
    Float new_cap=(directional || audit) ? psq_step_cap::directional(F.data(),dF.data(),limit,s(0),s(1),norm,old_cap) : old_cap;
    cap[i]=directional ? new_cap : old_cap;
    if(audit) {
        Float* a=audit+17*i; a[0]=old_cap; a[1]=new_cap; a[2]=s(0);a[3]=s(1);a[4]=norm;
        for(int j=0;j<6;++j) { a[5+j]=F.data()[j];a[11+j]=dF.data()[j]; }
    }
}
}

struct PsqProxy::Impl {
    bool enabled=false, replace=false;
    bool directional=false;
    bool exact_offset=false;
    std::ofstream cap_audit;
    int nc=0,nf=0,nt=0,nh=0,offset=0;
    Float lambda=0,mu=0,thickness=0,bending=0;
    psq_native::StrainParameters strain{1.068,1.034,1000.0};
    FiniteElementMethod* fem=nullptr;
    cuda_tool::DeviceBuffer<int> row,col,coarse_faces_raw,anchors,invalid;
    cuda_tool::DeviceBuffer<Float> val,inv_dm,gamma,tangent,areas,steps,step_min;
    cuda_tool::DeviceBuffer<Float> cap_records;
    cuda_tool::DeviceBuffer<Vector3i> faces,coarse_faces;
    cuda_tool::DeviceBuffer<Matrix2x2> IB,coarse_IB;
    cuda_tool::DeviceBuffer<Vector4i> hinges;
    cuda_tool::DeviceBuffer<Vector4> hinge_constants;
    cuda_tool::DeviceBuffer<Vector3> y,fg,cg;
    OrderedPsqGradients ordered;
    bool diagnostic_frozen=false;
    cuda_tool::DeviceBuffer<Vector3> diagnostic_x,diagnostic_y;
    void prolong(cuda_tool::CBufferView<Vector3> x) {
        cudaMemset(invalid.data(),0,sizeof(int));
        if(diagnostic_frozen) {
            diagnostic_frozen_prolong<<<(nf+127)/128,128>>>(nf,row.data(),col.data(),val.data(),
                x.data()+offset,diagnostic_x.data(),diagnostic_y.data(),y.data());
            return;
        }
        psq::prolong<<<(nf+127)/128,128>>>(nf,row.data(),col.data(),val.data(),
            coarse_faces_raw.data(),anchors.data(),inv_dm.data(),gamma.data(),tangent.data(),
            reinterpret_cast<const double*>(x.data()+offset),reinterpret_cast<double*>(y.data()),invalid.data());
    }
};

PsqProxy::PsqProxy():m(std::make_unique<Impl>()) {}
PsqProxy::~PsqProxy() {
    std::lock_guard<std::mutex> lock(registry_mutex);
    if(m->fem) registry.erase(m->fem);
}
bool PsqProxy::active()const{return m->enabled;}
bool PsqProxy::replaces_elastic()const{return m->replace;}
bool PsqProxy::preview_eligible()const{return m->replace && m->nf>m->nc;}
bool psq_preview_eligible(FiniteElementMethod& fem) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it=registry.find(&fem);return it!=registry.end() && it->second->preview_eligible();
}
SizeT PsqProxy::energy_count()const{return m->replace ? m->nt+m->nh : 0;}
SizeT PsqProxy::gradient_count()const{return m->replace ? m->nc : 0;}
psq_native::StrainParameters PsqProxy::strain_parameters()const{return m->strain;}
bool psq_replaces_bending() {
    auto path=std::getenv("UIPC_PSQ_BUNDLE"); auto mode=std::getenv("UIPC_PSQ_MODE");
    return path && *path && (!mode || std::strcmp(mode,"reference")!=0);
}

void PsqProxy::init(FiniteElementMethod& fem,cuda_tool::CBufferView<Vector3i> prims,
    cuda_tool::CBufferView<Matrix2x2> coarse_IB,const vector<Float>& lambdas,const vector<Float>& mus) {
    const char* path=std::getenv("UIPC_PSQ_BUNDLE"); if(!path || !*path) return;
    std::ifstream f(path,std::ios::binary); check(bool(f),"Cannot open PSQ bundle");
    auto magic=read<char>(f,8); check(std::string(magic.data(),8)=="PSQIPC01","Wrong PSQ bundle version");
    auto dims=read<int>(f,6); m->nc=dims[0]; int nct=dims[1];m->nf=dims[2];m->nt=dims[3];int nnz=dims[4];m->nh=dims[5];
    check(m->nc>0 && nct==(int)prims.size() && m->nf>0 && m->nt>0 && nnz>0 && m->nh>0,
          "PSQ requires exactly one matching NeoHookean cloth");
    auto parameters=read<Float>(f,7);
    m->lambda=parameters[0];m->mu=parameters[1];m->thickness=parameters[2];m->bending=parameters[3];
    m->strain={parameters[4],parameters[5],parameters[6]};
    check(m->strain.limit>m->strain.onset && m->strain.onset>=1 && m->strain.stiffness>0,"Invalid strain parameters");
    check(m->thickness>0 && m->bending>0,"Invalid PSQ shell material");
    for(SizeT i=0;i<lambdas.size();++i)
        check(fabs(lambdas[i]-m->lambda)<1e-8 && fabs(mus[i]-m->mu)<1e-8,"PSQ material mismatch");
    auto coarse_rest=read<Vector3>(f,m->nc);
    auto coarse_faces=read<Vector3i>(f,nct);
    vector<Vector3> native_rest(fem.x_bars().size());fem.x_bars().copy_to(native_rest.data());
    vector<Vector3i> native_faces(nct);prims.copy_to(native_faces.data());
    vector<Float> native_thickness(fem.thicknesses().size());fem.thicknesses().copy_to(native_thickness.data());
    m->offset=native_faces.front().minCoeff();
    for(auto t:native_faces)m->offset=std::min(m->offset,t.minCoeff());
    check(m->offset>=0 && m->offset+m->nc<=(int)native_rest.size(),"PSQ vertex count mismatch");
    vector<IndexT> fixed(fem.is_fixed().size());fem.is_fixed().copy_to(fixed.data());
    for(SizeT i=0;i<fixed.size();++i) if(i<m->offset || i>=m->offset+m->nc)
        check(fixed[i]!=0,"PSQ prototype only supports fixed additional FEM geometry");
    for(int i=0;i<m->nc;++i) check((coarse_rest[i]-native_rest[m->offset+i]).norm()<1e-12 &&
        fabs(2*native_thickness[m->offset+i]-m->thickness)<1e-12,"PSQ cloth rest/thickness mismatch");
    for(int i=0;i<nct;++i) check(coarse_faces[i]==(native_faces[i].array()-m->offset).matrix(),"PSQ face ordering mismatch");
    upload(m->coarse_faces,native_faces);
    vector<int> cf(nct*3);for(int i=0;i<nct;++i) for(int j=0;j<3;++j) cf[3*i+j]=coarse_faces[i](j);
    upload(m->coarse_faces_raw,cf);
    vector<Matrix2x2> cib(nct);coarse_IB.copy_to(cib.data());upload(m->coarse_IB,cib);
    auto rest=read<Vector3>(f,m->nf);auto faces=read<Vector3i>(f,m->nt);upload(m->faces,faces);
    auto row=read<int>(f,m->nf+1); auto col=read<int>(f,nnz);auto val=read<Float>(f,nnz);
    check(row.front()==0 && row.back()==nnz,"Invalid PSQ CSR offsets");
    for(int i=0;i<m->nf;++i) check(row[i]<=row[i+1],"Nonmonotone PSQ CSR");
    for(auto v:col) check(v>=0 && v<m->nc,"Invalid PSQ CSR column");
    upload(m->row,row); upload(m->col,col);upload(m->val,val);
    auto anchors=read<int>(f,m->nf);for(auto v:anchors)check(v>=0&&v<nct,"Invalid PSQ anchor");
    upload(m->anchors,anchors);upload(m->inv_dm,read<Float>(f,nct*4));
    upload(m->gamma,read<Float>(f,m->nf));upload(m->tangent,read<Float>(f,m->nf*2));
    auto hinges=read<Vector4i>(f,m->nh);upload(m->hinges,hinges);
    check(f.peek()==std::ifstream::traits_type::eof(),"Unexpected PSQ bundle trailing bytes");
    vector<Matrix2x2> ib(m->nt);vector<Float> areas(m->nt);
    for(int i=0;i<m->nt;++i) {
        Vector9 X;for(int j=0;j<3;++j) {check(faces[i](j)>=0&&faces[i](j)<m->nf,"Invalid proxy face");X.segment<3>(3*j)=rest[faces[i](j)];}
        Matrix2x2 B;NH::A(B,X);ib[i]=B.inverse();areas[i]=0.5*(rest[faces[i](1)]-rest[faces[i](0)]).cross(rest[faces[i](2)]-rest[faces[i](0)]).norm();
    }
    upload(m->IB,ib);upload(m->areas,areas);
    vector<Vector4> hc(m->nh);
    for(int i=0;i<m->nh;++i) {
        auto t=hinges[i];for(int j=0;j<4;++j)check(t(j)>=0&&t(j)<m->nf,"Invalid proxy hinge");
        DSB::compute_constants(hc[i](0),hc[i](1),hc[i](2),hc[i](3),rest[t(0)],rest[t(1)],rest[t(2)],rest[t(3)],0.,0.,0.,0.);
    }
    upload(m->hinge_constants,hc);
    m->y.resize(m->nf);m->fg.resize(m->nf);m->cg.resize(m->nc);m->invalid.resize(1);
    m->steps.resize(nct);m->step_min.resize(1);
    const char* mode=std::getenv("UIPC_PSQ_STRAIN_CAP");
    check(!mode || std::string(mode)=="legacy" || std::string(mode)=="directional","Unknown PSQ strain cap mode");
    m->directional=mode && std::string(mode)=="directional";
    if(const char* path=std::getenv("UIPC_PSQ_CAP_AUDIT")) {
        m->cap_audit.open(path);check(bool(m->cap_audit),"Cannot open PSQ cap audit file");
        m->cap_audit<<std::setprecision(17);m->cap_records.resize(17*nct);
    }
    logger::info("PSQ_STRAIN_CAP mode={} audit={}",m->directional?"directional":"legacy",m->cap_audit.is_open());
    m->exact_offset=std::getenv("UIPC_PSQ_EXACT_OFFSET") && std::string(std::getenv("UIPC_PSQ_EXACT_OFFSET"))=="1";
    logger::info("FLAG_PSQ_EXACT_OFFSET enabled={}",m->exact_offset);
    m->replace=psq_replaces_bending();m->enabled=true;m->fem=&fem;
    if(m->replace) m->ordered.build(m->nf,m->nc,faces,hinges,row,col,val,anchors,coarse_faces);
    { std::lock_guard<std::mutex> lock(registry_mutex);check(!registry.count(&fem),"Duplicate PSQ cloth adapter");registry[&fem]=this; }
    logger::info("PSQ_NATIVE_CONNECTED coarse={} proxy={} triangles={} hinges={} offset={} replace={} strain_limit={}",
        m->nc,m->nf,m->nt,m->nh,m->offset,m->replace,m->strain.limit);
}

void PsqProxy::energy(cuda_tool::CBufferView<Vector3> x,cuda_tool::BufferView<Float> E,Float dt) {
    if(!m->replace)return;
    m->prolong(x);
    proxy_membrane<<<(m->nt+127)/128,128>>>(m->nt,m->faces.data(),m->IB.data(),m->areas.data(),m->y.data(),E.data(),nullptr,m->lambda,m->mu,m->thickness,dt,false);
    proxy_bending<<<(m->nh+127)/128,128>>>(m->nh,m->hinges.data(),m->hinge_constants.data(),m->y.data(),E.data()+m->nt,nullptr,m->bending,dt,false);
}
void PsqProxy::gradient(cuda_tool::CBufferView<Vector3> x,cuda_tool::DoubletVectorView<Float,3> G,Float dt) {
    if(!m->replace)return;
    m->prolong(x);
    cudaMemset(m->fg.data(),0,m->nf*sizeof(Vector3));cudaMemset(m->cg.data(),0,m->nc*sizeof(Vector3));
    proxy_membrane<<<(m->nt+127)/128,128>>>(m->nt,m->faces.data(),m->IB.data(),m->areas.data(),m->y.data(),nullptr,m->ordered.slots.data(),m->lambda,m->mu,m->thickness,dt,true);
    proxy_bending<<<(m->nh+127)/128,128>>>(m->nh,m->hinges.data(),m->hinge_constants.data(),m->y.data(),nullptr,m->ordered.slots.data()+3*m->nt,m->bending,dt,true);
    m->ordered.elements.sum(m->ordered.slots.data(),m->fg.data());
    m->ordered.transpose.sum(m->fg.data(),m->cg.data());
    const bool exact_once=std::getenv("UIPC_PSQ_EXACT_ONCE") && std::string(std::getenv("UIPC_PSQ_EXACT_ONCE"))=="1";
    if(m->exact_offset || exact_once) {
        psq::restrict_offset_gradient<<<(m->nf+127)/128,128>>>(m->nf,
        m->coarse_faces_raw.data(),m->anchors.data(),m->inv_dm.data(),m->gamma.data(),m->tangent.data(),
        reinterpret_cast<const double*>(x.data()+m->offset),reinterpret_cast<const double*>(m->fg.data()),
        reinterpret_cast<double*>(m->ordered.slots.data()));
        m->ordered.offsets.sum(m->ordered.slots.data(),m->cg.data(),true);
    }
    if(exact_once) _putenv_s("UIPC_PSQ_EXACT_ONCE","0");
    write_proxy_gradient<<<(m->nc+127)/128,128>>>(m->nc,m->offset,m->cg.data(),G);
    int invalid=0;m->invalid.cview().copy_to(&invalid);check(!invalid,"PSQ proxy is degenerate at Newton assembly");
}
Float PsqProxy::feasible_step(Float alpha,SizeT frame,SizeT iteration) {
    int n=(int)m->coarse_faces.size();
    strain_step<<<(n+127)/128,128>>>(n,m->coarse_faces.data(),m->coarse_IB.data(),m->fem->xs(),m->fem->dxs(),m->strain.limit,m->steps.data(),m->directional,m->cap_audit.is_open()?m->cap_records.data():nullptr);
    cuda_tool::DeviceReduce().Min(m->steps.data(),m->step_min.data(),n);
    Float result; m->step_min.cview().copy_to(&result);
    if(m->cap_audit.is_open() && (!(result>0) || iteration==0 || iteration==1023 || (result<.001 && iteration%100==0))) {
        vector<Float> rows(17*n);m->cap_records.cview().copy_to(rows.data());
        int old_i=0,new_i=0,previous_i=0,near_count=0,recovered_count=0;
        for(int i=1;i<n;++i) { if(rows[17*i]<rows[17*old_i])old_i=i; if(rows[17*i+1]<rows[17*new_i+1])new_i=i; }
        Float target=m->strain.limit-64*DBL_EPSILON*fmax(1.,m->strain.limit);
        // Read-only evaluation of the preceding inward-recovery candidate
        // on the same native inputs, over ALL faces, for an honest control.
        auto previous_cap=[&](int i) {
            const Float* r=rows.data()+17*i;
            if(!(r[2]<m->strain.limit) || !(r[3]>1e-12) || !(r[4]>=0))return Float(0);
            if(r[4]==0)return Float(1);
            if(r[2]<target)return r[1];
            Float trial=fmin(1.,.9*(r[3]-1e-12)/r[4]);
            for(int k=0;k<48 && trial>r[0];++k,trial*=.5)
                if(psq_step_cap::maximum(r+5,r+11,trial)<target)return trial;
            return r[0];
        };
        for(int i=0;i<n;++i) {
            if(previous_cap(i)<previous_cap(previous_i))previous_i=i;
            if(rows[17*i+2]>=target) {
                ++near_count;
                if(rows[17*i+1]>previous_cap(i))++recovered_count;
            }
        }
        auto& out=m->cap_audit;
        out<<"{\"frame\":"<<frame<<",\"iteration\":"<<iteration<<",\"algorithm\":\"compensated_boundary_v2\",\"reference_formula\":\"inward_recovery_v1\",\"directional\":"<<(m->directional?"true":"false")
           <<",\"incoming_alpha\":"<<alpha<<",\"legacy_min\":"<<rows[17*old_i]<<",\"directional_min\":"<<rows[17*new_i+1]
           <<",\"previous_directional_min\":"<<previous_cap(previous_i)<<",\"near_boundary_faces\":"<<near_count
           <<",\"recovered_boundary_faces\":"<<recovered_count<<",\"faces\":[";
        int count=(previous_i!=old_i && previous_i!=new_i)?3:2;
        for(int k=0;k<count;++k) {
            int i=k==0?old_i:(k==1?new_i:previous_i); if(k)out<<",";
            out<<"{\"face\":"<<i<<",\"values\":[";
            for(int j=0;j<17;++j) {if(j)out<<",";out<<rows[17*i+j];} out<<"]}";
        }
        out<<"]}\n";out.flush();
    }
    check(result>0 && std::isfinite(result),"PSQ current state violates strain/rank feasibility");
    return std::min(alpha,result);
}
void PsqProxy::diagnostic_freeze_offset(bool enabled) {
    m->diagnostic_frozen=false;
    if(!enabled || !m->replace)return;
    m->prolong(m->fem->xs());
    m->diagnostic_x.resize(m->nc);m->diagnostic_y.resize(m->nf);
    CUDA_TOOL_CHECK(cudaMemcpy(m->diagnostic_x.data(),m->fem->xs().data()+m->offset,
        m->nc*sizeof(Vector3),cudaMemcpyDeviceToDevice));
    CUDA_TOOL_CHECK(cudaMemcpy(m->diagnostic_y.data(),m->y.data(),
        m->nf*sizeof(Vector3),cudaMemcpyDeviceToDevice));
    m->diagnostic_frozen=true;
}
void psq_diagnostic_freeze_offset(FiniteElementMethod& fem,bool enabled) {
    std::lock_guard<std::mutex> lock(registry_mutex);
    auto it=registry.find(&fem);
    if(it!=registry.end())it->second->diagnostic_freeze_offset(enabled);
}
Float psq_feasible_step(FiniteElementMethod& fem,Float alpha,SizeT frame,SizeT iteration) {
    try {
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto it=registry.find(&fem);return it==registry.end()?alpha:it->second->feasible_step(alpha,frame,iteration);
    } catch(const std::exception& e) {
        logger::error("PSQ feasible-step failure: {}",e.what());
        throw;
    }
}
}
