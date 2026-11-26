// Small host-buffer harness for the reusable device kernels. Not a solver,
// not a libuipc plugin. Allocations/copies here are for one-shot verification.
#include "psq_kernels.cuh"
#include <vector>
#include <stdexcept>

struct Buffers {
    std::vector<void*> ptrs;
    ~Buffers(){for(void* p:ptrs) cudaFree(p);}
    template<class T> T* copy(const T* h,size_t n) {
        T* p=nullptr;
        if(cudaMalloc((void**)&p,n*sizeof(T))!=cudaSuccess) throw std::runtime_error("alloc");
        ptrs.push_back(p);
        auto err=h ? cudaMemcpy(p,h,n*sizeof(T),cudaMemcpyHostToDevice) : cudaMemset(p,0,n*sizeof(T));
        if(err!=cudaSuccess) throw std::runtime_error("copy");
        return p;
    }
};
extern "C" __declspec(dllexport) int psq_check(
    int nf,int nc,int nt,int nnz,const int* row,const int* col,const double* val,
    const int* faces,const int* anchors,const double* inv_dm,const double* gamma,
    const double* tangent,const double* x,const double* fine_g,double* y,double* coarse_g) {
    try {
        Buffers m;
        auto dr=m.copy(row,nf+1),dc=m.copy(col,nnz),df=m.copy(faces,3*nt),da=m.copy(anchors,nf);
        auto dv=m.copy(val,nnz),di=m.copy(inv_dm,4*nt),dg=m.copy(gamma,nf),dt=m.copy(tangent,2*nf);
        auto dx=m.copy(x,3*nc),dgy=m.copy(fine_g,3*nf);
        auto dy=m.copy<double>(nullptr,3*nf),dgx=m.copy<double>(nullptr,3*nc);
        auto bad=m.copy<int>(nullptr,1);
        psq::prolong<<<(nf+127)/128,128>>>(nf,dr,dc,dv,df,da,di,dg,dt,dx,dy,bad);
        psq::restrict_gradient<<<(nf+127)/128,128>>>(nf,dr,dc,dv,dgy,dgx);
        if(cudaDeviceSynchronize()!=cudaSuccess) return 2;
        int invalid=0;
        if(cudaMemcpy(&invalid,bad,sizeof(int),cudaMemcpyDeviceToHost)!=cudaSuccess) return 3;
        if(invalid) return 4;
        if(cudaMemcpy(y,dy,3*nf*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess) return 3;
        if(cudaMemcpy(coarse_g,dgx,3*nc*sizeof(double),cudaMemcpyDeviceToHost)!=cudaSuccess) return 3;
        return 0;
    } catch(...) {return 1;}
}
