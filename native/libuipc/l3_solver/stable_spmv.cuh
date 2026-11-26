#pragma once
#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>

// Error-free binary64 products/sums. The low word is retained until after
// cancellation; no matrix entries, physical energies, or tolerances change.
namespace uipc_stable_linear {
struct DD { double hi, lo; };
__device__ inline DD add(DD a, DD b) {
    double s=__dadd_rn(a.hi,b.hi), v=__dsub_rn(s,a.hi);
    double e=__dadd_rn(__dsub_rn(a.hi,__dsub_rn(s,v)),__dsub_rn(b.hi,v));
    e=__dadd_rn(e,__dadd_rn(a.lo,b.lo));
    double h=__dadd_rn(s,e);
    return {h,__dsub_rn(e,__dsub_rn(h,s))};
}
__device__ inline DD product(double a,double b) {
    double h=__dmul_rn(a,b);return {h,fma(a,b,-h)};
}
__device__ inline DD multiply(DD a,double b) {
    return add(product(a.hi,b),product(a.lo,b));
}
struct Plus { __device__ DD operator()(DD a,DD b) const {return add(a,b);} };
__device__ inline void atomic_add(double* high,double* low,DD a) {
    double old=atomicAdd(high,a.hi);
    double s=__dadd_rn(old,a.hi),v=__dsub_rn(s,old);
    double e=__dadd_rn(__dsub_rn(old,__dsub_rn(s,v)),__dsub_rn(a.hi,v));
    atomicAdd(low,__dadd_rn(e,a.lo));
}
static __global__ void matvec(const int* rows,const int* cols,const double* blocks,
                              const int* count,const double* x,double* y,double* low,
                              double* dot,double* dotlow) {
    constexpr int threads=128;
    using Reduce=cub::BlockReduce<DD,threads>;
    __shared__ typename Reduce::TempStorage temp;
    int k=blockIdx.x*threads+threadIdx.x;
    DD energy{0,0};
    if(k<*count) {
        int i=rows[k],j=cols[k];const double* a=blocks+9*k;
        for(int r=0;r<3;++r) {
            DD value{0,0};
            for(int c=0;c<3;++c)value=add(value,product(a[r+3*c],x[3*j+c]));
            atomic_add(y+3*i+r,low+3*i+r,value);
            energy=add(energy,multiply(value,x[3*i+r]));
        }
        if(i!=j) {
            energy=multiply(energy,2.0);
            for(int r=0;r<3;++r) {
                DD value{0,0};
                for(int c=0;c<3;++c)value=add(value,product(a[c+3*r],x[3*i+c]));
                atomic_add(y+3*j+r,low+3*j+r,value);
            }
        }
    }
    DD sum=Reduce(temp).Reduce(energy,Plus{});
    if(threadIdx.x==0)atomic_add(dot,dotlow,sum);
}
static __global__ void finalize(double* y,const double* low,int n,double* dot,const double* dotlow) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n)y[i]=__dadd_rn(y[i],low[i]);
    if(i==0)*dot=__dadd_rn(*dot,*dotlow);
}
inline cudaError_t launch(const int* rows,const int* cols,const double* blocks,
                         const int* count,int capacity,const double* x,double* y,int n,
                         double* dot,cudaStream_t stream) {
    double *scratch=nullptr;
    auto status=cudaMallocAsync((void**)&scratch,(n+1)*sizeof(double),stream);
    if(status!=cudaSuccess)return status;
    cudaMemsetAsync(scratch,0,(n+1)*sizeof(double),stream);
    cudaMemsetAsync(y,0,n*sizeof(double),stream);cudaMemsetAsync(dot,0,sizeof(double),stream);
    matvec<<<(capacity+127)/128,128,0,stream>>>(rows,cols,blocks,count,x,y,scratch,dot,scratch+n);
    finalize<<<(n+255)/256,256,0,stream>>>(y,scratch,n,dot,scratch+n);
    status=cudaGetLastError();cudaFreeAsync(scratch,stream);return status;
}
}
