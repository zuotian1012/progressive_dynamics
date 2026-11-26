// Standalone CUDA replay of the production factor/apply kernels.
#include <cuda_runtime.h>
#include <Eigen/Dense>
#include <cassert>
#include <vector>
#include <cmath>
#define UIPC_KERNEL_ASSERT(condition, ...) assert(condition)
using IndexT = int;
constexpr int BANKSIZE = 16;
namespace cuda_tool {
template<class T> struct BufferView { T* p; __device__ T& operator()(int i) const { return p[i]; } };
template<class T> struct CBufferView { const T* p; __device__ const T& operator()(int i) const { return p[i]; } };
template<class T> struct VarView { T* p; __device__ T* data() const { return p; } };
template<class T> struct CDense { const T* p; __device__ const T& operator*() const { return *p; } };
}
struct ClusterMatrixSym { Eigen::Matrix3d M[136]; };
struct MASPreconditionerEngine {
    struct ClusterFactor { double lower[2304]; double scale[48]; double shift; };
};
__host__ __device__ int sym_index(int r, int c) { return 16*r-r*(r+1)/2+c; }
#include "mas_factor_kernels.inl"

extern "C" __declspec(dllexport) int evaluate(int n, const double* matrices, const double* rhs,
                                              double* factors, double* result)
{
    std::vector<ClusterMatrixSym> h(n);
    for(int b=0;b<n;++b) for(int br=0;br<16;++br) for(int bc=br;bc<16;++bc)
        for(int i=0;i<3;++i) for(int j=0;j<3;++j)
            h[b].M[sym_index(br,bc)](i,j)=matrices[b*2304+(br*3+i)*48+bc*3+j];
    ClusterMatrixSym* dh=nullptr;
    ClusterFactor* df=nullptr;
    Eigen::Vector3d* dr=nullptr;
    double3* dz=nullptr;
    int* flags=nullptr;
    cudaError_t error=cudaSuccess;
#define CHECK(call) if((error=(call))!=cudaSuccess) goto cleanup
    CHECK(cudaMalloc(&dh,n*sizeof(ClusterMatrixSym)));
    CHECK(cudaMalloc(&df,n*sizeof(ClusterFactor)));
    CHECK(cudaMalloc(&dr,n*16*sizeof(Eigen::Vector3d)));
    CHECK(cudaMalloc(&dz,n*16*sizeof(double3)));
    CHECK(cudaMalloc(&flags,2*sizeof(int)));
    CHECK(cudaMemset(flags,0,2*sizeof(int)));
    CHECK(cudaMemcpy(dh,h.data(),n*sizeof(ClusterMatrixSym),cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dr,rhs,n*48*sizeof(double),cudaMemcpyHostToDevice));
    MASPreconditionerEngine_factor_kernel<<<n,64>>>({df},{dh},{flags});
    CHECK(cudaGetLastError());
    MASPreconditionerEngine_factor_solve_kernel<<<(n+3)/4,128>>>({df},{dr},{dz},{flags+1},n);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(factors,df,n*sizeof(ClusterFactor),cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(result,dz,n*48*sizeof(double),cudaMemcpyDeviceToHost));
cleanup:
    cudaFree(dh);cudaFree(df);cudaFree(dr);cudaFree(dz);cudaFree(flags);
    return static_cast<int>(error);
}
