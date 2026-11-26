#include "psq_step_cap.h"
#include <cuda_runtime.h>
__global__ void check_caps(int n,const double* rows,double* result) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const double* f=rows+12*i;const double* d=f+6;
    double zero[6]={0},smax=psq_step_cap::maximum(f,zero,0),norm=0;
    double cx=f[1]*f[5]-f[2]*f[4],cy=f[2]*f[3]-f[0]*f[5],cz=f[0]*f[4]-f[1]*f[3];
    double smin=sqrt(cx*cx+cy*cy+cz*cz)/fmax(smax,1e-30);
    for(int j=0;j<6;++j)norm+=d[j]*d[j];norm=sqrt(norm);
    double gap=fmin(1.1-smax,smin-1e-12);
    double legacy=gap>0 ? (norm>0?fmin(1.,.9*gap/norm):1.):0.;
    result[2*i]=legacy;
    result[2*i+1]=psq_step_cap::directional(f,d,1.1,smax,smin,norm,legacy);
}
extern "C" __declspec(dllexport) int evaluate(int n,const double* input,double* output) {
    double *x=nullptr,*y=nullptr;cudaError_t err=cudaMalloc(&x,n*12*sizeof(double));if(err)return err;
    err=cudaMalloc(&y,n*2*sizeof(double));if(err){cudaFree(x);return err;}
    err=cudaMemcpy(x,input,n*12*sizeof(double),cudaMemcpyHostToDevice);
    if(!err) { check_caps<<<(n+127)/128,128>>>(n,x,y);err=cudaGetLastError(); }
    if(!err)err=cudaMemcpy(output,y,n*2*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(x);cudaFree(y);return err;
}

// Replay the exact production SVD/norm inputs, avoiding a second SVD with a
// different floating-point evaluation order at a machine-precision boundary.
__global__ void recorded_caps(int n,const double* rows,double* result) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const double* r=rows+17*i;
    result[2*i]=r[0];
    result[2*i+1]=psq_step_cap::directional(r+5,r+11,1.1,r[2],r[3],r[4],r[0]);
}
extern "C" __declspec(dllexport) int evaluate_recorded(int n,const double* input,double* output) {
    double *x=nullptr,*y=nullptr;cudaError_t err=cudaMalloc(&x,n*17*sizeof(double));if(err)return err;
    err=cudaMalloc(&y,n*2*sizeof(double));if(err){cudaFree(x);return err;}
    err=cudaMemcpy(x,input,n*17*sizeof(double),cudaMemcpyHostToDevice);
    if(!err) { recorded_caps<<<(n+127)/128,128>>>(n,x,y);err=cudaGetLastError(); }
    if(!err)err=cudaMemcpy(output,y,n*2*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(x);cudaFree(y);return err;
}
