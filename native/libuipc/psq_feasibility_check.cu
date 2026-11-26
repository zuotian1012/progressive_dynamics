// Isolated GPU replay of the production strain energy, derivatives and cap.
#include <type_define.h>
#include <Eigen/Dense>
#include <cuda_runtime.h>
#include <cuda_tool/eigen/analytic_inverse.h>
#include <finite_element/constitutions/neo_hookean_shell_2d_function.h>
#include "psq_strain.cuh"
using namespace uipc;
using namespace uipc::backend::cuda;

__global__ void rest_matrices(int n,const double* rows,Matrix2x2* ib) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    Vector9 r;for(int j=0;j<9;++j)r(j)=rows[18*i+j];
    Matrix2x2 b;sym::neo_hookean_shell_2d::A(b,r);
    ib[i]=uipc::backend::cuda_tool::eigen::AnalyticalInverse{}(b);
}
__global__ void energies(int n,const double* rows,const Matrix2x2* ib,double* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    Vector9 x;for(int j=0;j<9;++j)x(j)=rows[18*i+9+j];
    out[16*i]=psq_native::strain(x,ib[i],{1.1,1.05,1000.});
}
__global__ void derivatives(int n,const double* rows,const Matrix2x2* ib,double* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    Vector9 x,g;Matrix9x9 h;for(int j=0;j<9;++j)x(j)=rows[18*i+9+j];
    out[16*i+1]=psq_native::strain(x,ib[i],{1.1,1.05,1000.},&g,&h);
    out[16*i+12]=g.norm();out[16*i+13]=h.norm();
}
__global__ void caps(int n,const double* rows,const Matrix2x2* ib,double* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    Vector9 x;for(int j=0;j<9;++j)x(j)=rows[18*i+9+j];
    auto frame=psq_native::inverse_rest_frame(ib[i]);
    auto f=psq_native::deformation(x,frame);
    Matrix3x3 u;Matrix2x2 v;Vector2 s;psq_native::svd32(f,u,v,s);
    double zero[6]={};
    out[16*i+2]=s(0);out[16*i+3]=s(1);
#ifdef CHECK_COORDINATE_GUARD
    out[16*i+4]=psq_native::coordinate_step(x,Vector9::Zero(),frame,1.1,f,psq_native::Mat32::Zero(),s,1.);
#else
    out[16*i+4]=psq_step_cap::directional(f.data(),zero,1.1,s(0),s(1),0.,1.);
#endif
    out[16*i+5]=psq_step_cap::inside_stretch(f.data(),zero,0.,1.1);
    for(int j=0;j<6;++j)out[16*i+6+j]=f.data()[j];
    out[16*i+14]=frame.norm();out[16*i+15]=frame.determinant();
}
extern "C" __declspec(dllexport) int evaluate(int n,const double* input,double* output) {
    double *rows=nullptr,*out=nullptr;Matrix2x2* ib=nullptr;
    cudaError_t e=cudaMalloc(&rows,18*n*sizeof(double));if(e)return e;
    e=cudaMalloc(&out,16*n*sizeof(double));if(e){cudaFree(rows);return e;}
    e=cudaMalloc(&ib,n*sizeof(Matrix2x2));if(e){cudaFree(rows);cudaFree(out);return e;}
    e=cudaMemcpy(rows,input,18*n*sizeof(double),cudaMemcpyHostToDevice);
    if(!e){rest_matrices<<<(n+127)/128,128>>>(n,rows,ib);energies<<<(n+127)/128,128>>>(n,rows,ib,out);
        derivatives<<<(n+127)/128,128>>>(n,rows,ib,out);caps<<<(n+127)/128,128>>>(n,rows,ib,out);e=cudaGetLastError();}
    if(!e)e=cudaMemcpy(output,out,16*n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(rows);cudaFree(out);cudaFree(ib);return e;
}

__global__ void path_input(int n,const double* input,double* rows) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=18*n)return;
    rows[i]=input[(i/18)*27+i%18];
}
__global__ void path_step(int n,const double* input,const Matrix2x2* ib,double* rows,double* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    Vector9 x,d;for(int j=0;j<9;++j){x(j)=input[27*i+9+j];d(j)=input[27*i+18+j];}
    Matrix2x2 frame=psq_native::inverse_rest_frame(ib[i]);
    auto f=psq_native::deformation(x,frame),df=psq_native::deformation(d,frame);
    Matrix3x3 u;Matrix2x2 v;Vector2 s;psq_native::svd32(f,u,v,s);
    double norm=df.norm(),gap=fmin(1.1-s(0),s(1)-1e-12);
    double legacy=gap>0?(norm>0?fmin(1.,.9*gap/norm):1.):0.;
#ifdef CHECK_COORDINATE_GUARD
    double alpha=psq_native::coordinate_step(x,d,frame,1.1,f,df,s,legacy);
#else
    double alpha=psq_step_cap::directional(f.data(),df.data(),1.1,s(0),s(1),norm,legacy);
#endif
    out[8*i]=alpha;out[8*i+1]=psq_native::strain(x,ib[i],{1.1,1.05,1000.});
    out[8*i+3]=s(0);out[8*i+5]=psq_step_cap::inside_stretch(f.data(),df.data(),alpha,1.1);
    out[8*i+6]=s(1);
    double moved=0;
    for(int j=0;j<9;++j){double y=x(j)+alpha*d(j);rows[18*i+9+j]=y;moved=fmax(moved,fabs(y-x(j)));}
    out[8*i+7]=moved;
}
__global__ void path_energy(int n,const double* rows,const Matrix2x2* ib,double* out) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    Vector9 x;for(int j=0;j<9;++j)x(j)=rows[18*i+9+j];
    out[8*i+2]=psq_native::strain(x,ib[i],{1.1,1.05,1000.});
    auto f=psq_native::deformation(x,psq_native::inverse_rest_frame(ib[i]));
    Matrix3x3 u;Matrix2x2 v;Vector2 s;psq_native::svd32(f,u,v,s);out[8*i+4]=s(0);
}
extern "C" __declspec(dllexport) int evaluate_steps(int n,const double* input,double* output) {
    double *in=nullptr,*rows=nullptr,*out=nullptr;Matrix2x2* ib=nullptr;
    cudaError_t e=cudaMalloc(&in,27*n*sizeof(double));if(e)return e;
    e=cudaMalloc(&rows,18*n*sizeof(double));if(e){cudaFree(in);return e;}
    e=cudaMalloc(&out,8*n*sizeof(double));if(e){cudaFree(in);cudaFree(rows);return e;}
    e=cudaMalloc(&ib,n*sizeof(Matrix2x2));if(e){cudaFree(in);cudaFree(rows);cudaFree(out);return e;}
    e=cudaMemcpy(in,input,27*n*sizeof(double),cudaMemcpyHostToDevice);
    if(!e){path_input<<<(18*n+127)/128,128>>>(n,in,rows);rest_matrices<<<(n+127)/128,128>>>(n,rows,ib);
        path_step<<<(n+127)/128,128>>>(n,in,ib,rows,out);path_energy<<<(n+127)/128,128>>>(n,rows,ib,out);e=cudaGetLastError();}
    if(!e)e=cudaMemcpy(output,out,8*n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(in);cudaFree(rows);cudaFree(out);cudaFree(ib);return e;
}
