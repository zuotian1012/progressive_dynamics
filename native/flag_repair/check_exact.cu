#include "psq_exact_offset.cuh"
extern "C" __declspec(dllexport) int evaluate(int nc,int nf,int nt,
 const int* faces,const int* anchors,const double* inv,const double* gamma,
 const double* tangent,const double* x,const double* fg,double* cg) {
 int *df=nullptr,*da=nullptr;double *di=nullptr,*dg=nullptr,*dt=nullptr,*dx=nullptr,*dfg=nullptr,*dcg=nullptr;
 cudaError_t err=cudaSuccess;
 #define UP(dst,src,n) if(!err)err=cudaMalloc(&dst,(n)*sizeof(*dst));if(!err)err=cudaMemcpy(dst,src,(n)*sizeof(*dst),cudaMemcpyHostToDevice)
 UP(df,faces,3*nt);UP(da,anchors,nf);UP(di,inv,4*nt);UP(dg,gamma,nf);UP(dt,tangent,2*nf);UP(dx,x,3*nc);UP(dfg,fg,3*nf);
 if(!err)err=cudaMalloc(&dcg,3*nc*sizeof(double));if(!err)err=cudaMemset(dcg,0,3*nc*sizeof(double));
 if(!err){psq::restrict_offset_gradient<<<(nf+127)/128,128>>>(nf,df,da,di,dg,dt,dx,dfg,dcg);err=cudaGetLastError();}
 if(!err)err=cudaMemcpy(cg,dcg,3*nc*sizeof(double),cudaMemcpyDeviceToHost);
 cudaFree(df);cudaFree(da);cudaFree(di);cudaFree(dg);cudaFree(dt);cudaFree(dx);cudaFree(dfg);cudaFree(dcg);
 return err;
}
