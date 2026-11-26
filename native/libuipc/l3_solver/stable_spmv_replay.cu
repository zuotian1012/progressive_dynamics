#include "stable_spmv.cuh"
extern "C" __declspec(dllexport) int stable_spmv_replay(int n,int k,const int* rows,const int* cols,
 const double* blocks,const double* x,double* result,double* dot_result,int repeats,int graph) {
    int *r=nullptr,*c=nullptr,*count=nullptr;double *a=nullptr,*v=nullptr,*y=nullptr,*dot=nullptr;
    cudaMalloc((void**)&r,k*sizeof(int));cudaMalloc((void**)&c,k*sizeof(int));cudaMalloc((void**)&count,sizeof(int));
    cudaMalloc((void**)&a,9*k*sizeof(double));cudaMalloc((void**)&v,n*sizeof(double));cudaMalloc((void**)&y,n*sizeof(double));cudaMalloc((void**)&dot,sizeof(double));
    cudaMemcpy(r,rows,k*sizeof(int),cudaMemcpyHostToDevice);cudaMemcpy(c,cols,k*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(count,&k,sizeof(int),cudaMemcpyHostToDevice);cudaMemcpy(a,blocks,9*k*sizeof(double),cudaMemcpyHostToDevice);cudaMemcpy(v,x,n*sizeof(double),cudaMemcpyHostToDevice);
    cudaStream_t stream;cudaStreamCreate(&stream);int status=0;
    if(graph) {
        cudaGraph_t g;cudaGraphExec_t e;
        status=cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal);
        if(!status)status=uipc_stable_linear::launch(r,c,a,count,k,v,y,n,dot,stream);
        if(!status)status=cudaStreamEndCapture(stream,&g);
        if(!status)status=cudaGraphInstantiate(&e,g,0);
        if(!status){for(int i=0;i<repeats;++i)cudaGraphLaunch(e,stream);status=cudaStreamSynchronize(stream);cudaGraphExecDestroy(e);cudaGraphDestroy(g);}
    } else {
        for(int i=0;i<repeats&&!status;++i)status=uipc_stable_linear::launch(r,c,a,count,k,v,y,n,dot,stream);
        if(!status)status=cudaStreamSynchronize(stream);
    }
    if(!status) {cudaMemcpy(result,y,n*sizeof(double),cudaMemcpyDeviceToHost);cudaMemcpy(dot_result,dot,sizeof(double),cudaMemcpyDeviceToHost);}
    cudaStreamDestroy(stream);cudaFree(r);cudaFree(c);cudaFree(count);cudaFree(a);cudaFree(v);cudaFree(y);cudaFree(dot);return status;
}
