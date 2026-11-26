#pragma once
#include <cuda_runtime.h>
#include <cmath>
namespace psq {
// Exact VJP of n(F)*gamma + polar(F)*t. The linear U^T term is added separately.
// Six directional differentials of the closed-form 3x2 polar map; no finite differences.
__global__ void restrict_offset_gradient(int nf,const int* faces,const int* anchors,
    const double* inv_dm,const double* gamma,const double* tangent,
    const double* x,const double* fg,double* cg) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=nf)return;
    if(gamma[i]==0 && tangent[2*i]==0 && tangent[2*i+1]==0)return;
    int f=anchors[i],a=faces[3*f],b=faces[3*f+1],c=faces[3*f+2];
    const double* M=inv_dm+4*f;
    double F[6],gF[6];
    for(int j=0;j<3;++j){double e=x[3*b+j]-x[3*a+j],h=x[3*c+j]-x[3*a+j];F[2*j]=e*M[0]+h*M[2];F[2*j+1]=e*M[1]+h*M[3];}
    double A=0,B=0,C=0;for(int j=0;j<3;++j){A+=F[2*j]*F[2*j];B+=F[2*j]*F[2*j+1];C+=F[2*j+1]*F[2*j+1];}
    double n[3]={F[2]*F[5]-F[4]*F[3],F[4]*F[1]-F[0]*F[5],F[0]*F[3]-F[2]*F[1]};
    double s=sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]),q2=A+C+2*s;
    double k=1/(s*sqrt(q2)),i00=(C+s)*k,i01=-B*k,i11=(A+s)*k;
    double t0=tangent[2*i],t1=tangent[2*i+1];
    for(int d=0;d<6;++d){
        double D[6]={0,0,0,0,0,0};D[d]=1;
        double dn[3]={D[2]*F[5]+F[2]*D[5]-D[4]*F[3]-F[4]*D[3],
            D[4]*F[1]+F[4]*D[1]-D[0]*F[5]-F[0]*D[5],
            D[0]*F[3]+F[0]*D[3]-D[2]*F[1]-F[2]*D[1]};
        double ds=(n[0]*dn[0]+n[1]*dn[1]+n[2]*dn[2])/s;
        double dA=0,dB=0,dC=0;for(int j=0;j<3;++j){dA+=2*F[2*j]*D[2*j];dB+=D[2*j]*F[2*j+1]+F[2*j]*D[2*j+1];dC+=2*F[2*j+1]*D[2*j+1];}
        double dk=-k*(ds/s+(dA+dC+2*ds)/(2*q2));
        double di00=(dC+ds)*k+(C+s)*dk,di01=-dB*k-B*dk,di11=(dA+ds)*k+(A+s)*dk;
        gF[d]=0;
        for(int j=0;j<3;++j){
            double dy=gamma[i]*(dn[j]/s-n[j]*ds/(s*s))
                +D[2*j]*(i00*t0+i01*t1)+D[2*j+1]*(i01*t0+i11*t1)
                +F[2*j]*(di00*t0+di01*t1)+F[2*j+1]*(di01*t0+di11*t1);
            gF[d]+=fg[3*i+j]*dy;
        }
    }
    for(int j=0;j<3;++j){double gb=gF[2*j]*M[0]+gF[2*j+1]*M[1],gc=gF[2*j]*M[2]+gF[2*j+1]*M[3];
        atomicAdd(cg+3*a+j,-gb-gc);atomicAdd(cg+3*b+j,gb);atomicAdd(cg+3*c+j,gc);}
}
}
