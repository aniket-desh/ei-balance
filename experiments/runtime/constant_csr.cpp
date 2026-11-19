// Exact constant-block sparse complement action. Build is optional; SciPy fallback.
#include <cstdint>
extern "C" void ei_action(int n,int ne,const int32_t* ptr,const int32_t* ix,const int32_t* split,const double* weights,const double* rank,const double* x,double* y) {
 double pop[2]={0.,0.};
 for(int j=0;j<ne;j++)pop[0]+=x[j];
 for(int j=ne;j<n;j++)pop[1]+=x[j];
 for(int i=0;i<n;i++){
   int a=(i>=ne);double s0=0.,s1=0.;
   for(int k=ptr[i];k<split[i];k++)s0+=x[ix[k]];
   for(int k=split[i];k<ptr[i+1];k++)s1+=x[ix[k]];
   y[i]=weights[2*a]*s0+weights[2*a+1]*s1+rank[2*a]*pop[0]+rank[2*a+1]*pop[1];
 }
}
