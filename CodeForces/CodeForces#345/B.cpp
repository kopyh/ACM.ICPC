//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 11234
using namespace std;
long long n,m,sum,res,flag;
long long a[N],b[N];
int main()
{
    long long i,j,k,kk,cas,T,t,x,y,z;
    while(scanf("%I64d",&n)!=EOF)
    {
        for(i=0;i<n;i++)scanf("%I64d",&a[i]);
        sort(a,a+n);
        res=0;sum=1;t=a[0];a[0]=0;
        while(sum<n)
        {
            for(i=0;i<n;i++)
                if(a[i]>t)
                    sum++,t=a[i],a[i]=0,res++;
            t=0;
            for(i=0;!t&&i<n;i++)
                if(a[i])
                {
                    t=a[i];a[i]=0;
                    sum++;
                    break;
                }
        }
        printf("%I64d\n",res);
    }
    return 0;
}
