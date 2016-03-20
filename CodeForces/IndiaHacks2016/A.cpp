//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 11234
using namespace std;
long long  n,m,sum,res,flag;
long long a[N],b[N];
int main()
{
    long long i,j,k,kk,cas,T,t,x,y,z;
    while(scanf("%I64d",&n)!=EOF)
    {
        memset(b,0,sizeof(b));
        for(i=0;i<n;i++)
            scanf("%I64d",&a[i]);
        sort(a,a+n);
        res=0;
        for(i=0;i<n;i++)b[a[i]]++;
        for(i=2;i<=1000;i++)
            if(b[i]&&b[i-1]&&b[i-2])
                res++;
        printf("%s\n",res?"YES":"NO");
    }
    return 0;
}
