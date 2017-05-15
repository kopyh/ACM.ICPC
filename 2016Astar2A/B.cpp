//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f3f3f3f3f
#define MOD 1000000007
#define N 16
using namespace std;
long long n,m,sum,res,flag;
long long a[N],b[N],dp[1<<16][16];
int main()
{
    long long i,j,k,cas,T,t,x,y,z;
    scanf("%d",&T);
    cas=0;
    while(T--)
    {
        scanf("%I64d",&n);
        for(i=0;i<n;i++)scanf("%I64d%I64d",&a[i],&b[i]);
        for(i=0;i<(1<<n);i++)for(j=0;j<n;j++)dp[i][j]=-INF;
        for(unsigned int i=0;i<(1<<n);i++)
            for(j=0;j<n;j++)
                if(dp[i][j]!=-INF||__builtin_popcount(i)==0)
                    for(k=0;k<n;k++)
                        if(!((1<<k)&i) && (b[k]==-1||b[k]==__builtin_popcount(i)))
                            dp[(1<<k)|i][k] = max(dp[(1<<k)|i][k],dp[i][j]!=-INF?dp[i][j]+a[j]*a[k]:0);
        res=-INF;
        for(i=0;i<n;i++)res=max(res,dp[(1<<n)-1][i]);
        printf("Case #%I64d:\n%I64d\n",++cas,res);
    }
    return 0;
}
