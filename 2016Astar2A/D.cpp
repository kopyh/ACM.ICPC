//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123
using namespace std;
int n,m,sum,res,flag;
int a[N],d[N],b[N][N],dp[N][N];
int main()
{
    int i,j,k,cas,T,t,x,y,z,len,l,r;
    scanf("%d",&T);
    cas=0;
    while(T--)
    {
        scanf("%d%d",&n,&m);
        memset(b,0,sizeof(b));
        memset(dp,0,sizeof(dp));
        for(i=1;i<=n;i++)
            scanf("%d",&a[i]);
        for(i=1;i<=m;i++)
        {
            scanf("%d",&d[i]);
            for(j=1;j<=n;j++)
                for(k=j+1;k<=n;k++)
                    if(a[k]-a[j]==d[i])
                        b[j][k] = 1;
        }
        for(len=1;len<=n;len++)
            for(l=1;l<=n-len;l++)
            {
                r=l+len;
                dp[l][r]=max(dp[l][r-1],dp[l+1][r]);
                if(b[l][r]&&dp[l+1][r-1]==(r-l-1))
                    dp[l][r] = max(dp[l][r],dp[l+1][r-1]+2);
                for(i=l+1;i<r;i++)
                {
                    dp[l][r] = max(dp[l][r],dp[l][i]+dp[i+1][r]);
                    if(b[l][i]&&b[i][r]&&a[i]-a[l]==a[r]-a[i]&&dp[l+1][i-1]==(i-l-1)&&dp[i+1][r-1]==(r-i-1))
                        dp[l][r] = max(dp[l][r],r-l+1);
                }
            }
        printf("%d\n",dp[1][n]);
    }
    return 0;
}
