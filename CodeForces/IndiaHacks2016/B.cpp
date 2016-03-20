//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 1123456
using namespace std;
long long  n,m,sum,res,flag;
char s[10],ss[10];
long long a[10][10],b[10][10];
int main()
{
    long long i,j,k,kk,cas,T,t,x,y,z;
    while(scanf("%I64d%I64d",&n,&m)!=EOF)
    {
        memset(a,0,sizeof(a));
        memset(b,0,sizeof(b));
        for(i=0;i<m;i++)
        {
            scanf("%s%s",s,ss);
            x=s[0]-'a';
            y=ss[0]-'a';
            a[y][x]++;
        }
        b[1][0]=1;
        for(i=2;i<=n;i++)
            for(j=0;j<6;j++)
                for(k=0;k<6;k++)
                    b[i][j]+=a[k][j]*b[i-1][k];
        for(res=0,i=0;i<6;i++)
            res+=b[n][i];
        printf("%I64d\n",res);
    }
    return 0;
}
