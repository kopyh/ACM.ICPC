//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
long long  n,m,sum,res,flag;
long long a[N];
int main()
{
    long long i,j,k,T,cas,x,y,t;
    while(scanf("%I64d%I64d",&n,&m)!=EOF)
    {
        for(i=1;i<=n;i++)
            scanf("%I64d",&a[i]);
        sum=0;
        for(i=1;i<=n;i++)
        {
            sum+=i;
            if(sum>=m)break;
        }
        sum-=i;
        m-=sum;
        printf("%I64d\n",a[m]);
    }
    return 0;
}
