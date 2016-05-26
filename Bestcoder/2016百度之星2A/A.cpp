//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 11234
using namespace std;
long long  n,m,sum,res,flag;
long long a[N];
int main()
{
    long long  i,j,k,cas,T,t,x,y,z,c;
    scanf("%I64d",&T);
    cas=0;
    while(T--)
    {
        scanf("%I64d%I64d%I64d%I64d",&x,&m,&k,&c);
        sum=0;
        t=x; y=1;
        while(t<k)t*=10,t+=x,y++;
        if(y>=m)
        {
            y=0;
            while(m--)y*=10,y+=x;
            printf("Case #%I64d:\n",++cas);
            printf(y%k==c?"Yes\n":"No\n");
            continue;
        }
        m-=y;m++;
        t%=k;
        memset(a,0,sizeof(a));
        while(!a[t])
        {
            a[t]=++sum;
            t*=10;t+=x;
            t%=k;
        }
        m%=sum;
        if(!m)m=sum;
        t=-1;
        for(i=0;i<N;i++)if(a[i]==m)t=i;
        printf("Case #%I64d:\n",++cas);
        printf(t==c?"Yes\n":"No\n");
    }
    return 0;
}
