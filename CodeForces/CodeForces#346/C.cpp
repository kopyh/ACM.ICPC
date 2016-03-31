//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
long long  n,m,sum,flag,res;
long long  a[N],b[N];
long long fen(long long x,long long y)
{
    if(x==y)return x;
    long long mm=(x+y)/2;
    if(y-x==1)mm=y;
    long long t=(mm+1)*mm/2-b[upper_bound(a,a+n+2,mm)-a-1];
    if(t<=m)return fen(mm,y);
    else return fen(x,mm-1);
}
int main()
{
    long long  i,j,k,cas,T,t,x,y,z;
    while(scanf("%I64d%I64d",&n,&m)!=EOF)
    {
        for(i=1;i<=n;i++)
            scanf("%I64d",&a[i]);
        sort(a+1,a+1+n);
        a[n+1]=INF;
        a[0]=b[0]=0;
        for(i=1;i<=n;i++)
            b[i]=b[i-1]+a[i];
        res=fen(1,m);
        while(res>0 && a[lower_bound(a,a+n+2,res)-a]==res)res--;
        long long t=(res+1)*res/2-b[upper_bound(a,a+n+2,res)-a-1];
        m-=t;
        y=res+m;
        while(y>res && a[lower_bound(a,a+n+2,y)-a]==y)y--;
        for(i=j=1,sum=1;i<res;i++)
        {
            if(a[j]==i)j++;
            else sum++;
        }
        printf("%I64d\n",y?sum:0);
        for(i=j=1;i<res;i++)
        {
            if(a[j]==i)j++;
            else printf("%I64d ",i);
        }
        if(y)printf("%I64d\n",y);
    }
    return 0;
}
