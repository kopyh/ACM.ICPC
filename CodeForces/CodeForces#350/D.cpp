//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
long long n,m,sum,res,flag;
long long a[N],b[N];
long long fen(long long x,long long y)
{
    if(x==y)return x;
    if(y-x==1)
    {
        sum=0;
        for(int i=0;i<n&&sum<=m;i++)
            sum+=max(0LL,a[i]*y-b[i]);
        if(sum<=m)return y;
        else return x;
    }
    int mm=(x+y)>>1;
    sum=0;
    for(int i=0;i<n&&sum<=m;i++)
        sum+=max(0LL,a[i]*mm-b[i]);
    if(sum<=m)return fen(mm,y);
    else return fen(x,mm-1);
}
int main()
{
    long long i,j,k,T,cas,x,y,t;
    while(scanf("%d%d",&n,&m)!=EOF)
    {
        for(i=0;i<n;i++)scanf("%d",&a[i]);
        for(i=0;i<n;i++)scanf("%d",&b[i]);
        printf("%d\n",fen(0,2000000000LL));
    }
    return 0;
}
