//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123
using namespace std;
int n,m,sum,flag,res;
int a[N];
int main()
{
    int i,j,k,cas,T,t,x,y,z;
    while(scanf("%d",&n)!=EOF)
    {
        memset(a,0,sizeof(a));
        for(i=0;i<n;i++)
        {
            scanf("%d",&t);
            a[t]++;
        }
        t=res=0;
        for(i=1;i<=1000;i++)
            if(t<a[i])
                res+=a[i]-t, t=a[i];
        printf("%d\n",n-res);
    }
    return 0;
}
