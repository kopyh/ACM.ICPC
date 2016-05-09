//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 112345
using namespace std;
int n,m,sum,res,flag;
int a[N];
int main()
{
    int i,j,k,T,cas,x,y,t;
    while(scanf("%d",&n)!=EOF)
    {
        a[0]=0;res=90;
        for(i=1;i<=n;i++)
        {
            scanf("%d",&a[i]);
            if(a[i]-a[i-1]>15)res=min(res,a[i-1]+15);
        }
        a[i]=90;
        if(a[i]-a[i-1]>15)res=min(res,a[i-1]+15);
        printf("%d\n",res);
    }
    return 0;
}
