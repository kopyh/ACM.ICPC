//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
int n,m,sum,res,flag;
int a[N],b[N],c[N];
int main()
{
    int i,j,k,T,cas,x,y,t;
    while(scanf("%d",&n)!=EOF)
    {
        for(i=0;i<n;i++)
            scanf("%d",&a[i]);
        scanf("%d",&m);
        for(i=1;i<=m;i++)scanf("%d",&b[i]);
        for(i=1;i<=m;i++)scanf("%d",&c[i]);
        sort(a,a+n);
        res=1;x=y=0;
        for(i=1;i<=m;i++)
        {
            t=upper_bound(a,a+n,b[i])-lower_bound(a,a+n,b[i]);
            if(t>x)
            {
                x=t;res=i;
                y=upper_bound(a,a+n,c[i])-lower_bound(a,a+n,c[i]);
            }
            else if(t==x)
            {
                k=upper_bound(a,a+n,c[i])-lower_bound(a,a+n,c[i]);
                if(k>y)
                {
                    x=t;y=k;res=i;
                }
            }
        }
        printf("%d\n",res);
    }
    return 0;
}
