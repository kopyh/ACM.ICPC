//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 11234
using namespace std;
int n,m,sum,res,flag;
int a[N],b[N],c[N];
int main()
{
    int i,j,k,cas,T,t,x,y,z;
    while(scanf("%d",&n)!=EOF)
    {
        for(i=0;i<n;i++)scanf("%d",&a[i]);
        memset(b,0,sizeof(b));
        for(i=0;i<n;i++)
        {
            x=y=0;
            memset(c,0,sizeof(c));
            for(j=i;j<n;j++)
            {
                c[a[j]]++;
                if(c[a[j]]>x||c[a[j]]==x&&a[j]<y)
                    x=c[a[j]];y=a[j];
                b[y]++;
            }
        }
        for(i=1;i<=n;i++)
            printf("%d ",b[i]);
        printf("\n");
    }
    return 0;
}
