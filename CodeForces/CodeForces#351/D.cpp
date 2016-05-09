//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
int n,m,sum,res,flag;
int main()
{
    int i,j,k,cas,T,t,x,y,z;
    int a,b,c,d;
    while(scanf("%d%d",&n,&m)!=EOF)
    {
        scanf("%d%d%d%d",&a,&b,&c,&d);
        if(m<n+1||n==4)
        {
            printf("-1\n");
            continue;
        }
        printf("%d %d",a,c);
        for(i=1;i<=n;i++)
            if(i!=a&&i!=b&&i!=c&&i!=d)
                printf(" %d",i);
        printf(" %d %d\n",d,b);
        printf("%d %d",c,a);
        for(i=1;i<=n;i++)
            if(i!=c&&i!=d&&i!=a&&i!=b)
                printf(" %d",i);
        printf(" %d %d\n",b,d);
    }
    return 0;
}
