//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 11234
using namespace std;
int n,m,sum,res,flag;
int main()
{
    int i,j,k,T,cas,x,y,t;
    while(scanf("%d%d",&n,&m)!=EOF)
    {
        x=1,y=n;
        for(i=0;i<m;i++)
        {
            scanf("%d%d",&j,&k);
            x=max(x,min(j,k));
            y=min(y,max(j,k));
        }
        if(x>=y)printf("0\n");
        else printf("%d\n",y-x);
    }
    return 0;
}
