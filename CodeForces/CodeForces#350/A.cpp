//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
int n,m,sum,res,flag;
int main()
{
    int i,j,k,T,cas,x,y,t;
    while(scanf("%d",&n)!=EOF)
    {
        m=2*(n/7);
        t=n%7;
        y=m+(t==6);
        x=m+min(t,2);
        printf("%d %d\n",y,x);
    }
    return 0;
}
