//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 1123456
using namespace std;
int n,m,sum,res,flag;
int main()
{
    int i,j,k,kk,cas,T,t,x,y,z;
    scanf("%d",&T);
    while(T--)
    {
        scanf("%d",&n);
        for(i=0;i<n;i++)scanf("%d",&x);
        printf("%d\n",n==1?x:0);
    }
    return 0;
}
