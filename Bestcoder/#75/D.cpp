//kopyh
#include <bits/stdc++.h>
#define PI acos(-1.0)
#define INF 0x3f3f3f3f3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 11234
using namespace std;
long long n,m,sum,res,flag;
//long long a[N],b[N];
//char s[N];
int main()
{
    long long i,j,k,kk,cas,T,t,x,y,z;
    #ifndef ONLINE_JUDGE
        freopen("test.txt","r",stdin);
    #endif
    scanf("%I64d",&T);
    cas=0;
    while(T--)
    {
        scanf("%I64d",&n);
        m=n-1;
        for(i=2,res=0;i<=n;i++,m--)
            res=(res+m)%i;
        printf("%I64d\n",res+1);
    }
    return 0;
}