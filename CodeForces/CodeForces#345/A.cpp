//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 1123456
using namespace std;
long long n,m,sum,res,flag;
int main()
{
    long long i,j,k,kk,cas,T,t,x,y,z;
    while(scanf("%I64d%I64d",&n,&m)!=EOF)
    {
        if(n<m)n^=m,m^=n,n^=m;
        sum=0;
        while(n>1&&m)
        {
            n-=2;m+=1;
            if(n<m){n^=m,m^=n,n^=m;}
            sum++;
        }
        printf("%I64d\n",sum);
    }
    return 0;
}
