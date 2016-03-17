//kopyh
#include <bits/stdc++.h>
using namespace std;
long long n,m,sum,res,flag;
int main()
{
    long long i,j,k,kk,cas,T,t,x,y,z;
    scanf("%I64d",&T);
    cas=0;
    while(T--)
    {
        scanf("%I64d%I64d",&n,&m);
        sum=1;
        while(n&&m)
        {
            if(n==m)break;
            if(n<m)n^=m,m^=n,n^=m;
            n-=m;sum++;
        }
        printf("%I64d\n",sum);
    }
    return 0;
}