//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 61
using namespace std;
long long n,m,sum,res,flag;
long long b[N],d[N],len[N];
void init()
{
    b[1]=len[1]=1;d[1]=0;
    for(int i=2;i<N;i++)
        b[i]=b[i-1]+d[i-1]+1,d[i]=d[i-1]+b[i-1],len[i]=len[i-1]*2+1;
}
long long solve(long long n)
{
    if(n<=1)return n;
    long long t = lower_bound(len,len+N,n)-len;
    if(len[t]==n)return b[t];
    return b[t-1]+1+solve(n-len[t-1]-1)-(n>=(len[t-1]+1)/2+1+len[t-1]);
}
int main()
{
    init();
    long long i,j,k,cas,T,t,x,y,z;
    scanf("%I64d",&T);
    cas=0;
    while(T--)
    {
        scanf("%I64d%I64d",&x,&y);
        printf("%I64d\n",solve(y)-solve(x-1));
    }
    return 0;
}
