//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 1123456
using namespace std;
long long n,m,sum,res,flag;
char s[N];
long long fac[N];
void init()
{
    fac[0]=1;
    for(long long i=1;i<=1000;i++)
        fac[i]=(fac[i-1]*i)%MOD;
}
long long power(long long x,long long k,long long mod)
{
    long long ans = 1;
    while(k)
    {
        if(k & 1) ans=ans*x%mod;
        x=x*x%mod;
        k >>= 1;
    }
    return ans;
}
int main()
{
    long long i,j,k,kk,cas,T,t,x,y,z;
    init();
    scanf("%I64d",&T);
    cas=0;
    while(T--)
    {
        scanf("%s",s);
        n=strlen(s);
        long long a[30]={0};
        for(i=0;i<n;i++)a[s[i]-'a']++;
        m=0;
        for(i=0;i<26;i++)
            if(a[i]&1)m++;
        if(m>1)printf("0\n");
        else
        {
            sum=1;
            for(i=0;i<26;i++)
                sum=(sum*fac[a[i]/2])%MOD;
            res=fac[n/2]*power(sum,MOD-2,MOD)%MOD;
            printf("%I64d\n",res);
        }
    }
    return 0;
}
