//kopyh
#include <bits/stdc++.h>
#define PI acos(-1.0)
#define INF 0x3f3f3f3f3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 11234
using namespace std;
long long n,m,sum,res,flag;
long long a[N],b[N];
//char s[N];
int main()
{
    long long i,j,k,kk,cas,T,t,x,y,z;
    #ifndef ONLINE_JUDGE
        freopen("test.txt","r",stdin);
    #endif
	memset(a,0,sizeof(a));
	a[2]=1;
	for(i=3;i<=2000+1;i++)
		a[i]=(a[i-1]+a[i-2]+a[i-3])%MOD*25%MOD;
    scanf("%I64d",&T);
    cas=0;
    while(T--)
    {
        scanf("%I64d",&n);
        sum=(a[n+1]+a[n]+a[n-1])%MOD*26%MOD;
        printf("%I64d\n",sum);
    }
    return 0;
}