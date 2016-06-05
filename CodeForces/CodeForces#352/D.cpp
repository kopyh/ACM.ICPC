//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
long long n,m,sum,res,flag;
long long a[N];
int main()
{
    long long i,j,k,cas,T,t,x,y,z,l,r,b,c;
    while(scanf("%I64d%I64d",&n,&m)!=EOF)
    {
        sum=0;
        for(i=1;i<=n;i++)
        {
            scanf("%I64d",&a[i]);
            sum+=a[i];
        }
        sort(a+1,a+n+1);
        z=sum/n;
        y=sum-z*n;
        x=n-y;
        t=0;
        for(i=1;i<=x;i++)t+=abs(a[i]-z);
        for(;i<=n;i++)t+=abs(z+1-a[i]);
        t/=2;
        if(t<=m)printf(y?"1\n":"0\n");
        else
        {
            x=a[1];y=a[n];
            l=1;r=n;
            b=c=0;
            while(b<m||c<m)
            {
                while(b<m&&x==a[l+1])l++;
                while(c<m&&y==a[r-1])r--;
                if(b<=c)b+=(a[l+1]-x)*l,x=a[l+1];
                else if(b>c)c+=(y-a[r-1])*(n-r+1),y=a[r-1];
            }
            if(b>m)x-=(b-m+l-1)/l;
            if(c>m)y+=(c-m+n-r)/(n-r+1);
            printf("%I64d\n",y-x);
        }
    }
    return 0;
}





