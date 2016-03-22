//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 1123456
using namespace std;
long long  n,m,sum,res,flag;
long long  a[N],b[N],c[N];
char s[N];
int main()
{
    long long  i,j,k,kk,cas,T,t,x,y,z,r,l;
    while(scanf("%I64d%I64d%I64d%I64d",&n,&x,&y,&T)!=EOF)
    {
        scanf("%s",s);
        memset(a,0,sizeof(a));
        for(i=0;i<n;i++)
            a[i]=a[i+n]=(s[i]=='w')?1+y:1;
        for(i=n,t=0;i<2*n;i++)
        {
            if(t+a[i]+x*(i-n)>T)break;
            t+=a[i];
        }
        r=i-1;l=n;z=t;
        res=r-l+1;
        if(res==n)printf("%I64d\n",res);
        else
        {
            while(res<n&&r>=n&&l>0)
            {
                t=x*min((r-n)*2+n-l,(n-l)*2+r-n);
                while(t+z<=T&&res<n&&r>=n&&l>0)
                {
                    res=max(res,r-l+1);
                    l--;
                    z+=a[l];
                    t=x*min((r-n)*2+n-l,(n-l)*2+r-n);
                }
                z-=a[r];
                r--;
            }
            printf("%I64d\n",res);
        }
    }
    return 0;
}
