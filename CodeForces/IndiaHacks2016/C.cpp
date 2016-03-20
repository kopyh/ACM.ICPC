//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 1123456
using namespace std;
long long  n,m,sum,res,flag;
long long a[N],b[N];
long long dir[9][2]={0,0,0,1,0,-1, 1,1,1,0,1,-1, -1,0,-1,-1,-1,1};
bool check(int i)
{
    if(i%2==0&&a[i]>a[i-1]&&a[i]>a[i+1])return true;
    else if(i%2==1&&a[i]<a[i-1]&&a[i]<a[i+1])return true;
    return false;
}
void exchange(int i,int j)
{
    int t=a[i];a[i]=a[j];a[j]=t;
}
int main()
{
    long long i,j,k,kk,cas,T,t,x,y,z,xx,yy;
    while(scanf("%I64d",&n)!=EOF)
    {
        for(i=1;i<=n;i++)
            scanf("%I64d",&a[i]);
        a[0]=INF;a[n+1]=n%2?INF:0;
        sum=0;x=y=0;
        for(i=1;i<=n;i++)
        {
            if(i%2&&a[i]>=a[i-1]){if(x)y=i;else x=i;i++;sum++;}
            else if(i%2==0&&a[i]<=a[i-1]){if(x)y=i;else x=i;i++;sum++;}
        }
        res=0;
        if(sum==2)
            for(i=0;i<9;i++)
            {
                xx=x+dir[i][0];yy=y+dir[i][1];
                if(xx<1||yy>n)continue;
                exchange(xx,yy);
                if(xx<yy&&check(x)&&check(y)&&check(xx)&&check(yy))res++;
                exchange(xx,yy);
            }
        else if(sum==1)
            for(i=1;i<=n;i++)
                for(j=-1;j<=1;j++)
                {
                    exchange(x+j,i);
                    if(x!=i&&check(x)&&check(i)&&check(x+j))res++;
                    exchange(x+j,i);
                }
        printf("%I64d\n",res);
    }
    return 0;
}
