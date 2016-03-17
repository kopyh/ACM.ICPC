//kopyh
#include <bits/stdc++.h>
#define PI acos(-1.0)
#define INF 0x3f3f3f3f3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 112
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
    scanf("%I64d",&T);
    cas=0;
    while(T--)
    {
        scanf("%I64d",&n);
        for(i=0;i<n;i++)
            scanf("%I64d",&a[i]);
        memset(b,0,sizeof(b));
        flag=1;
        if(n<4)flag=0;
        else
        {
            if(a[0]<1||a[0]>9)flag=0;
            if(flag)b[a[0]]++;
            for(i=1;i<n&&flag;i++)
            {
                x=a[i-1],y=a[i];
                if(b[y])flag=0;
                if(a[i]<1||a[i]>9)flag=0;
                b[y]++;
                if(x>y)x^=y,y^=x,x^=y;
                if(x==1&&y==3&&!b[2])flag=0;
                if(x==4&&y==6&&!b[5])flag=0;
                if(x==7&&y==9&&!b[8])flag=0;
                if(x==1&&y==7&&!b[4])flag=0;
                if(x==2&&y==8&&!b[5])flag=0;
                if(x==3&&y==9&&!b[6])flag=0;
                if(x==1&&y==9&&!b[5])flag=0;
                if(x==3&&y==7&&!b[5])flag=0;
            }
        }
        printf("%s\n",flag?"valid":"invalid");
    }
    return 0;
}