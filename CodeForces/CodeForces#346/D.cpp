//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
int n,m,sum,flag,res;
int a[N],b[N];
int left(int i)
{
    if(b[i]>b[i-1] && a[i]>a[i+1])return 1;
    else if(b[i]<b[i-1] && a[i]<a[i+1])return 1;
    else if(a[i]>a[i-1] && b[i+1]>b[i])return 1;
    else if(a[i]<a[i-1] && b[i]>b[i+1])return 1;
    else return 0;
}
int main()
{
    int i,j,k,cas,T,t,x,y,z;
    while(scanf("%d",&n)!=EOF)
    {
        for(i=0;i<=n;i++)
            scanf("%d%d",&a[i],&b[i]);
        x=y=0;
        for(i=1;i<n;i++)
        {
            if(left(i))x++;
            else y++;
        }
        printf("%d\n",min(x,y));
    }
    return 0;
}
