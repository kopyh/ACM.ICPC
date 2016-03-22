//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 1123456
using namespace std;
struct node
{
    long long  x,y;
}a[N];
bool cmpx(node a,node b)
{
	return a.x < b.x;
}
bool cmpy(node a,node b)
{
	return a.y < b.y;
}
bool cmp(node a,node b)
{
    if(a.y == b.y)return a.x > b.x;
	return a.y > b.y;
}
long long  n,m,sum,res,flag;
int main()
{
    long long  i,j,k,kk,cas,T,t,x,y,z;
    while(scanf("%I64d",&n)!=EOF)
    {
        for(i=0;i<n;i++)
            scanf("%I64d%I64d",&a[i].x,&a[i].y);
        sum=0;
        sort(a,a+n,cmpx);
        t=1;x=a[0].x;
        for(i=1;i<n;i++)
        {
            if(a[i].x==x)t++;
            else
            {
                x=a[i].x;
                sum+=((t-1)*t/2);
                t=1;
            }
        }
        sum+=((t-1)*t/2);
        sort(a,a+n,cmpy);
        t=1;y=a[0].y;
        for(i=1;i<n;i++)
        {
            if(a[i].y==y)t++;
            else
            {
                y=a[i].y;
                sum+=((t-1)*t/2);
                t=1;
            }
        }
        sum+=((t-1)*t/2);
        sort(a,a+n,cmp);
        x=a[0].x;y=a[0].y;t=1;
        for(i=1;i<n;i++)
        {
            if(x==a[i].x&&y==a[i].y)t++;
            else
            {
                x=a[i].x;y=a[i].y;
                sum-=((t-1)*t/2);
                t=1;
            }
        }
        sum-=((t-1)*t/2);
        printf("%I64d\n",sum);
    }
    return 0;
}

