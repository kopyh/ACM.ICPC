//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 1123456
using namespace std;
int n,m,sum,res,flag;
int a[N],b[N];
char s[N];
int f[N],g[510][510];
int dir[4][2]={1,0,0,1,-1,0,0,-1};
void init(int n)
{
    for(int i=1;i<=n;i++)f[i]=i;
}
int getf(int v)
{
    while(f[v] != v)
    {
        f[v]=f[f[v]];
        v = f[v];
    }
    return f[v];
}
void unions(int x,int y)
{
    x = getf(x);
    y = getf(y);
    if(x == y) return;
    f[y] = x;
}
int main()
{
    int i,j,k,kk,cas,T,t,x,y,z;
    scanf("%d",&T);
    cas=0;
    while(T--)
    {
        scanf("%d%d",&n,&m);
        for(i=0;i<n;i++)
        {
            scanf("%s",s);
            for(j=0;j<m;j++)
                g[i][j]=s[j]-'0';
        }
        scanf("%d",&sum);
        for(i=1;i<=sum;i++)
        {
            scanf("%d%d",&a[i],&b[i]);
            g[a[i]][b[i]]=1;
        }
        init(n*m+2);
        for(i=0;i<n;i++)
            for(j=0;j<m;j++)
                if(!g[i][j])
                    for(k=0;k<4;k++)
                    {
                        x=i+dir[k][0];y=j+dir[k][1];
                        if(x<0||y<0||x>=n||y>=m)continue;
                        if(!g[x][y])
                            unions(i*m+j,x*m+y);
                        if(i==0)
                            unions(i*m+j,n*m);
                        if(i==n-1)
                            unions(i*m+j,n*m+1);
                    }
        res=sum;
        while(getf(n*m)!=getf(n*m+1))
        {
            i=a[res];j=b[res];
            g[i][j]=0;
            for(k=0;k<4;k++)
            {
                x=i+dir[k][0];y=j+dir[k][1];
                if(x<0||y<0||x>=n||y>=m)continue;
                if(!g[x][y])
                    unions(i*m+j,x*m+y);
                if(i==0)
                    unions(i*m+j,n*m);
                if(i==n-1)
                    unions(i*m+j,n*m+1);
            }
            res--;
        }
        printf("%d\n",res==sum?-1:res+1);
    }
    return 0;
}
