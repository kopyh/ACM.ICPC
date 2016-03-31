//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 112345
using namespace std;
int n,m,sum,flag,res;
int a[N],b[N];
int main()
{
    int i,j,k,cas,T,t,x,y,z;
    while(scanf("%d%d",&n,&m)!=EOF)
    {
        memset(a,0,sizeof(a));
        memset(b,0,sizeof(b));
        vector<int>g[N];
        for(i=0;i<m;i++)
        {
            scanf("%d%d",&x,&y);
            a[x]++;a[y]++;
            g[x].push_back(y);
            g[y].push_back(x);
        }
        queue<int>q;
        for(i=1;i<=n;i++)
            if(a[i]==1)q.push(i);
        while(!q.empty())
        {
            x=q.front();q.pop();
            b[x]=1;
            if(a[x]==0)continue;
            for(i=0;i<g[x].size();i++)
                if(b[g[x][i]]==0)
                {
                    y=g[x][i];
                    break;
                }
            a[y]--;
            if(a[y]==1 && b[y]==0)q.push(y);

        }
        sum=0;
        for(i=1;i<=n;i++)if(a[i])sum++;
        printf("%d\n",n-sum);
    }
    return 0;
}
