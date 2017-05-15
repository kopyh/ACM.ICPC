//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
int n,m,sum,res,flag;
vector<int>g[N];
priority_queue<int>q;
int a[N],b[N],tot;
int main()
{
    int i,j,k,cas,T,t,x,y,z;
    scanf("%d",&T);
    cas=0;
    while(T--)
    {
        scanf("%d%d",&n,&m);
        memset(a,0,sizeof(a));
        for(i=1;i<=n;i++)g[i].clear();
        tot=0;
        while(!q.empty())q.pop();
        for(i=0;i<m;i++)
        {
            scanf("%d%d",&x,&y);
            g[x].push_back(y);
            a[y]++;
        }
        for(i=1;i<=n;i++)if(!a[i])q.push(i);
        while(!q.empty())
        {
            t=q.top();q.pop();
            b[tot++]=t;
            for(i=0;i<g[t].size();i++)
            {
                a[g[t][i]]--;
                if(!a[g[t][i]])q.push(g[t][i]);
            }
        }
        long long sum=0;t=INF;
        for(i=0;i<n;i++)
        {
            t=min(t,b[i]);
            sum+=t;
        }
        printf("%I64d\n",sum);
    }
    return 0;
}
