//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 11234
using namespace std;
int n,m,sum,flag,res;
struct node
{
    int x;
    string s;
    friend bool operator < (node a, node b)
    {
        return a.x > b.x;
    }
};
vector<node>a[N];
int main()
{
    int i,j,k,cas,T,t,x,y,z;
    while(scanf("%d%d",&n,&m)!=EOF)
    {
        string s;
        for(i=0;i<n;i++)
        {
            cin>>s>>x>>y;
            node t;
            t.s=s,t.x=y;
            a[x].push_back(t);
        }
        for(i=1;i<=m;i++)
            sort(a[i].begin(),a[i].end());
        for(i=1;i<=m;i++)
        {
            if(a[i].size()>2&&a[i][1].x==a[i][2].x)printf("?\n");
            else cout<<a[i][0].s<<" "<<a[i][1].s<<endl;
        }
    }
    return 0;
}
