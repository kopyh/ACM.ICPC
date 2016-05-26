//kopyh
#pragma comment(linker, "/STACK:1024000000,1024000000")
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f3f3f3f3f
#define MOD 1000000007
#define N 212345
using namespace std;
long long n,m,res,flag;
vector<long long>g[N];
long long val[N],sum[N];
long long a[N],l[N],r[N];
void init()
{
    for(int i=0;i<n;i++)g[i].clear();
}
#define root 1 , n , 1
#define lson l , m , rt << 1
#define rson m + 1 , r , rt << 1 | 1
struct node
{
    long long pos,val;
    node(long long x=0,long long y=0){pos=x,val=y;}
    friend bool operator < (node a, node b)
    {
        return a.val < b.val;
    }
}arr[N<<2];
long long add[N<<2],tot;
void pushUp(long long rt)
{
    arr[rt] = max(arr[rt<<1],arr[rt<<1|1]);
}
void pushDown(long long l,long long r,long long rt)
{
    if(add[rt])
    {
        long long m = (l+r)>>1;
        add[rt<<1] += add[rt];
        add[rt<<1|1] += add[rt];
        arr[rt<<1].val += add[rt];
        arr[rt<<1|1].val += add[rt];
        add[rt] = 0;
    }
}
void updata(long long l,long long r,long long rt,long long ql,long long qr,long long val)
{
    if(l>qr||ql>r)return;
    if(l>=ql&&r<=qr)
    {
        arr[rt].val += val;
        add[rt] += val;
        return;
    }
    pushDown(l,r,rt);
    long long m = (l+r)>>1;
    if(ql<=m)updata(lson,ql,qr,val);
    if(qr>m)updata(rson,ql,qr,val);
    pushUp(rt);
}
void build(long long l,long long r,long long rt)
{
    add[rt]=0;
    if(l == r)
    {
        arr[rt].val = sum[a[++tot]];
        arr[rt].pos = tot;
        return;
    }
    long long m = (l+r)>>1;
    build(lson);
    build(rson);
    pushUp(rt);
}
node query(long long l,long long r,long long rt,long long ql,long long qr)
{
    if(l>qr||ql>r)
        return node(-INF,-INF);
    if(l>=ql&&r<=qr)
        return arr[rt];
    pushDown(l,r,rt);
    long long m = (l+r)>>1;
    return max(query(lson,ql,qr),query(rson,ql,qr));
}
void dfs(long long pos, long long fa)
{
    a[++tot]=pos;
    l[pos]=tot+1;
    for(int i=0;i<g[pos].size();i++)
    {
        if(g[pos][i]==fa)continue;
        sum[g[pos][i]]=sum[pos]+val[g[pos][i]];
        dfs(g[pos][i],pos);
    }
    r[pos]=tot+1;
}
int main()
{
    long long i,j,k,cas,T,t,x,y,z;
    scanf("%I64d",&T);
    cas=0;
    while(T--)
    {
        printf("Case #%I64d:\n",++cas);
        scanf("%I64d%I64d",&n,&m);
        init();
        for(i=0;i<n-1;i++)
        {
            scanf("%I64d%I64d",&x,&y);
            g[x].push_back(y);
            g[y].push_back(x);
        }
        for(i=0;i<n;i++)scanf("%I64d",&val[i]);
        sum[0]=val[0];tot=-1;
        dfs(0,0);
        tot=-1;
        build(root);
        for(i=0;i<m;i++)
        {
            scanf("%I64d",&z);
            if(z)
            {
                scanf("%I64d",&x);
                printf("%I64d\n",query(root,l[x],r[x]).val);
            }
            else
            {
                scanf("%I64d%I64d",&x,&y);
                updata(root,l[x],r[x],y-val[x]);
                val[x]=y;
            }
        }
    }
    return 0;
}
