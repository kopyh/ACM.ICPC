//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
int n,m,sum,res,flag;
int l[N],r[N],pos[N],vis[N];
char s[N],ss[N];
int main()
{
    int i,j,k,cas,T,t,x,y,z;
    while(scanf("%d%d%d",&n,&m,&z)!=EOF)
    {
        scanf("%s",s+1);
        stack<int>st;
        for(i=1;i<=n;i++)
        {
            if(s[i]=='(')st.push(i);
            else
            {
                x=st.top();
                st.pop();
                pos[x] = i;
                pos[i] = x;
            }
            l[i]=i-1;
            r[i]=i+1;
        }
        memset(vis,0,sizeof(vis));
        scanf("%s",ss);
        for(i=0;i<m;i++)
        {
            if(ss[i]=='L')z=l[z];
            else if(ss[i]=='R')z=r[z];
            else
            {
                x=min(z,pos[z]);
                y=max(z,pos[z]);
                r[l[x]]=r[y];
                l[r[y]]=l[x];
                z=r[y];
                if(z>n)z=l[x];
                vis[x]=vis[y]=1;
            }
        }
        z=1;
        while(vis[z])z=pos[z]+1;
        while(z<=n)
        {
            printf("%c",s[z]);
            z=r[z];
        }
        printf("\n");
    }
    return 0;
}
