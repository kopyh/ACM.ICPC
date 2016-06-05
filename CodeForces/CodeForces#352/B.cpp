//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
int n,m,sum,res,flag;
char s[N];
int a[30];
int main()
{
    int i,j,k,cas,T,t,x,y,z;
    while(scanf("%d",&n)!=EOF)
    {
        scanf("%s",s);sum=0;
        memset(a,0,sizeof(a));
        for(i=0;i<n;i++)sum+=(a[s[i]-'a']==0),a[s[i]-'a']=1;
        printf("%d\n",n>26?-1:n-sum);
    }
    return 0;
}
