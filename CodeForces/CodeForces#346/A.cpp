//kopyh
#include <bits/stdc++.h>
using namespace std;
int n,m,sum,flag,res,x,y;
int main()
{
    while(scanf("%d%d%d",&n,&x,&y)!=EOF)
        printf("%d\n",(x+n*100+y-1)%n+1);
    return 0;
}
