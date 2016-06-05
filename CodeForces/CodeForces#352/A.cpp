//kopyh
#include <bits/stdc++.h>
#define N 100010
#define INF 0x3f3f3f3f
#define MOD 1000000007
using namespace std;
int n,m,sum,res,flag;
int main()
{
    int i,j,k,t,x,y,T,cas;
    while(scanf("%d",&n)!=EOF)
    {
        for(i=1,x=y=1;;i++)
        {
            y*=10;
            if(n>(y-x)*i)n-=(y-x)*i;
            else
            {
                x+=n/i;
                if(n%i==0)
                {
                    x--;
                    res=x%10;
                    break;
                }
                for(j=0;j<i-n%i;j++)x/=10;
                res=x%10;
                break;
            }
            x=y;
        }
        printf("%d\n",res);
    }
    return 0;
}
