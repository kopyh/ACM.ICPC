//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1e18+3
#define N 1123456
using namespace std;
int n,m,sum,res,flag;
double x[N],y[N];
double l[N];
int main()
{
    int i,j,k,cas,T,t;
    double ax,ay,bx,by,tx,ty;
    while(scanf("%lf%lf%lf%lf%lf%lf%d",&ax,&ay,&bx,&by,&tx,&ty,&n)!=EOF)
    {
        for(i=0;i<n;i++)
        {
            scanf("%lf%lf",&x[i],&y[i]);
            l[i] = 2.0*sqrt(1.0*(x[i]-tx)*(x[i]-tx)+1.0*(y[i]-ty)*(y[i]-ty));
        }
        double la1,la2,lb1,lb2;
        int a1,a2,b1,b2;
        la1=la2=lb1=lb2=MOD;
        for(i=0;i<n;i++)
        {
            double t=sqrt(1.0*(ax-x[i])*(ax-x[i])+1.0*(ay-y[i])*(ay-y[i]))+sqrt(1.0*(x[i]-tx)*(x[i]-tx)+1.0*(y[i]-ty)*(y[i]-ty))-l[i];
            if(t<la1)
                la2=la1,a2=a1,la1=t,a1=i;
            else if(t<la2)
                la2=t,a2=i;
           t=sqrt(1.0*(bx-x[i])*(bx-x[i])+1.0*(by-y[i])*(by-y[i]))+sqrt(1.0*(x[i]-tx)*(x[i]-tx)+1.0*(y[i]-ty)*(y[i]-ty))-l[i];
            if(t<lb1)
                lb2=lb1,b2=b1,lb1=t,b1=i;
            else if(t<lb2)
                lb2=t,b2=i;
        }
        double res=0,ans=0;
        for(i=0;i<n;i++)
            res+=l[i];
        ans=res+la1;
        ans=min(ans,res+lb1);
        if(a1==b1)
        {
            if(la1+lb2<la2+lb1)
                lb1=lb2,b1=b2;
            else
                la1=la2,a1=a2;
        }
        ans=min(ans,res+la1+lb1);
        printf("%.12lf\n",ans);
    }
    return 0;
}
