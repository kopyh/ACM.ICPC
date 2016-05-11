//kopyh
#include <bits/stdc++.h>
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define N 1123456
using namespace std;
int n,m,sum,res,flag;
char s[N],ss[N];
int a[10],b[10];
char ans[N],t1[N],t2[N];
int bigCmp(char *str1,char *str2)
{
    int len1=strlen(str1),len2=strlen(str2);
    if(len1>len2)
        return 1;
    if(len1<len2)
        return -1;
    for(int i=0;i<len1;i++)
    {
        if(str1[i]>str2[i])
            return 1;
        if(str1[i]<str2[i])
            return -1;
    }
    return 0;
}
int main()
{
    int i,j,k,cas,T,t,x,y,z;
    while(scanf("%s%s",s,ss)!=EOF)
    {
        n=strlen(s),m=strlen(ss);
        memset(a,0,sizeof(a));
        for(i=0;i<n;i++)a[s[i]-'0']++;
        for(i=0;i<m;i++)b[ss[i]-'0']++,a[ss[i]-'0']--;
        for(i=m;;i++)
        {
            t=i;
            while(t)a[t%10]--,t/=10;
            z=0;flag=1;
            for(j=0;j<10;j++)z+=a[j],flag=flag&&(a[j]>=0);
            if(z+m==i&&flag)break;
            t=i;
            while(t)a[t%10]++,t/=10;
        }
        int len=i;
        z=0;
        for(i=1;i<=9;i++)
            if(a[i])
                break;
        if(i<10)ans[z++]=i+'0';
        for(i=0;i<10;i++)
        {
            for(j=0;j<((len!=a[0]&&i==ans[0]-'0')?a[i]-1:a[i]);j++)
                ans[z++]=i+'0';
        }
        ans[z]='\0';
        x=0;
        for(i=1;i<m;i++)
        {
            if(ss[i]>ss[i-1])
            {
                x=1;
                break;
            }
            else if(ss[i]<ss[i-1])
                break;
        }
        if(x)
        {
            x=1;y=0;k=0;
            t1[k++]=ans[0];
            if(ans[0]=='\0')k--;
            while(ans[x]<=ss[y]&&x<z)
                t1[k++]=ans[x++];
            for(y=0;y<m;y++)
                t1[k++]=ss[y];
            while(x<z)
                t1[k++]=ans[x++];
            t1[k]='\0';
        }
        else
        {
            x=1,y=0;k=0;
            t1[k++]=ans[0];
            if(ans[0]=='\0')k--;
            while(ans[x]<ss[y]&&x<z)
                t1[k++]=ans[x++];
            for(y=0;y<m;y++)
                t1[k++]=ss[y];
            while(x<z)
                t1[k++]=ans[x++];
            t1[k]='\0';
        }
        if(ss[0]>'0')
        {
            for(i=0;i<10;i++)
                for(j=0;j<a[i];j++)
                    ss[m++]=i+'0';
            ss[m]='\0';
            if(bigCmp(t1,ss)==1||t1[0]=='0')
            {
                printf("%s\n",ss);
            }
            else
                printf("%s\n",t1);
        }
        else
        {
            printf("%s\n",t1);
        }
    }
    return 0;
}
