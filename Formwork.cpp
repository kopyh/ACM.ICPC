/*             My Formwork v2.10
*          2014-2016 code by kopyh
**********************************************************************************************************/
/******catalogue***********
简单算法与STL！！！！！
    STL
    输入输出外挂
	大数处理
		大数加法，减法，乘法, 比较, JAVA

DP！！！！！
    背包问题
        01背包
        完全背包
        多重背包
        二维费用背包
    长度DP
        最大连续子序列之和
        最大子矩阵和
        最长递增子序列(LIS)
        最长公共子序列(LCS)
    数位DP
    树DP
    区间DP
        最小代价构造回文串
        最大括号匹配数
        Sparse Table
            一维RMQ
            二维RMQ

搜索！！！！！
    DFS
    BFS(队列解法)
    A*启发式搜索算法
    IDA*迭代深化A*搜索
    Dancing Link

数据结构！！！！！
    并查集
    线段树
    划分树
    树状数组
    树链剖分
    字典树
    Treap
    伸展树(splay tree)
    主席树
        静态区间k大
        动态区间k大
        区间不相同数数量
        树上路径点权第k大
    哈希表
        ELFhash（字符串哈希函数）
    莫队算法

字符串！！！！！
    字符串最小表示法
    Manacher最长回文子串
    KMP
    扩展kmp
    AC自动机
    后缀数组
        DA倍增算法
        DC3算法
    后缀自动机

图论！！！！！！
    最小生成树
        prim
        kruskal
    次小生成树
    单源最短距离
        Dijkstra
        堆优化Dijkstra
        Bellman-Ford
        SPFA
    全图最短路
        Floyd
    欧拉回路
        fleury
    网络流
        最大流
            EdmondsKarp
            dinic
            SAP
            ISAP
        最小费用最大流
    二分图
        匈牙利算法
        Hopcroft-Karp算法
        KM算法
    强连通分量
        tarjan
    双连通分量
        边双连通分量
        点双连通分量
    最近公共祖先(LCA)
        tarjan
        ST-RMQ在线算法

数论！！！！！
    Fibonacci Number
    Greatest Common Divisor 最大公约数,欧几里德算法
    Lowest Common Multiple 最小公倍数
    扩展欧几里德算法
    快速幂运算
    乘积求模
    次方求模
        按位乘积法
        快速幂算法
    质数判断
        简单Prime判断
        Sieve Prime素数筛选法
        Miller-Rabin素数测试算法
    唯一分解定理，因子分解求和
    质因数分解
        pollard_rho 算法
    欧拉函数
    约瑟夫环
    高斯消元
        开关问题

组合数学！！！！！
    排列组合
    Lucas定理
    全排列
    错排公式
    母函数
        整数划分
    康托展开
    逆康托展开
    Catalan Number
    Stirling Number(Second Kind)
    容斥原理
    矩阵

计算几何！！！！！
    坐标向量等
    三角形
        三角形面积公式
        三角形内切圆半径公式
        三角形外接圆半径公式
        圆内接四边形面积公式
        多边形面积
    凸包
    三分求极值

博弈论！！！！！
    巴什博弈：
    威佐夫博弈：
    nim博弈：
    k倍减法博弈：
    sg函数
**/
///简单算法与STL！！！！！
#pragma comment(linker, "/STACK:1024000000,1024000000")

///C
# include<stdio.h>
sprintf(result,"%d%d",num1,num2);
sscanf(init_str,"%d %s",&num,&str);

#include <stdlib.h>
int *p; p=(int*)malloc(sizeof(int)); free(p);
int cmp( const void *a,const void *b )
{return *(int*)b-*(int*)a;}//降序
int cmp2(const void *a,const void *b)
{return *(double*)a>(double*)*b?1:-1;}//double
qsort(str,n,sizeof(str[0]),cmp);

#include <math.h>
double fabs(double x);
double ceil(double x);//向上取整
double floor(double x);//向下取整
double round(double x);//最接近x的整数
double asin(double arg);//求反正弦 arg∈[-1,1],返回值∈[-pi/2,+pi/2]
double sin(double arg);//求正弦 arg为弧度,返回值∈[-1, 1]
double exp(double arg);//求e的arg次方
double log(double num);//求num的对数,基数为e
k = (int)(log((double)n) / log(2.0));//2^k==n
double sqrt(double num);
double pow(double base,double exp);//求base的exp次方
memset( the_array, 0, sizeof(the_array) );
memcpy( the_array, src, sizeof(src));

#include <string.h>
int strcmp( const char *str1, const char *str2 );//str1<str2,return负数
void strcpy(int a[],int b[]);//字符串复制b赋给a
int strlen(int a[]);
char *strstr(char *str1, char *str2);//str1中str2串第一次出现的指针,找不到返回NULL

#include <time.h>
clock_t clockBegin, clockEnd;
clockBegin = clock();
//do something
clockEnd = clock();
printf("time:%d\n", clockEnd - clockBegin);

///C++
///iostream
#include<iostream>
cin.getline(s,N,'\n');

///sstream
#include<sstream>
string s;
stringstream ss;
stringstream ss(s);
ss.str(s);
ss.clear();
ss>>a>>b>>c;

///string
#include<string>
string s,st[maxn],ss(sss);//char sss[N];
s.c_str();//返回字符数组；
string st = s.substr(now,len);//截取从now处开始长度为len的子字符串；
st.assign(s,now,len);//功能同上
getline(cin,s,'\n');

///pair
pair<int,int> tmp;
tmp=make_pair<1,2>;
tmp.first; tmp.second;

///vector;
#include<vector>
vector<int>vs[n];
vector<int>::iterator it;
vs.front();  vs.back();   vs.push_back(x); vs.assign(begin,end);  vs.at(pos);
vs.size();   vs.empty();  vs.capacity();
vs.resize(num);  vs.reserve();
vs.begin();  vs.end();
vs.insert(it,x); vs.insert(it,n,x);   vs.insert(it,first,last);
vs.pop_back();   vs.erase(it);    vs.erase(first,last);    vs.clear();
vs.swap(v);
//去重，unique删除begin到end之间的相邻重复元素后返回一个新的结尾的迭代器
vs.erase(unique(s.begin(),s.end()),s.end())；

///map,key—value关联式容器，可利用key值快速查找记录
#include<map>
map<int,string> mp;//int做key值字符串为value
//string name; mp[name]+=123;
multimap<int, string> mp;//允许重复
map<string , int>::iterator it;//用指针处理
m.insert();
m.begin();  m.end();    m.find();
m.clear();  m.erase();
m.count();  m.empty();  m.size();
m.swap();
//返回一个非递减序列[first, last)中的第一个大于等于值val的位置的迭代期。
lower_bound(first,last,val);
//返回一个非递减序列[first, last)中第一个大于val的位置的迭代期。
upper_bound(first,last,val);
m.max_size();//返回可以容纳的最大元素个数

///stack：
#include<stack>
stack<int>s;
s.push(x);
s.pop();
s.top();
s.empty();  e.size();

///queue;
#include<queue>
queue<int>q;
q.push(x);
q.pop();
q.front();  q.back();
q.empty();  q.size();
//优先队列，优先输出优先级高的队列项:
struct node
{
    int x, y;
    friend bool operator < (node a, node b)
    {
//        if(a.y==b.y)return a.x > b.x;//二维判断，按x降序
        return a.y < b.y;//按y降序
    }
};
priority_queue<node>q;//定义结构体
struct mycmp
{
    bool operator()(const int &a,const int &b)
    {
        return a>b;//升序
    }
};
priority_queue<int,vector<int>,mycmp> q;
priority_queue<int,vector<int>,greater<int> > q;//数组升序优先队列
priority_queue<int,vector<int>,less<int> > q;//数组降序优先队列
q.top();//返回优先队列对顶元素

///set,红黑树的平衡二叉检索树建立，用于快速检索
#include<set>
struct cmp
{
    bool operator()( const int &a, const int &b ) const
    {   return a>b;  }//降序
};
set<int,cmp> st;
set<int> s;//默认升序
bool operator<(const node &a, const node &b)
{   return a.y > b.y; }//降序
struct node
{
    int x, y;
    friend bool operator < (node a, node b)
    {
//        if(a.y==b.y)return a.x > b.x;//二维判断，按x降序
        return a.y > b.y;//按y降序
    }
} t;
set<node> st;
multiset<int> st;//与set不同，允许相同元素
multiset<int>::iterator it;//取指针
it=st.begin();
//如果是:st.erase(*it);删除元素的话，所有的相同元素都会被删除
st.erase(it);   st.clear();
st.begin(); st.end(); st.rbegin();
st.insert();
st.count(); st.empty(); st.size();  st.find();
st.equal_range();//返回集合中与给定值相等的上下限的两个迭代器;
st.swap();
st.lower_bound();//返回指向(升序时)大于（或等于）某值的第一个元素的迭代器
//结构体：it = st.lower_bound(tn); printf("%d\n",(*it).x);
st.upper_bound();//返回大于某个值元素的迭代器

///bitset
#include<bitset>
unsigned long u;    string str;
bitset<N> bit;
bitset<N> bit(u);
bitset<N> bit(str);
bitset<N> bit(str,pos);
bitset<N> bit(str,pos,num);
bit.set();  bit.set(pos);   bit.reset();    bit.reset(pos);
bit.flip(); bit.flip(pos); //按位取反
bit.size(); bit.count();//1的个数
bit.any();  bit.none(); //有1，无1；return bool
bit.test(pos) == bit[pos]; //pos处为1？
u = bit.to_ulong();
str = bit.to_string(); //str[n-i-i]==bit[i]

///algorithm;
#include<algorithm>
sort(begin,end);//升序排列；
bool cmp(int a,int b)
{   return a>b; }//降序
bool cmp(node a,node b)
{
//    if(a.y == b.y)return a.x > b.x;//降序
	return a.y > b.y;//降序
}
struct node
{
    int x,y;
    friend bool operator < (node a, node b)
    {
//        if(a.y==b.y)return a.x < b.x;//二维判断，降序
        return a.y > b.y;//降序
    }
};
sort(begin,end,cmp);
sort(begin,end);//结构体中友元定义排序规则
stable_sort(begin,end,cmp);//稳定排序
//去除相邻的重复值(把重复的放到最后)，然后返回去重后的最后一个元素的地址；
unique(begin,end);
//在[begin,end)中查找val，返回第一个符合条件的元素的迭代器，否则返回end指针
find(begin,end,val);
//count，将返回从start到end 范围之内的序列中某个元素的数量。
n = count(begin,end,val);
next_permutation(begin,end);//返回该段序列的字典序下一种排列

///输入输出外挂
int Scan()
{
    int res=0,ch,flag=0;
    if((ch=getchar())=='-')
        flag=1;
    else if(ch>='0'&&ch<='9')
        res=ch-'0';
    while((ch=getchar())>='0'&&ch<='9')
        res=res*10+ch-'0';
    return flag?-res:res;
}
void Out(int a)
{
    if(a>9) Out(a/10);
    putchar(a%10+'0');
}

///大数处理
///大数加法
void bigAdd(char a[], char b[], char ans[])
{
    int lena = strlen(a), lenb = strlen(b);
    int i = lena-1,j = lenb-1,k = i>j?i:j, p = 0;
    while(i>=0 || j>=0)
    {
        int t = (i>=0?a[i]-'0':0)+(j>=0?b[j]-'0':0)+p;
        ans[k] = '0'+t%10;
        p = t/10;
        i--,j--,k--;
    }
    ans[lena>lenb?lena:lenb]='\0';
    if(p)
    {
        for(i=lena>lenb?lena:lenb; i>=0; i--)ans[i+1]=ans[i];
        ans[0]='0'+p;
    }
}

///大数减法
char * bigSub(char *str1,char *str2)
{
    int la=strlen(str1),lb=strlen(str2),flag=0,cp=0,i,j;
    char * temp;

    if(la<lb)flag=1;
    else if(la==lb)
        for(i=0;i<la;i++)
        {
            if(str1[i]==str2[i])continue;
            if(str1[i]>str2[i])break;
            if(str1[i]<str2[i]){flag=1;break;}
        }
    if(flag)
    {
        la^=lb;lb^=la;la^=lb;
        temp=str1;str1=str2;str2=temp;
    }
    for(la--,lb--,cp=0;la>=0||lb>=0;la--,lb--)
    {
        if(la>=0&&lb>=0&&str1[la]>=str2[lb]+cp){str1[la]=str1[la]-str2[lb]-cp+'0';cp=0;}
        else if(la>=0&&lb>=0&&str1[la]<str2[lb]+cp){str1[la]=str1[la]+10-str2[lb]-cp+'0';cp=1;}
        else if(la>=0&&lb<0&&str1[la]>='1'){str1[la]=str1[la]-cp;cp=0;}
        else if(la>=0&&lb<0&&str1[la]<'1'){str1[la]='9';cp=1;}
    }
    while(str1[i-1]=='0')
    {
        i--;
        if(i==0)printf("0");
    }
    if(flag)printf("-");
    return str1;
}
///大数乘法
char * bigMul(char *str1,char *str2)
{
    int l1=strlen(str1),l2=strlen(str2);
    int len=max(l1,l2)*10,cp=0,i,j;
    int af[len],bf[len],cf[len];
    memset(cf,0,sizeof(cf));
    for(i=0;i<l1;i++)
        af[i]=str1[l1-i-1]-'0';
    for(i=0;i<l2;i++)
        bf[i]=str2[l2-i-1]-'0';
    for(j=0;j<l2;j++)
        for(i=0;i<l1;i++)
        {
            cp=cf[i+j];
            cf[i+j]=(cp+bf[j]*af[i])%10;
            cf[i+1+j]+=(cp+bf[j]*af[i])/10;
        }
    i=l1+l2;
    while(!cf[i])i--;
    for(len=i;i>=0;i--)
        str1[len-i]=cf[i]+'0';
    str1[len+1]='\0';
    return str1;
}
///大数比较
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
///JAVA:
import java.io.PrintWriter;
import java.math.BigInteger;
import java.math.BigDecimal;
import java.util.Scanner;
public class Main {
    static BigInteger gcd(BigInteger a,BigInteger b){
        if(!b.equals(BigInteger.ZERO)) return gcd(b,a.mod(b));
        return a;
    }
    public static void main(String args[]){
        Scanner cin=new Scanner(System.in);
        PrintWriter cout = new PrintWriter(System.out, true);

        int n,m;
        BigInteger a,b,c;
        BigDecimal aa,bb,cc;
        String s;
        BigInteger zero = new BigInteger("0");
//        int T = cin.nextInt();
//        while(T>0)
        while(cin.hasNext())
        {
//        	T--;
            n = cin.nextInt();
	        a = BigInteger.valueOf(n);
	        b = cin.nextBigInteger();
	        c = cin.nextBigInteger(2);//二进制
	        System.out.println(c.toString(2));
        	aa = cin.nextBigDecimal();
        	System.out.println(a+" "+aa);

            if(a.compareTo(zero)==0)break;
	        a = a.add(b);
	        a = a.subtract(b);
	        a = a.divide(b);
	        a = a.multiply(b);
	        a = a.remainder(b);//a%b
	        a = a.pow(n);

	        s = aa.toPlainString();
        	if (s.charAt(n) == '0')
				cout.println("0");
            int x = 0, y = s.length() - 1;
            cout.println(s.substring(x, y));
        }
    }
}

///DP！！！！！
///背包问题
///01背包
//O(n*m)
//转移方程：dp[i][v]=max{dp[i-1][v],dp[i-1][v-c[i]]+w[i]}

memset(dp,0,sizeof(dp));//要求背包不一定全满
for(dp[0]=0,i=1;i<n;i++)dp[i]=-INF;//要求背包必须全满

for(i=0;i<n;i++)
    for(j=m;j>=w[i];j--)
        dp[j]=max{dp[j],dp[j-w[i]]+v[i]};
return dp[n][m];

///完全背包
//O(n*m)
//转移方程：dp[i][v]=max{dp[i-1][v-k*c[i]]+k*w[i]};

for(i=0;i<n;i++)
    for(j=w[i];j>=m;j--)
        dp[j]=max{dp[j],dp[j-w[i]]+v[i]};

///多重背包
//O(n*m)
//转移方程：dp[i][v]=max{dp[i-1][v-k*c[i]]+k*w[i]};

memset(dp,0,sizeof(dp));
for(i=0;i<n;i++)
{
    memset(counts,0,sizeof(counts));
    for(j=w[i];j<=m;j++)
        if(dp[j]<dp[j-w[i]]+v[i] && counts[j-w[i]]<sum[i])
            dp[j]=dp[j-w[i]]+v[i], counts[j]=counts[j-w[i]]+1;
}

///二维费用背包
//转移方程：dp[i][v][u]=max{dp[i-1][v][u],dp[i-1][v-a[i]][u-b[i]]+w[i]};

///长度DP
///最大连续子序列之和
//O(n)
int solve(int a[],int n)
{
    int i,maxx;
    int sum[N]={0};
    sum[0]=max(a[0],0);
    for(i=1;i<n;i++)
        sum[i]=max(sum[i-1]+a[i],a[i]);
    return sum[i-1];
}

///最大子矩阵和
//O(n*n*m)
int n,m,g[N][N],a[N];
int maxSumLen()
{
    a[0] = max(a[0],0);
    int ans = 0;
    for(int i=1;i<=m;i++)
        a[i]=max(a[i],a[i]+a[i-1]),ans=max(ans,a[i]);
    return ans;
}
int maxSum()
{
    int res = 0;
    for(int i=1;i<=n;i++)
        for(int j=i;j<=n;j++)
        {
            for(int k=1;k<=m;k++)
                a[k]=g[j][k]-g[i-1][k];
            res=max(res,maxSumLen());
        }
    return res;
}
void solve()
{
    memset(g,0,sizeof(g));
    for(int i=1;i<=n;i++)
        for(int j=1;j<=m;j++)
        {
            scanf("%d",&g[i][j]);
            g[i][j] += g[i-1][j];
        }
    printf("%d\n",maxSum());
}

///最长递增子序列(LIS)
//O(n^2)
//转移方程：b[k]=max(max(b[j]|a[j]<a[k],j<k)+1,1);
int LIS(int *a,int len)
{
    int i,j,maxx=0;
    int b[N];
    for(i=0;i<len;i++)
    {
        b[i] = 1;
        for(j=0;j<i;j++)
            if(a[i]>a[j])
                b[i] = max(b[i],b[j]+1);
    }
    for(i=0;i<len;i++)
        if(b[i]>maxx)
            maxx = b[i];
    return maxx;
}

//二分O(n*logn);
int fen(int num,int l,int r,int b[])
{
    int mid;
    while(l<=r)
    {
        mid=(l+r)>>1;
        if(num>=b[mid]) l=mid+1;
        else r=mid-1;
    }
    return l;
}
int LIS(int a[],int b[],int len)
{
    int i,lenn;
    b[0]=a[0];
    lenn=1;
    for(i=1;i<len;i++)
    {
        if(a[i]>=b[lenn-1]) b[lenn++]=a[i];
        else b[fen(a[i],0,lenn-1,b)]=a[i];
    }
    return lenn;
}

///最长公共子序列(LCS)
//O(n*m)
int dp[N][N],n,m;
int LCS(char *a, char *b)
{
	memset(dp,0,sizeof(dp));
	for(int i = 1; i <= n; i++)
		for(int j = 1; j <= m; j++)
        {
			if(a[i-1] == b[j-1])
                dp[i][j] = dp[i-1][j-1] + 1;
            else
                dp[i][j] = max(dp[i][j-1],dp[i-1][j]);
		}
    return dp[n][m];
}

///数位dp
//是13的倍数包含13的个数
///预处理
//dp[i][j][k]第i位余数为j的是否包含13的数的个数，k==0不包含，k==1首位为3，k==2包含13
int dp[N][13][3],a[N];
void init()
{
    a[0]=1;
    for(int i=1;i<N;i++)
        a[i] = a[i-1]*10%13;

    memset(dp,0,sizeof(dp));
    dp[0][0][0] = 1;
    for(int i=1;i<N;i++)
        for(int j=0;j<13;j++)
        {
            for(int k=0;k<10;k++)
            {
				dp[i][(j+a[i-1]*k)%13][0]+=dp[i-1][j][0];
				dp[i][(j+a[i-1]*k)%13][2]+=dp[i-1][j][2];
			}
			dp[i][(j+a[i-1]*3)%13][1] += dp[i-1][j][0];
            dp[i][(j+a[i-1])%13][0] -= dp[i-1][j][1];
            dp[i][(j+a[i-1])%13][2] += dp[i-1][j][1];
        }
}
int solve(int x)
{
    int dig[N],len=0;
    while(x)dig[len++]=x%10,x/=10;
    dig[len]=0;
    int flag=0,ans=0,mod=0;
    for(int i=len-1;i>=0;i--)
    {
//枚举当前位为j，得到前面的余数为(mod+j*a[i])%13，
//后面位数中余数是(13-(mod+j*a[i])%13)%13的个数都是答案。
        for(int j=0;j<dig[i];j++)
            ans+=dp[i][(13-(mod+j*a[i])%13)%13][2];
        if(flag)
        {
//前面已经有了13，只要是前面和后面所有位数的余数为0的就是答案。
            for(int j=0;j<dig[i];j++)
                ans+=dp[i][(13-(mod+j*a[i])%13)%13][0];
        }
        else
        {
			//首位是3的余数和为0就是答案。
            if(dig[i+1]==1&&dig[i]>3)
                ans+=dp[i+1][(13-mod)%13][1];
			//取当前位为1后面首位为3的是答案。
            if(dig[i]>1)
                ans+=dp[i][(13-(mod+a[i])%13)%13][1];
        }
        if(dig[i+1]==1&&dig[i]==3)flag=1;
        mod=(mod+dig[i]*a[i])%13;
    }
    return ans;
}

///dfs
int dp[N][13][3],dig[N];
//len:当前位，mod:前面留下的余数大小,k:前面是否出现13，flag:前面每一位是否都是上限。
int dfs(int len, int mod, int k, int flag)
{
    if(len<=0)return (!mod && k==2);
    if(!flag && dp[len][mod][k]!=-1)return dp[len][mod][k];
    int num = flag?dig[len]:9;
    int ans=0;
    for(int i=0;i<=num;i++)
    {
        int modt = (mod*10+i)%13;
        int kt;
        if(k==2 || k==1&&i==3)kt=2;
        else if(i==1)kt=1;
        else kt=0;

        ans+=dfs(len-1,modt,kt,flag&&num==i);
    }
    if(!flag)dp[len][mod][k] = ans;
    return ans;
}
int solve(int x)
{
    memset(dp,-1,sizeof(dp));
    int len=0;
    while(x)dig[++len]=x%10,x/=10;
    return dfs(len,0,0,1);
}

///树dp
//O(n*m*m)
//n节点技能树点m个技能获得最大权值的办法
//dp[k][i] 在以k节点为根节点的子树中选择i权值的最大value, w[i] i节点所需权值
vector<int>g[N];
int dp[N][N],w[N];
void init(int n, int m)
{
    for(int i=0;i<=n;i++)for(int j=0;j<=m;j++)dp[i][j]=-INF;
    for(int i=0;i<=n;i++)g[i].clear();
}
void dfs(int pos,int fa)
{
    int i,j,k;
    for(i=0;i<g[pos].size();i++)if(g[pos][i]!=fa)dfs(g[pos][i],pos);
    for(k=0,i=0;i<=m;i++)k=max(k,dp[pos][i]),dp[pos][i]=k;
    for(i=0;i<g[pos].size();i++)
        if(g[pos][i]!=fa)
            for(j=m;j>=w[pos];j--)//降序更新，防止重复加上子树的权值
                for(k=1;j+k<=m;k++)
                    dp[pos][j+k] = max(dp[pos][j+k], dp[pos][j]+dp[g[pos][i]][k]);
}

///区间DP
///最小代价构造回文串；
//O(n^2)
//w[]储存增添或删除不同字符所需代价，S[]储存原始串；
//f[i][j] 将i到j构造为回文串所需的代价。
int getRes(int w[],char s[])
{
    int i,j,len = strlen(s);
    int f[len][len]={0};
    for(i=len-1;i>=0;i--)
        for(j=i;j<len;j++)
        {
            f[i][j] = min(f[i+1][j]+w[s[i]],f[i][j-1]+w[s[j]]);
            if(s[i]==s[j])
                f[i][j] = min(f[i][j],f[i+1][j-1]);
        }
    return f[0][len-1];
}

///括号匹配
//O(n^3)
//求最大括号匹配数
bool is(char a,char b)
{
    return a=='('&&b==')' || a=='['&&b==']' || a=='{'&&b=='}';
}
int getRes(char s[])
{
    int len = strlen(s);
    int i,j,k,f[len][len];
    memset(f,0,sizeof(f));
    for(i=len-2;i>=0;i--)
        for(j=i+1;j<len;j++)
        {
            f[i][j] = f[i+1][j];
            for(k=i+1;k<=j;k++)
                if(is(s[i],s[k]))
                    f[i][j] = max(f[i][j],f[i+1][k-1]+f[k+1][j]+2);
        }
    return f[0][len-1];
}

///Sparse Table
///一维RMQ
//ST(n)初始化：O(n*logn)
//dp[i][j] 从i到i+2^j -1中最小的一个值(从i开始持续2^j个数)
//dp[i][j]=min{dp[i][j-1],dp[i+2^(j-1)][j-1]}
//RMQ(i,j)查询
//将i-j 分成两个2^k的区间 k=(int)log2(i-j+1)
//查询结果应该为 min(f2[i][k],f2[j-2^k+1][k])
int dp[N][35],num[N];
void ST(int n)
{
    for(int i=0;i<n;i++)
        dp[i][0] = num[i];

    int k = (int)(log((double)n) / log(2.0));
    for(int j=1;j<=k;j++)
        for(int i=0;i+(1<<j)-1<n;i++)
            dp[i][j] = max(dp[i][j-1], dp[i+(1<<(j-1))][j-1]);
}
//查询i和j之间的最值,注意i是从0开始的
int RMQ(int l,int r)
{
    int mid = (int)(log((double)(r-l+1)) / log(2.0));
    int t = max(dp[l][mid],dp[r-(1<<mid)+1][mid]);
    return t;
}

///二维RMQ
//预处理 O(n*m*logn*logm)
//数组下标从1开始
int val[N][N];
int dp[N][N][9][9];
int mm[N];//二进制位数减一，使用前初始化
void initRMQ(int n,int m)
{
    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= m; j++)
            dp[i][j][0][0] = val[i][j];
    for(int ii = 0; ii <= mm[n]; ii++)
        for(int jj = 0; jj <= mm[m]; jj++)
            if(ii+jj)
                for(int i = 1; i + (1<<ii) - 1 <= n; i++)
                    for(int j = 1; j + (1<<jj) - 1 <= m; j++)
                    {
                        if(ii)dp[i][j][ii][jj] = max(dp[i][j][ii-1][jj],dp[i+(1<<(ii-1))][j][ii-1][jj]);
                        else dp[i][j][ii][jj] = max(dp[i][j][ii][jj-1],dp[i][j+(1<<(jj-1))][ii][jj-1]);
                    }
}
//查询矩形内的最大值(x1<=x2,y1<=y2)
int rmq(int x1,int y1,int x2,int y2)
{
    int k1 = mm[x2-x1+1];
    int k2 = mm[y2-y1+1];
    x2 = x2 - (1<<k1) + 1;
    y2 = y2 - (1<<k2) + 1;
    return max(max(dp[x1][y1][k1][k2],dp[x1][y2][k1][k2]),max(dp[x2][y1][k1][k2],dp[x2][y2][k1][k2]));
}
int main()
{
    //在外面对mm数组进行初始化
    mm[0] = -1;
    for(int i = 1; i < N; i++)
        mm[i] = ((i&(i-1))==0)?mm[i-1]+1:mm[i-1];
    int n,m;
    int Q;
    int r1,c1,r2,c2;
    while(scanf("%d%d",&n,&m) == 2)
    {
        for(int i = 1; i <= n; i++)
            for(int j = 1; j <= m; j++)
                scanf("%d",&val[i][j]);
        initRMQ(n,m);
        scanf("%d",&Q);
        while(Q--)
        {
            scanf("%d%d%d%d",&r1,&c1,&r2,&c2);
            if(r1 > r2)swap(r1,r2);
             if(c1 > c2)swap(c1,c2);
            int tmp = rmq(r1,c1,r2,c2);
            printf("%d ",tmp);
        }
    }
    return 0;
}

///搜索！！！！！
///DFS
///BFS
///A*启发式搜索算法(曼哈顿算法处理估价函数)
//带有估价函数的广搜
///IDA*迭代深化A*搜索
//就是深搜，不断加大深搜上限，加上一个估价函数，
//如过估价值加上当前搜索深度大于搜索深度上限的话就剪枝

///Dancing Link
//用于求精确覆盖问题，数独的每个位置转换为覆盖矩阵的4个位置
#define N 9
const int MaxN = N*N*N + 10;
const int MaxM = N*N*4 + 10;
const int maxnode = MaxN*4 + MaxM + 10;
char g[MaxN];
struct DLX
{
    int n,m,size;
    int U[maxnode],D[maxnode],R[maxnode],L[maxnode],Row[maxnode],Col[maxnode];
    int H[MaxN],S[MaxM];
    int ansd,ans[MaxN];
    void init(int _n,int _m)
    {
        n = _n;
        m = _m;
        for(int i = 0; i <= m; i++)
        {
            S[i] = 0;
            U[i] = D[i] = i;
            L[i] = i-1;
            R[i] = i+1;
        }
        R[m] = 0;
        L[0] = m;
        size = m;
        for(int i = 1; i <= n; i++)H[i] = -1;
    }
    void Link(int r,int c)
    {
        ++S[Col[++size]=c];
        Row[size] = r;
        D[size] = D[c];
        U[D[c]] = size;
        U[size] = c;
        D[c] = size;
        if(H[r] < 0)H[r] = L[size] = R[size] = size;
        else
        {
            R[size] = R[H[r]];
            L[R[H[r]]] = size;
            L[size] = H[r];
            R[H[r]] = size;
        }
    }
    void remove(int c)
    {
        L[R[c]] = L[c];
        R[L[c]] = R[c];
        for(int i = D[c]; i != c; i = D[i])
            for(int j = R[i]; j != i; j = R[j])
            {
                U[D[j]] = U[j];
                D[U[j]] = D[j];
                --S[Col[j]];
            }
    }
    void resume(int c)
    {
        for(int i = U[c]; i != c; i = U[i])
            for(int j = L[i]; j != i; j = L[j])
                ++S[Col[U[D[j]]=D[U[j]]=j]];
        L[R[c]] = R[L[c]] = c;
    }
    bool Dance(int d)
    {
        if(R[0] == 0)
        {
            for(int i = 0; i < d; i++)g[(ans[i]-1)/9] = (ans[i]-1)%9 + '1';
            for(int i = 0; i < N*N; i++)printf("%c",g[i]);
            printf("\n");
            return true;
        }
        int c = R[0];
        for(int i = R[0]; i != 0; i = R[i])
            if(S[i] < S[c])
                c = i;
        remove(c);
        for(int i = D[c]; i != c; i = D[i])
        {
            ans[d] = Row[i];
            for(int j = R[i]; j != i; j = R[j])remove(Col[j]);
            if(Dance(d+1))return true;
            for(int j = L[i]; j != i; j = L[j])resume(Col[j]);
        }
        resume(c);
        return false;
    }
};
void place(int &r,int &c1,int &c2,int &c3,int &c4,int i,int j,int k)
{
    r = (i*N+j)*N + k;
    c1 = i*N+j+1;
    c2 = N*N+i*N+k;
    c3 = N*N*2+j*N+k;
    c4 = N*N*3+((i/3)*3+(j/3))*N+k;
}
DLX dlx;
int main()
{
    while(scanf("%s",g) == 1)
    {
        dlx.init(N*N*N,N*N*4);
        int r,c1,c2,c3,c4;
        for(int i = 0; i < N; i++)
            for(int j = 0; j < N; j++)
                for(int k = 1; k <= N; k++)
                    if(g[i*N+j] == '.' || g[i*N+j] == '0'+k)
                    {
                        place(r,c1,c2,c3,c4,i,j,k);
                        dlx.Link(r,c1);
                        dlx.Link(r,c2);
                        dlx.Link(r,c3);
                        dlx.Link(r,c4);
                    }
        dlx.Dance(0);
    }
    return 0;
}

///数据结构！！！！！
///并查集
void init(int n)
{
    for(int i=1;i<=n;i++)f[i]=i;
}
int getf(int v)
{
    while(f[v] != v)
    {
        f[v]=f[f[v]];
        v = f[v];
    }
    return f[v];
}
void unions(int x,int y)
{
    x = getf(x);
    y = getf(y);
    if(x == y) return;
    f[y] = x;
}

///线段树
#define root 1 , n , 1
#define lson l , m , rt << 1
#define rson m + 1 , r , rt << 1 | 1

int sum[N<<2],add[N<<2];
void pushUp(int rt)
{
    sum[rt] = sum[rt<<1]+sum[rt<<1|1];
}
void pushDown(int l,int r,int rt)
{
    if(add[rt])
    {
        int m = (l+r)>>1;
        add[rt<<1] += add[rt];
        add[rt<<1|1] += add[rt];
        sum[rt<<1] += (m-l+1)*add[rt];
        sum[rt<<1|1] += (r-m)*add[rt];
        add[rt] = 0;
    }
}
void update(int l,int r,int rt,int ql,int qr,int val)
{
    if(l>qr||ql>r)return;
    if(l>=ql&&r<=qr)
    {
        sum[rt] += (r-l+1)*val;
        add[rt] += val;
        return;
    }
    pushDown(l,r,rt);
    int m = (l+r)>>1;
    if(ql<=m)update(lson,ql,qr,val);
    if(qr>m)update(rson,ql,qr,val);
    pushUp(rt);
}
void build(int l,int r,int rt)
{
    add[rt]=0;
    if(l == r)
    {
        scanf("%d",&sum[rt]);
        return;
    }
    int m = (l+r)>>1;
    build(lson);
    build(rson);
    pushUp(rt);
}
int query(int l,int r,int rt,int ql,int qr)
{
    if(l>qr||ql>r)
        return 0;
    if(l>=ql&&r<=qr)
        return sum[rt];
    pushDown(l,r,rt);
    int m = l+r>>1;
    return query(l,m,rt<<1,ql,qr)+query(m+1,r,rt<<1|1,ql,qr);
}

///划分树
//求区间第K大值，区间内大于小于k大值的数的和
#define N 123456
int sorted[N]={0};    //对原集合中元素排序后的值
int val[20][N]={0};  //val记录第k层当前位置的值
int num[20][N]={0}; //记录元素所在区间当前位置前的元素进入到左子树的个数
int lnum, rnum;    //询问区间里面k-th数左侧和右侧的数的个数
long long sum[20][N]={0};//记录比当前元素小的元素的和
long long lsum, rsum;   //询问区间里面k-th数左侧数之和与右侧数之和
void build(int l, int r, int d)
{
    if (l == r) return ;
    int mid = (l + r) >> 1;
    int same = mid - l + 1;
    for (int i=l; i<=r; i++)
        if (val[d][i] > sorted[mid])
            same--;
    int lp = l, rp = mid+1;
    for (int i=l; i<=r; i++)
    {
        if (i == l)num[d][i]=0, sum[d][i]=0;
        else num[d][i]=num[d][i-1], sum[d][i]=sum[d][i-1];
        if (val[d][i] > sorted[mid])
            num[d][i]++, sum[d][i]+=val[d][i], val[d+1][lp++]=val[d][i];
        else if (val[d][i] < sorted[mid])
            val[d+1][rp++] = val[d][i];
        else
        {
            if (same)
            {
                same--;
                num[d][i]++;
                sum[d][i] += val[d][i];
                val[d+1][lp++] = val[d][i];
            }
            else
                val[d+1][rp++] = val[d][i];
        }
    }
    build(l, mid, d+1);
    build(mid+1, r, d+1);
}
int query(int a, int b, int k, int l, int r, int d)
{
    if (a == b)
        return val[d][a];
    int mid = (l + r) >> 1;
    int s, ss;
    long long sss;
    if (a == l)
    {
        s = num[d][b];
        ss = 0;
        sss = sum[d][b];
    }
    else
    {
        s = num[d][b] - num[d][a-1];
        ss = num[d][a-1];
        sss = sum[d][b] - sum[d][a-1];
    }
    if (s >= k)
    {
        a = l + ss;
        b = l + ss + s - 1;
        return query(a, b, k, l, mid, d+1);
    }
    else
    {
        lnum += s;
        lsum += sss;
        a = mid+1 + a - l - ss;
        b = mid+1 + b - l - num[d][b];
        return query(a, b, k-s, mid+1, r, d+1);
    }
}
void solve(int n,int m)
{
    long long s[N]={0};
    for(int i=1;i<=n;i++)
    {
        scanf("%d",&sorted[i]);
        val[0][i] = sorted[i];
        s[i] = s[i-1] + sorted[i];
    }
    sort(sorted+1,sorted+1+n);
    build(1,n,0);

    int x,y,k,res;
    while(m--)
    {
        scanf("%d%d%d",&x,&y,&k);
        lsum = lnum = 0;
        res = query(x,y,k,1,n,0);
        printf("%d ",res);  //输出第k大值
        rnum = y-x+1 - lnum;
        rsum = s[y] - s[x-1] - lsum - res;
        printf("%lld %lld\n",lsum,rsum);    //区间内比第k大值小的和大的值的和；
    }
}

///树状数组
#define N 1010000
int inverse[N];
void init()
{
    memset(inverse,0,sizeof(inverse));
}
int lowbit(int t)
{
	return t & (t^(t-1));//2^k(k是t末尾零的个数)
        //return t & (-t);
}
void add(int pos,int num)
{
	while (pos<=n)
    {
		inverse[pos]+=num;
		pos+=lowbit(pos);
	}
}
int getSum(int now)
{
	int sum=0;
	while (now>0)
    {
		sum+=inverse[now];
		now-=lowbit(now);
	}
	return sum;
}

///树链剖分
#define root 1 , tot , 1
#define lson l , m , rt << 1
#define rson m + 1 , r , rt << 1 | 1
//dep节点深度,siz节点子树大小,fa节点父亲,id节点在线段树中位置序号,top节点所在重链的顶端,val节点新编号对应父边权值
int dep[N],siz[N],fa[N],id[N],son[N],top[N],val[N];
int tot,n;
vector<int>v[N];
struct tree
{
    int x,y,val;
}e[N];
//重儿子
void dfs1(int u,int f,int d)
{
    dep[u]=d;siz[u]=1;son[u]=0;fa[u]=f;
    for(int i=0;i<v[u].size();i++)
    {
        int ff=v[u][i];
        if(f==ff)continue;
        dfs1(ff,u,d+1);
        siz[u]+=siz[ff];
        if(siz[son[u]]<siz[ff])son[u]=ff;
    }
}
//重链
void dfs2(int u,int tp)
{
    top[u]=tp;id[u]=++tot;
    if(son[u])dfs2(son[u],tp);
    for(int i=0;i<v[u].size();i++)
    {
        int ff=v[u][i];
        if(ff==fa[u] || ff==son[u])continue;
        dfs2(ff,ff);
    }
}
void init()
{
    tot=0;
    for(int i=1;i<=n;i++)
        v[i].clear();
}
int sum[N<<2];
void pushUp(int rt)

{
    sum[rt] = max(sum[rt<<1],sum[rt<<1|1]);
}
void update(int l,int r,int rt,int ql,int qr,int val)
{
    if(l>qr||ql>r)return;
    if(l>=ql&&r<=qr)
    {
        sum[rt] = val;
        return;
    }
    int m = (l+r)>>1;
    if(ql<=m)update(lson,ql,qr,val);
    if(qr>m)update(rson,ql,qr,val);
    pushUp(rt);
}
void build(int l,int r,int rt)
{
    if(l == r)
    {
        sum[rt] = val[l];
        return;
    }
    int m = (l+r)>>1;
    build(lson);
    build(rson);
    pushUp(rt);
}
int query(int l,int r,int rt,int ql,int qr)
{
    if(l>qr||ql>r)
        return 0;
    if(l>=ql&&r<=qr)
        return sum[rt];
    int m = l+r>>1;
    return max(query(l,m,rt<<1,ql,qr),query(m+1,r,rt<<1|1,ql,qr));
}
//寻找x,y之间的最大权值边
int solve(int x,int y)
{
    int xx=top[x],yy=top[y];
    int ans=0;
    while(xx!=yy)
    {
        if(dep[xx]<dep[yy])
        {
            swap(xx,yy);
            swap(x,y);
        }
        ans = max(ans,query(root,id[xx],id[x]));
        x=fa[xx];
        xx=top[x];
    }
    if(x==y)return ans;
    if(dep[x] > dep[y])swap(x,y);
    ans = max(ans,query(root,id[son[x]],id[y]));
    return ans;
}
int main()
{
    int i,j,k,kk,cas,T,t,x,y,z;
    init();
    scanf("%d",&n);
    for(i=1;i<n;i++)
    {
        scanf("%d%d%d",&e[i].x,&e[i].y,&e[i].val);
        v[e[i].x].push_back(e[i].y);
        v[e[i].y].push_back(e[i].x);
    }
    dfs1(1,0,1);
    dfs2(1,1);
    for(i=1;i<n;i++)
    {
        if(dep[e[i].x] < dep[e[i].y])swap(e[i].x,e[i].y);
        val[id[e[i].x]] = e[i].val;
    }
    build(root);
    char s[202];
    while(scanf("%s",s)!=EOF)
    {
        if(strcmp(s,"DONE")==0)break;
        scanf("%d%d",&x,&y);
        if(strcmp(s,"QUERY")==0)printf("%d\n",solve(x,y));
        else if(strcmp(s,"CHANGE")==0)update(root,id[e[x].x],id[e[x].x],y);
    }
    return 0;
}

///字典树
int sum=0,res=0;
struct node
{
    int x,y;
    int r;
    int next[26];
    void init()
    {
        r=0;
        memset(next,-1,sizeof(next));
    }
}tree[1000005];
void insert_tree(char *str)
{
    int i, j, t;
    i = j = t = 0;
    while(str[i])
    {
        t = str[i]-'a';
        if(tree[j].next[t] == -1)
        {
            tree[sum].init();
            tree[j].next[t] = sum++;
        }
        j = tree[j].next[t];
        tree[j].r++;//记录以此为前缀的单词的数量
        i++;
    }
}
void query_tree(char *str)
{
    int i,j,t;
    i = j = t = 0;
    while(str[i])
    {
        t = str[i]-'a';
        if(tree[j].next[t] == -1)
        {
            res=0;
            return;
        }
        j=tree[j].next[t];
        i++;
    }
    res=tree[j].r;//以此为前缀的单词的数量
}

///Treap
struct Treap
{
    int size;
    int key,fix;
    Treap *ch[2];
    Treap(int key)
    {
        size=1;
        fix=rand();
        this->key=key;
        ch[0]=ch[1]=NULL;
    }
    int compare(int x) const
    {
        if(x==key) return -1;
        return x<key? 0:1;
    }
    void Maintain()
    {
        size=1;
        if(ch[0]!=NULL) size+=ch[0]->size;
        if(ch[1]!=NULL) size+=ch[1]->size;
    }
};
void Rotate(Treap* &t,int d)
{
    Treap *k=t->ch[d^1];
    t->ch[d^1]=k->ch[d];
    k->ch[d]=t;
    t->Maintain();  //必须先维护t，再维护k，因为此时t是k的子节点
    k->Maintain();
    t=k;
}
void Insert(Treap* &t,int x)
{
    if(t==NULL) t=new Treap(x);
    else
    {
        //int d=t->compare(x);   //如果值相等的元素只插入一个
        int d=x < t->key ? 0:1;  //如果值相等的元素都插入
        Insert(t->ch[d],x);
        if(t->ch[d]->fix > t->fix)
            Rotate(t,d^1);
    }
    t->Maintain();
}
//一般来说，在调用删除函数之前要先用Find()函数判断该元素是否存在
void Delete(Treap* &t,int x)
{
    int d=t->compare(x);
    if(d==-1)
    {
        Treap *tmp=t;
        if(t->ch[0]==NULL)
        {
            t=t->ch[1];
            delete tmp;
            tmp=NULL;
        }
        else if(t->ch[1]==NULL)
        {
            t=t->ch[0];
            delete tmp;
            tmp=NULL;
        }
        else
        {
            int k=t->ch[0]->fix > t->ch[1]->fix ? 1:0;
            Rotate(t,k);
            Delete(t->ch[k],x);
        }
    }
    else Delete(t->ch[d],x);
    if(t!=NULL) t->Maintain();
}
bool Find(Treap *t,int x)
{
    while(t!=NULL)
    {
        int d=t->compare(x);
        if(d==-1) return true;
        t=t->ch[d];
    }
    return false;
}
//求第k大的数
int Kth(Treap *t,int k)
{
    if(t==NULL||k<=0||k>t->size)
        return -1;
    if(t->ch[0]==NULL&&k==1)
        return t->key;
    if(t->ch[0]==NULL)
        return Kth(t->ch[1],k-1);
    if(t->ch[0]->size>=k)
        return Kth(t->ch[0],k);
    if(t->ch[0]->size+1==k)
        return t->key;
    return Kth(t->ch[1],k-1-t->ch[0]->size);
}
//求x是升序排列的第几个的数
int Rank(Treap *t,int x)
{
    int r;
    if(t->ch[0]==NULL) r=0;
    else  r=t->ch[0]->size;
    if(x==t->key) return r+1;
    if(x<t->key)
        return Rank(t->ch[0],x);
    return r+1+Rank(t->ch[1],x);
}
void DeleteTreap(Treap* &t)
{
    if(t==NULL) return;
    if(t->ch[0]!=NULL) DeleteTreap(t->ch[0]);
    if(t->ch[1]!=NULL) DeleteTreap(t->ch[1]);
    delete t;
    t=NULL;
}
void Print(Treap *t)
{
    if(t==NULL) return;
    Print(t->ch[0]);
    cout<<t->key<<endl;
    Print(t->ch[1]);
}
Treap *root=NULL;

///伸展树(splay tree)
//数列的插入，删除，修改，翻转，循环，求极值，求和
class SplayTree
{
private:
    struct SplayNode
    {
        int value, sizes, lazy;
        SplayNode *parent, *lchild, *rchild;
        int mins;
        bool isReverse;
    } nil, node[N + N];
    int nodeNumber;
    SplayNode *root;
    SplayNode *newNode(SplayNode *parent, const int value)
    {
        node[nodeNumber].value = value;
        node[nodeNumber].sizes = 1;
        node[nodeNumber].lazy = 0;
        node[nodeNumber].parent = parent;
        node[nodeNumber].lchild = &nil;
        node[nodeNumber].rchild = &nil;
        node[nodeNumber].mins = value;
        node[nodeNumber].isReverse = false;
        return &node[nodeNumber++];
    }
    SplayNode *make(int l, int r, SplayNode *parent, int arrays[])
    {
        if(l > r) return &nil;
        int mid = (l + r) >> 1;
        SplayNode *x = newNode(parent, arrays[mid]);
        x->lchild = make(l, mid - 1, x, arrays);
        x->rchild = make(mid + 1, r, x, arrays);
        update(x);
        return x;
    }
    void update(SplayNode *x)
    {
        if(x == &nil) return;
        x->sizes = x->lchild->sizes + x->rchild->sizes + 1;
        x->mins = min(x->value, min(x->lchild->mins, x->rchild->mins));
    }
    void pushdown(SplayNode *x)
    {
        if(x == &nil) return;
        if(x->isReverse)
        {
            swap(x->lchild, x->rchild);
            x->lchild->isReverse ^= true;
            x->rchild->isReverse ^= true;
            x->isReverse = false;
        }
        if(x->lazy)
        {
            x->value += x->lazy;
            x->mins += x->lazy;
            x->lchild->lazy += x->lazy;
            x->rchild->lazy += x->lazy;
            x->lazy = 0;
        }
    }
    void rotateLeft(SplayNode *x)
    {
        SplayNode *p = x->parent;
        pushdown(x->lchild);
        pushdown(x->rchild);
        pushdown(p->lchild);
        p->rchild = x->lchild;
        p->rchild->parent = p;
        x->lchild = p;
        x->parent = p->parent;
        if(p->parent->lchild == p) p->parent->lchild = x;
        else p->parent->rchild = x;
        p->parent = x;
        update(p);
        update(x);
        if(root == p) root = x;
    }
    void rotateRight(SplayNode *x)
    {
        SplayNode *p = x->parent;
        pushdown(x->lchild);
        pushdown(x->rchild);
        pushdown(p->rchild);
        p->lchild = x->rchild;
        p->lchild->parent = p;
        x->rchild = p;
        x->parent = p->parent;
        if(p->parent->lchild == p) p->parent->lchild = x;
        else p->parent->rchild = x;
        p->parent = x;
        update(p);
        update(x);
        if(root == p) root = x;
    }
    void splay(SplayNode *x, SplayNode *y)
    {
        pushdown(x);
        while(x->parent != y)
        {
            if(x->parent->parent == y)
            {
                if(x->parent->lchild == x) rotateRight(x);
                else rotateLeft(x);
            }
            else if(x->parent->parent->lchild == x->parent)
            {
                if(x->parent->lchild == x)
                {
                    rotateRight(x->parent);
                    rotateRight(x);
                }
                else
                {
                    rotateLeft(x);
                    rotateRight(x);
                }
            }
            else
            {
                if(x->parent->rchild == x)
                {
                    rotateLeft(x->parent);
                    rotateLeft(x);
                }
                else
                {
                    rotateRight(x);
                    rotateLeft(x);
                }
            }
        }
        update(x);
    }
    void finds(int k, SplayNode *y)
    {
        SplayNode *x = root;
        pushdown(x);
        while(k != x->lchild->sizes + 1)
        {
            if(k <= x->lchild->sizes)
                x = x->lchild;
            else
            {
                k -= x->lchild->sizes + 1;
                x = x->rchild;
            }
            pushdown(x);
        }
        splay(x, y);
    }
    void print(SplayNode *x)
    {
        if(x == &nil) return;
        pushdown(x);
        print(x->lchild);
        printf("%d: %d %d %d %d\n", x->value, x->mins, x->parent->value, x->lchild->value, x->rchild->value);
        print(x->rchild);
    }
    void prints(SplayNode *x)
    {
        if(x == &nil) return;
        pushdown(x);
        if(x->value == INF) printf("INF : ");
        else printf("%d : ", x->value);
        if(x->lchild == &nil) printf("nil ");
        else
        {
            if(x->lchild->value == INF) printf("INF ");
            else printf("%d ", x->lchild->value);
        }
        if(x->rchild == &nil) printf("nil\n");
        else
        {
            if(x->rchild->value == INF) printf("INF\n");
            else printf("%d\n", x->rchild->value);
        }
        prints(x->lchild);
        prints(x->rchild);
    }

public:
    SplayTree()
    {
        nil.sizes = 0;
        nil.value = INF;
        nil.mins = INF;
        nil.lchild = &nil;
        nil.rchild = &nil;
        nil.parent = &nil;
    }
    //建树，数列1 ~ n-2
    void make(int arrays[], int n)
    {
        nodeNumber = 0;
        int mid = (n - 1) >> 1;
        root = newNode(&nil, arrays[mid]);
        root->lchild = make(0, mid - 1, root, arrays);
        root->rchild = make(mid + 1, n - 1, root, arrays);
        update(root);
    }
    //区间加和
    void ADD(int x, int y, int D)
    {
        finds(x, &nil);
        finds(y + 2, root);
        root->rchild->lchild->lazy += D;
    }
    //区间反转
    void REVERSE(int x, int y)
    {
        finds(x, &nil);
        finds(y + 2, root);
        root->rchild->lchild->isReverse ^= true;
    }
    //区间循环位移T次
    void REVOLVE(int x, int y, int T)
    {
        int len = y - x + 1;
        T = ((T % len) + len) % len;
        if(T)
        {
            finds(y - T + 1, &nil);
            finds(y + 2, root);
            SplayNode *d = root->rchild->lchild;
            root->rchild->lchild = &nil;
            finds(x, &nil);
            finds(x + 1, root);
            root->rchild->lchild = d;
            d->parent = root->rchild;
        }
    }
    //x位置后面插上P
    void INSERT(int x, int P)
    {
        finds(x + 1, &nil);
        finds(x + 2, root);
        root->rchild->lchild = newNode(root->rchild, P);
    }
    //删除x位置
    void DELETE(int x)
    {
        finds(x, &nil);
        finds(x + 2, root);
        root->rchild->lchild = &nil;
    }
    //区间最小值
    int MIN(int x, int y)
    {
        finds(x, &nil);
        finds(y + 2, root);
        pushdown(root->rchild->lchild);
        return root->rchild->lchild->mins;
    }
    void print()
    {
        printf("Splay Linear: \n");
        print(root);
        printf("\n");
    }
    void prints()
    {
        printf("Splay Structure: \n");
        prints(root);
        printf("\n");
    }
} spT;
int main()
{
    a[0]=a[n+1]=INF;
    spT.make(a,n+2);
    return 0;
}

///主席树
//可持续化记录历史，就是建立历史上所有更新的线段树
//对于每次询问从T[l]和T[r+1]同时开始查找，差值为区间内结果
///静态区间k大
#define N 112345
#define M N*3
int n,q,m,tot;
int a[N], t[N];
int T[N], lson[M], rson[M], c[M];
void initHash()
{
    for(int i = 1; i <= n; i++)
        t[i] = a[i];
    sort(t+1,t+1+n);
    m = unique(t+1,t+1+n)-t-1;
}
int build(int l,int r)
{
    int root = tot++;
    c[root] = 0;
    if(l != r)
    {
        int mid = (l+r)>>1;
        lson[root] = build(l,mid);
        rson[root] = build(mid+1,r);
    }
    return root;
}
int hashs(int x)
{
    return lower_bound(t+1,t+1+m,x) - t;
}
int update(int root,int pos,int val)
{
    int newroot = tot++, tmp = newroot;
    c[newroot] = c[root] + val;
    int l = 1, r = m;
    while(l < r)
    {
        int mid = (l+r)>>1;
        if(pos <= mid)
        {
            lson[newroot] = tot++;
            rson[newroot] = rson[root];
            newroot = lson[newroot];
            root = lson[root];
            r = mid;
        }
        else
        {
            rson[newroot] = tot++;
            lson[newroot] = lson[root];
            newroot = rson[newroot];
            root = rson[root];
            l = mid+1;
        }
        c[newroot] = c[root] + val;
    }
    return tmp;
}
int query(int leftRoot,int rightRoot,int k)
{
    int l = 1, r = m;
    while( l < r)
    {
        int mid = (l+r)>>1;
        int t = c[lson[leftRoot]]-c[lson[rightRoot]];
        if( t >= k )
        {
            r = mid;
            leftRoot = lson[leftRoot];
            rightRoot = lson[rightRoot];
        }
        else
        {
            l = mid + 1;
            k -= t;
            leftRoot = rson[leftRoot];
            rightRoot = rson[rightRoot];
        }
    }
    return l;
}
int main()
{
    while(scanf("%d%d",&n,&q) == 2)
    {
        tot = 0;
        for(int i = 1; i <= n; i++)
            scanf("%d",&a[i]);
        initHash();
        T[n+1] = build(1,m);
        for(int i = n; i ; i--)
        {
            int pos = hashs(a[i]);
            T[i] = update(T[i+1],pos,1);
        }
        while(q--)
        {
            int l,r,k;
            scanf("%d%d%d",&l,&r,&k);
            printf("%d\n",t[query(T[l],T[r+1],k)]);
        }
    }
    return 0;
}

///动态区间k大
#define N 60010
#define M 2500010
int n,q,m,tot;
int a[N], t[N];
int T[N], lson[M], rson[M],c[M];
int S[N];
struct Query
{
    int kind;
    int l,r,k;
} query[10010];
void initHash(int k)
{
    sort(t,t+k);
    m = unique(t,t+k) - t;
}
int hashs(int x)
{
    return lower_bound(t,t+m,x)-t;
}
int build(int l,int r)
{
    int root = tot++;
    c[root] = 0;
    if(l != r)
    {
        int mid = (l+r)/2;
        lson[root] = build(l,mid);
        rson[root] = build(mid+1,r);
    }
    return root;
}
int Insert(int root,int pos,int val)
{
    int newroot = tot++, tmp = newroot;
    int l = 0, r = m-1;
    c[newroot] = c[root] + val;
    while(l < r)
    {
        int mid = (l+r)>>1;
        if(pos <= mid)
        {
            lson[newroot] = tot++;
            rson[newroot] = rson[root];
            newroot = lson[newroot];
            root = lson[root];
            r = mid;
        }
        else
        {
            rson[newroot] = tot++;
            lson[newroot] = lson[root];
            newroot = rson[newroot];
            root = rson[root];
            l = mid+1;
        }
        c[newroot] = c[root] + val;
    }
    return tmp;
}
int lowbit(int x)
{
    return x&(-x);
}
int use[N];
void add(int x,int pos,int val)
{
    while(x <= n)
    {
        S[x] = Insert(S[x],pos,val);
        x += lowbit(x);
    }
}
int sum(int x)
{
    int ret = 0;
    while(x > 0)
    {
        ret += c[lson[use[x]]];
        x -= lowbit(x);
    }
    return ret;
}
int Query(int left,int right,int k)
{
    int left_root = T[left-1];
    int right_root = T[right];
    int l = 0, r = m-1;
    for(int i = left-1; i; i -= lowbit(i)) use[i] = S[i];
    for(int i = right; i ; i -= lowbit(i)) use[i] = S[i];
    while(l < r)
    {
        int mid = (l+r)/2;
        int tmp = sum(right) - sum(left-1) + c[lson[right_root]] - c[lson[left_root]];
        if(tmp >= k)
        {
            r = mid;
            for(int i = left-1; i ; i -= lowbit(i))
                use[i] = lson[use[i]];
            for(int i = right; i; i -= lowbit(i))
                use[i] = lson[use[i]];
            left_root = lson[left_root];
            right_root = lson[right_root];
        }
        else
        {
            l = mid+1;
            k -= tmp;
            for(int i = left-1; i; i -= lowbit(i))
                use[i] = rson[use[i]];
            for(int i = right; i ; i -= lowbit(i))
                use[i] = rson[use[i]];
            left_root = rson[left_root];
            right_root = rson[right_root];
        }
    }
    return l;
}
void Modify(int x,int p,int d)
{
    while(x <= n)
    {
        S[x] = Insert(S[x],p,d);
        x += lowbit(x);
    }
}
int main()
{
    int cas;
    scanf("%d",&cas);
    while(cas--)
    {
        scanf("%d%d",&n,&q);
        tot = 0;
        m = 0;
        for(int i = 1; i <= n; i++)
        {
            scanf("%d",&a[i]);
            t[m++] = a[i];
        }
        char op[10];
        for(int i = 0; i < q; i++)
        {
            scanf("%s",op);
            if(op[0] == 'Q')
            {
                query[i].kind = 0;
                scanf("%d%d%d",&query[i].l,&query[i].r,&query[i].k);
            }
            else
            {
                query[i].kind = 1;
                scanf("%d%d",&query[i].l,&query[i].r);
                t[m++] = query[i].r;
            }
        }
        initHash(m);
        T[0] = build(0,m-1);
        for(int i = 1; i <= n; i++)
            T[i] = Insert(T[i-1],hashs(a[i]),1);
        for(int i = 1; i <= n; i++)
            S[i] = T[0];
        for(int i = 0; i < q; i++)
        {
            if(query[i].kind == 0)
                printf("%d\n",t[Query(query[i].l,query[i].r,query[i].k)]);
            else
            {
                Modify(query[i].l,hashs(a[query[i].l]),-1);
                Modify(query[i].l,hashs(query[i].r),1);
                a[query[i].l] = query[i].r;
            }
        }
    }
    return 0;
}

///区间不相同数数量
#define N 30010
#define M 100*N
int n,q,tot;
int a[N];
int T[N],lson[M],rson[M],c[M];
int build(int l,int r)
{
    int root = tot++;
    c[root] = 0;
    if(l != r)
    {
        int mid = (l+r)>>1;
        lson[root] = build(l,mid);
        rson[root] = build(mid+1,r);
    }
    return root;
}
int update(int root,int pos,int val)
{
    int newroot = tot++, tmp = newroot;
    c[newroot] = c[root] + val;
    int l = 1, r = n;
    while(l < r)
    {
        int mid = (l+r)>>1;
        if(pos <= mid)
        {
            lson[newroot] = tot++;
            rson[newroot] = rson[root];
            newroot = lson[newroot];
            root = lson[root];
            r = mid;
        }
        else
        {
            rson[newroot] = tot++;
            lson[newroot] = lson[root];
            newroot = rson[newroot];
            root = rson[root];
            l = mid+1;
        }
        c[newroot] = c[root] + val;
    }
    return tmp;
}
int query(int root,int pos)
{
    int ret = 0;
    int l = 1, r = n;
    while(pos < r)
    {
        int mid = (l+r)>>1;
        if(pos <= mid)
        {
            r = mid;
            root = lson[root];
        }
        else
        {
            ret += c[lson[root]];
            root = rson[root];
            l = mid+1;
        }
    }
    return ret + c[root];
}
int main()
{
    while(scanf("%d",&n) == 1)
    {
        tot = 0;
        for(int i = 1; i <= n; i++)
            scanf("%d",&a[i]);
        T[n+1] = build(1,n);
        map<int,int>mp;
        for(int i = n; i>= 1; i--)
        {
            if(mp.find(a[i]) == mp.end())
            {
                T[i] = update(T[i+1],i,1);
            }
            else
            {
                int tmp = update(T[i+1],mp[a[i]],-1);
                T[i] = update(tmp,i,1);
            }
            mp[a[i]] = i;
        }
        scanf("%d",&q);
        while(q--)
        {
            int l,r;
            scanf("%d%d",&l,&r);
            printf("%d\n",query(T[l],r));
        }
    }
    return 0;
}

///树上路径点权第k大
#define N 200010
#define M 40*N
//主席树:
int n,q,m,TOT;
int a[N], t[N];
int T[N], lson[M], rson[M], c[M];
void Init_hash()
{
    for(int i = 1; i <= n; i++)
        t[i] = a[i];
    sort(t+1,t+1+n);
    m = unique(t+1,t+n+1)-t-1;
}
int build(int l,int r)
{
    int root = TOT++;
    c[root] = 0;
    if(l != r)
    {
        int mid = (l+r)>>1;
        lson[root] = build(l,mid);
        rson[root] = build(mid+1,r);
    }
    return root;
}
int hash(int x)
{
    return lower_bound(t+1,t+1+m,x) - t;
}
int update(int root,int pos,int val)
{
    int newroot = TOT++, tmp = newroot;
    c[newroot] = c[root] + val;
    int l = 1, r = m;
    while( l < r)
    {
        int mid = (l+r)>>1;
        if(pos <= mid)
        {
            lson[newroot] = TOT++;
            rson[newroot] = rson[root];
            newroot = lson[newroot];
            root = lson[root];
            r = mid;
        }
        else
        {
            rson[newroot] = TOT++;
            lson[newroot] = lson[root];
            newroot = rson[newroot];
            root = rson[root];
            l = mid+1;
        }
        c[newroot] = c[root] + val;
    }
    return tmp;
}
int query(int left_root,int right_root,int LCA,int k)
{
    int lca_root = T[LCA];
    int pos = hash(a[LCA]);
    int l = 1, r = m;
    while(l < r)
    {
        int mid = (l+r)>>1;
        int tmp = c[lson[left_root]] + c[lson[right_root]] - 2*c[lson[lca_root]] + (pos >= l && pos <= mid);
        if(tmp >= k)
        {
            left_root = lson[left_root];
            right_root = lson[right_root];
            lca_root = lson[lca_root];
            r = mid;
        }
        else
        {
            k -= tmp;
            left_root = rson[left_root];
            right_root = rson[right_root];
            lca_root = rson[lca_root];
            l = mid + 1;
        }
    }
    return l;
}
//LCA部分:
//rmq数组，就是欧拉序列对应的深度序列
int rmq[2*N];
struct ST
{
    int mm[2*N];
    int dp[2*N][20];
    void init(int n)
    {
        mm[0] = -1;
        for(int i = 1; i <= n; i++)
        {
            mm[i] = ((i&(i-1)) == 0)?mm[i-1]+1:mm[i-1];
            dp[i][0] = i;
        }
        for(int j = 1; j <= mm[n]; j++)
            for(int i = 1; i + (1<<j) - 1 <= n; i++)
                dp[i][j] = rmq[dp[i][j-1]] < rmq[dp[i+(1<<(j-1))][j-1]]?dp[i][j-1]:dp[i+(1<<(j-1))][j-1];
    }
    int query(int a,int b)
    {
        if(a > b)swap(a,b);
        int k = mm[b-a+1];
        return rmq[dp[a][k]] <= rmq[dp[b-(1<<k)+1][k]]?dp[a][k]:dp[b-(1<<k)+1][k];
    }
};
struct Edge
{
    int to,next;
};
Edge edge[N*2];
int tot,head[N];
int F[N*2];//欧拉序列，就是dfs遍历的顺序，长度为2*n-1,下标从1开始
int P[N];//P[i]表示点i在F中第一次出现的位置
int cnt;
ST st;
void init()
{
    tot = 0;
    memset(head,-1,sizeof(head));
}
void addedge(int u,int v)
{
    edge[tot].to = v;
    edge[tot].next = head[u];
    head[u] = tot++;
}
void dfs(int u,int pre,int dep)
{
    F[++cnt] = u;
    rmq[cnt] = dep;
    P[u] = cnt;
    for(int i = head[u]; i != -1; i = edge[i].next)
    {
        int v = edge[i].to;
        if(v == pre)continue;
        dfs(v,u,dep+1);
        F[++cnt] = u;
        rmq[cnt] = dep;
    }
}
void LCA_init(int root,int node_num)
{
    cnt = 0;
    dfs(root,root,0);
    st.init(2*node_num-1);
}
int query_lca(int u,int v)
{
    return F[st.query(P[u],P[v])];
}
void dfs_build(int u,int pre)
{
    int pos = hash(a[u]);
    T[u] = update(T[pre],pos,1);
    for(int i = head[u]; i != -1; i = edge[i].next)
    {
        int v = edge[i].to;
        if(v == pre)continue;
        dfs_build(v,u);
    }
}
int main()
{
    while(scanf("%d%d",&n,&q) == 2)
    {
        for(int i = 1; i <= n; i++)
            scanf("%d",&a[i]);
        Init_hash();
        init();
        TOT = 0;
        int u,v;
        for(int i = 1; i < n; i++)
        {
            scanf("%d%d",&u,&v);
            addedge(u,v);
            addedge(v,u);
        }
        LCA_init(1,n);
        T[n+1] = build(1,m);
        dfs_build(1,n+1);
        int k;
        while(q--)
        {
            scanf("%d%d%d",&u,&v,&k);
            printf("%d\n",t[query(T[u],T[v],query_lca(u,v),k)]);
        }
        return 0;
    }
    return 0;
}

///哈希表
///ELFhash（字符串哈希函数）
const int MAXhash = 149993;
typedef struct
{
    char e[11];//输出字符
    char f[11];//输入字符
    int next;
}Entry;
Entry entry[MAXhash];
int i = 1;      //词典总条数
int hashIndex[MAXhash];
int ELFHash(char *key)
{
    unsigned long h=0;
    while(*key)
    {
        //h左移4位，把当前字符ASCII存入h低四位。
        h=(h<<4)+(*key++);
        unsigned long g=h&0Xf0000000L;
        //如果最高的四位不为0，则说明字符多余7个，现在正在存第7个字符，如果不处理，再加下一个字符时，第一个字符会被移出，因此要有如下处理。
        //该处理，如果最高位为0，就会仅仅影响5-8位，否则会影响5-31位，因为C语言使用的算数移位
        //因为1-4位刚刚存储了新加入到字符，所以不能>>28
        if(g) h^=g>>24;
        //上面这行代码并不会对g有影响，本身g和h的高4位相同，下面这行代码&~即对28-31(高4位)位清零。
        h&=~g;
    }
    //返回一个符号位为0的数，即丢弃最高位，以免函数外产生影响。(我们可以考虑，如果只有字符，符号位不可能为负)
//    return h&0x7FFFFFFF;
    return h%MAXhash;
}
void finds(char f[])
{
    int hashs = ELFHash(f);
    for(int k = hashIndex[hashs]; k; k = entry[k].next)
    {
        if(strcmp(f, entry[k].f) == 0)
        {
            printf("%s\n",entry[k].e);
            return;
        }
    }
    printf("eh\n");
}
int main()
{
    char str[22];
    while(gets(str))
    {
        if(str[0] == '\0')//判断空行，结束字典输入
            break;
        sscanf(str,"%s %s",entry[i].e,entry[i].f);//字典处理
        int hashs = ELFHash(entry[i].f);//elfhash处理字符串找key
        entry[i].next = hashIndex[hashs];//链表法处理冲突
        hashIndex[hashs] = i;
        i++;
    }
    while(gets(str))
        finds(str);//开始输入查找字符串
    return 0;
}
///莫队算法
///分块有序暴力枚举
long long n,m,sum,res,flag;
long long num[N],ans[N],col[N];
struct node
{
    long long l,r,id,pl;
    friend bool operator < (node a, node b)
    {
        if(a.pl == b.pl) return a.r < b.r;
        return a.pl < b.pl;
    }
}q[N];
//区间操作,区间不相同数个数
void updata(long long pos, int flag)
{
    res-=num[col[pos]];
    num[col[pos]]=flag;
    res+=num[col[pos]];
}
int main()
{
    long long  i,j,k,cas,T,t,x,y,z,l,r;
    while(scanf("%I64d%I64d",&n,&m)!=EOF)
    {
        memset(num,0,sizeof(num));
        for(i=1;i<=n;i++)
            scanf("%I64d",&col[i]);
        for(i=0;i<m;i++)
        {
            scanf("%I64d%I64d",&q[i].l,&q[i].r);
            q[i].id=i; q[i].pl=(q[i].l-1)/(ceil(sqrt(1.0*n)));
        }
        sort(q,q+m);
        l=1;r=res=0;
        for(i=0;i<m;i++)
        {
            for(j=min(l,q[i].l);j<max(l,q[i].l);j++)
                updata(j,(l>q[i].l?1:0));
            for(j=max(r,q[i].r);j>min(r,q[i].r);j--)
                updata(j,(r>q[i].r?0:1));
            r = q[i].r; l = q[i].l;
            ans[q[i].id] = res;
        }
        for(i=0;i<m;i++)
            printf("%I64d\n",ans[i]);
    }
    return 0;
}

///字符串！！！！！
///字符串最小表示法
//循环串字典序最小表示，O(n)
int minString(char *s)
{
    int i=0,j=1,k=0;
    int len=strlen(s);
    while(i<len&&j<len&&k<len)
    {
        if(s[(i+k)%len]==s[(j+k)%len])k++;
        else
        {
            if(s[(i+k)%len]>s[(j+k)%len])i=i+k+1;
            else j=j+k+1;
            if(i==j)j++;
            k=0;
        }
    }
    return i<j?i:j;
}

///Manacher最长回文子串
//O(n)
int p[N*2];
int Manacher(char *str)
{
    int res=0,ans=0,len=strlen(str);
    for(int i=len;i>=0;i--)
    {
        str[i+i+2] = str[i];
        str[i+i+1] = '#';
    }
    str[0] = '*';
    for(int i=2;i<2*len+1;i++)
    {
        if(p[res]+res > i)
            p[i] = min(p[2*res-i],p[res]+res-i);
        else
            p[i] = 1;
        while(str[i-p[i]] == str[i+p[i]])
            p[i]++;
        if(res+p[res]<i+p[i])res=i;
        if(ans<p[i])ans=p[i];
    }
    return ans-1;
}

///KMP
//O(n+m),n%(n-next[n])==0得到循环节
void getNext(char *pre, int len, int *next)
{
    int i = 0,j = -1;
    next[0] = -1;
    while(i < len)
    {
        if(j == -1 || pre[i] == pre[j])
        {
            i++,j++;
            //next[i] = j;
            if(pre[i]!=pre[j]) next[i] = j;
            else next[i] = next [j];
        }
        else   j = next[j];
    }
}
int kmp(char *str, char *pre)
{
    int lenp = strlen(pre), lens = strlen(str);
    int next[lenp+1];
    getNext(pre, lenp, next);
    int i,j; i=j=0;
    while(i < lens && j < lenp)
    {
        if(j == -1 || str[i] == pre[j]) i++, j++;
        else j = next[j];
//        if(j == lenp) sum++;//记数(去掉while()中的j<lenp);
    }
    if(j >= lenp) return i - lenp;
    else return -1;
}

///扩展kmp
//next[i]:pre[i...lenp-1]与pre[0...lenp-1]的最长公共前缀
//extend[i]:str[i...lens-1]与pre[0...lenp-1]的最长公共前缀
void getEnext(char pre[],int len,int next[])
{
    next[0]=len;
    int j=0;
    while(j+1<len && pre[j]==pre[j+1])j++;
    next[1]=j;
    int k=1;
    for(int i=2;i<len;i++)
    {
        int p=next[k]+k-1;
        int L=next[i-k];
        if(i+L<p+1)next[i]=L;
        else
        {
            j=max(0,p-i+1);
            while(i+j<len && pre[i+j]==pre[j])j++;
            next[i]=j;
            k=i;
        }
    }
}
void ekmp(char pre[],char str[],int next[],int extend[])
{
    int lenp=strlen(pre),lens=strlen(str);
    getEnext(pre,lenp,next);
    int j=0;
    while(j<lens && j<lenp && pre[j]==str[j])j++;
    extend[0]=j;
    int k=0;
    for(int i=1;i<lens;i++)
    {
        int p=extend[k]+k-1;
        int L=next[i-k];
        if(i+L<p+1)extend[i]=L;
        else
        {
            j=max(0,p-i+1);
            while(i+j<lens && j<lenp && str[i+j]==pre[j])j++;
            extend[i]=j;
            k=i;
        }
    }
}

///AC自动机
#define N 100010
#define M 201
#define ZIFU 128
char str[N][M];
int num[N],n,m;
struct Trie
{
    int next[N][ZIFU],fail[N];
    vector<int>ends[N];
    int root,L;
    int idx(char c)
    {
        return c;
    }
    int newnode()
    {
        for(int i = 0;i < ZIFU;i++)
            next[L][i] = -1;
        ends[L++].clear();
        return L-1;
    }
    void init()
    {
        L = 0;
        root = newnode();
    }
    void inserts(char s[],int id)
    {
        int len = strlen(s);
        int now = root;
        for(int i = 0;i < len;i++)
        {
            if(next[now][idx(s[i])] == -1)
                next[now][idx(s[i])] = newnode();
            now = next[now][idx(s[i])];
        }
        ends[now].push_back(id);
    }
    void build()
    {
        queue<int>Q;
        fail[root] = root;
        for(int i = 0;i < ZIFU;i++)
            if(next[root][i] == -1)
                next[root][i] = root;
            else
            {
                fail[next[root][i]] = root;
                Q.push(next[root][i]);
            }
        while(!Q.empty())
        {
            int now = Q.front();
            Q.pop();
            for(int i = 0;i < ZIFU;i++)
                if(next[now][i] == -1)
                    next[now][i]=next[fail[now]][i];
                else
                {
                    fail[next[now][i]]=next[fail[now]][i];
                    Q.push(next[now][i]);
                }
        }
    }
    int query(char buf[])
    {
        int sum=0;
        for(int i = 0;i < n;i++)
            num[i] = 0;
        int len=strlen(buf);
        int now=root;
        for(int i=0;i<len;i++)
        {
            now=next[now][idx(buf[i])];
            int temp = now;
            while( temp != root )
            {
                for(int j=0;j<ends[temp].size();j++)
                    num[ends[temp][j]]++,sum++;
                temp = fail[temp];
            }
        }
        return sum;
    }

};
char buf[1000010];
Trie ac;
int main()
{
    scanf("%d",&n);
    ac.init();
    for(int i=0;i<n;i++)
    {
        scanf("%s",&str[i]);
        ac.inserts(str[i],i);
    }
    ac.build();
    scanf("%s",&buf);
    printf("%d\n",ac.query(buf));
    return 0;
}

///后缀数组
<<<<<<< HEAD
=======
//处理子串重复或不相同的问题
>>>>>>> 6d46811724b7a8a08edc853e39078c7bc70df43f
//seq[]原始串，sa[i]排在i位的后缀子串的起始位置，ranks[i]i位开始的后缀子串的排序位置，height[i]排在i与i-1位的后缀子串的相同前缀长度。
///DA倍增算法 O(n*logn)
int seq[N], sa[N], ranks[N], height[N];
int wwa[N], wwb[N], wws[N], wwv[N];
bool cmp(int r[], int a, int b, int l)
{
    return r[a] == r[b] && r[a+l] == r[b+l];
}
void da(int r[],int n, int m)
{
    int i, j, p, *x = wwa, *y = wwb;
    for (i = 0; i < m; ++i) wws[i] = 0;
    for (i = 0; i < n; ++i) wws[x[i]=r[i]]++;
    for (i = 1; i < m; ++i) wws[i] += wws[i-1];
    for (i = n-1; i >= 0; --i) sa[--wws[x[i]]] = i;
    for (j = 1, p = 1; p < n; j *= 2, m = p)
    {
        for (p = 0, i = n - j; i < n; ++i) y[p++] = i;
        for (i = 0; i < n; ++i) if (sa[i] >= j) y[p++] = sa[i] - j;
        for (i = 0; i < n; ++i) wwv[i] = x[y[i]];
        for (i = 0; i < m; ++i) wws[i] = 0;
        for (i = 0; i < n; ++i) wws[wwv[i]]++;
        for (i = 1; i < m; ++i) wws[i] += wws[i-1];
        for (i = n-1; i >= 0; --i) sa[--wws[wwv[i]]] = y[i];
        for (swap(x, y), p = 1, x[sa[0]] = 0, i = 1; i < n; ++i)
            x[sa[i]] = cmp(y, sa[i-1], sa[i], j) ? p-1 : p++;
    }
}
void calheight(int r[], int n)
{
    int i, j, k = 0;
    for (i = 1; i <= n; ++i) ranks[sa[i]] = i;
    for (i = 0; i < n; height[ranks[i++]] = k)
        for (k?k--:0, j = sa[ranks[i]-1]; r[i+k] == r[j+k]; k++);
}
void solve()
{
    //初始序0~n，字符数值1~m-1。
    seq[n]=0;
    da(seq,n+1,m);
    calheight(seq,n);
}
//求不重复最长相同子串
int longestString()
{
    solve();
    int i,x=1,y=n,z,res=0,flag;
    while(x<=y)
    {
        z=(x+y)/2;
        flag=false;
        for(i=2;i<=n&&!flag;i++)
        {
            int l,r;l=r=sa[i-1];
            while(height[i]>=z)l=min(l,sa[i]),r=max(r,sa[i]),i++;
            if(r-l>=z)flag=true;
        }
        if(flag)
            x=z+1,res=z;
        else
            y=z-1;
    }
    return res;
}

///DC3算法 O(N)
int seq[3*N], sa[3*N], ranks[3*N], height[3*N];
int wwa[3*N], wwb[3*N], wws[3*N], wwv[3*N];
int c0(int *r,int a,int b)
{return r[a]==r[b]&&r[a+1]==r[b+1]&&r[a+2]==r[b+2];}
int c12(int k,int *r,int a,int b)
{if(k==2) return r[a]<r[b]||r[a]==r[b]&&c12(1,r,a+1,b+1);
else return r[a]<r[b]||r[a]==r[b]&&wwv[a+1]<wwv[b+1];}
void sort(int *r,int *a,int *b,int n,int m)
{
    int i;
    for(i=0;i<n;i++) wwv[i]=r[a[i]];
    for(i=0;i<m;i++) wws[i]=0;
    for(i=0;i<n;i++) wws[wwv[i]]++;
    for(i=1;i<m;i++) wws[i]+=wws[i-1];
    for(i=n-1;i>=0;i--) b[--wws[wwv[i]]]=a[i];
    return;
}
void dc3(int *r,int *sa,int n,int m)
{
    int i,j,*rn=r+n,*san=sa+n,ta=0,tb=(n+1)/3,tbc=0,p;
    r[n]=r[n+1]=0;
    for(i=0;i<n;i++) if(i%3!=0) wwa[tbc++]=i;
    sort(r+2,wwa,wwb,tbc,m);
    sort(r+1,wwb,wwa,tbc,m);
    sort(r,wwa,wwb,tbc,m);
    for(p=1,rn[((wwb[0])/3+((wwb[0])%3==1?0:tb))]=0,i=1;i<tbc;i++)
    rn[((wwb[i])/3+((wwb[i])%3==1?0:tb))]=c0(r,wwb[i-1],wwb[i])?p-1:p++;
    if(p<tbc) dc3(rn,san,tbc,p);
    else for(i=0;i<tbc;i++) san[rn[i]]=i;
    for(i=0;i<tbc;i++) if(san[i]<tb) wwb[ta++]=san[i]*3;
    if(n%3==1) wwb[ta++]=n-1;
    sort(r,wwb,wwa,ta,m);
    for(i=0;i<tbc;i++) wwv[wwb[i]=((san[i])<tb?(san[i])*3+1:((san[i])-tb)*3+2)]=i;
    for(i=0,j=0,p=0;i<ta && j<tbc;p++)
    sa[p]=c12(wwb[j]%3,r,wwa[i],wwb[j])?wwa[i++]:wwb[j++];
    for(;i<ta;p++) sa[p]=wwa[i++];
    for(;j<tbc;p++) sa[p]=wwb[j++];
    return;
}
void calheight(int r[], int n)
{
    int i, j, k = 0;
    for (i = 1; i <= n; ++i) ranks[sa[i]] = i;
    for (i = 0; i < n; height[ranks[i++]] = k)
        for (k?k--:0, j = sa[ranks[i]-1]; r[i+k] == r[j+k]; k++);
}
void solve()
{
    //初始序0~n，字符数值1~m-1。
    seq[n]=0;
    dc3(seq,sa,n+1,m);
    calheight(seq,n);
}

///后缀自动机SAM
struct SAM
{
    struct Node
    {
        int ch[26];
        //失败指针,当前匹配位置的长度
        int f, len;
        void init()
        {
            f = -1, len = 0;
            memset(ch, 0xff, sizeof (ch));
        }
    };
    Node sn[N<<1];
    int idx, last;
    void init()
    {
        idx = last = 0;
        sn[idx++].init();
    }
    int newnode()
    {
        sn[idx].init();
        return idx++;
    }
    void add(int c)
    {
        int end = newnode();
        int tmp = last;
        sn[end].len = sn[last].len + 1;
        for ( ; tmp != -1 && sn[tmp].ch[c] == -1; tmp = sn[tmp].f)
            sn[tmp].ch[c] = end;
        if (tmp == -1) sn[end].f = 0;
        else
        {
            int nxt = sn[tmp].ch[c];
            if (sn[tmp].len + 1 == sn[nxt].len) sn[end].f = nxt;
            else
            {
                int np = newnode();
                sn[np] = sn[nxt];
                sn[np].len = sn[tmp].len + 1;
                sn[end].f = sn[nxt].f = np;
                for (; tmp != -1 && sn[tmp].ch[c] == nxt; tmp = sn[tmp].f)
                    sn[tmp].ch[c] = np;
            }
        }
        last = end;
    }
};
SAM sam;
//最长公共子串
int n,m,sum,res,flag;
char s[N];
int main()
{
    int i,j,k,kk,cas,T,t,x,y,z;
    while(scanf("%s",s)!=EOF)
    {
        sam.init();
        n=strlen(s);
        for(i=0;i<n;i++)
            sam.add(s[i]-'a');
        scanf("%s",s);
        m=strlen(s);
        res=t=x=0;
        for(i=0;i<m;i++)
        {
            y=s[i]-'a';
            if(sam.sn[x].ch[y]!=-1)
                t++,x=sam.sn[x].ch[y];
            else
            {
                for(;x!=-1&&sam.sn[x].ch[y]==-1;x=sam.sn[x].f);
                if(x==-1)t=x=0;
                else
                {
                    t=sam.sn[x].len+1;
                    x=sam.sn[x].ch[y];
                }
            }
            res=max(res,t);
        }
        printf("%d\n",res);
    }
    return 0;
}

///图论！！！！！！
///最小生成树
///prim
//O(n^2)
bool vis[N];
int g[N][N],dist[N];
int prim(int n)
{
    int minn,i,j,pos=1,res=0;
    memset(vis,false,sizeof(vis));
    for(i=1; i<=n; i++)
        dist[i]=g[pos][i];
    dist[pos]=0;vis[pos]=true;
    for(i=1; i<n; i++)
    {
        minn=INF;
        for(j=1; j<=n; j++)
            if(!vis[j]&&dist[j]<minn)
            {
                minn=dist[j];
                pos=j;
            }
        res+=minn;
        vis[pos]=true;
        //更新权值，与dijkstra不同的地方(起点为所有加入生成树的节点，找到树最近的点)
        for(j=1; j<=n; j++)
            if(!vis[j]&&dist[j]>g[pos][j])
                dist[j]=g[pos][j];
    }
    return res;
}

///kruskal
//O(ElogE)
int f[N];
int cnt;
struct node
{
    int u,v;
    int r;
    //升序排列
    friend bool operator < (node a, node b)
    {
        return a.r < b.r;
    }
}edge[M];
void inserts(int u,int v,int w)
{
    edge[cnt].u = u;
    edge[cnt].v = v;
    edge[cnt++].r = w;
}
void init(int n)
{
    cnt=0;
    for(int i=1;i<=n;i++)
        f[i] = i;
}
int getf(int u)
{
    if(f[u] != u)
        f[u] = getf(f[u]);
    return f[u];
}
bool unions(int x,int y)
{
    x=getf(x);
    y=getf(y);
    if(x!=y)
    {
        f[x] = y;
        return false;
    }
    return true;
}
int kruskal(int n,int m)
{
    sort(edge,edge+m);
    int i,k,res;
    i=k=res=0;
    while(k<n-1)
    {
        if(i == m)break;
        //并查集判断不形成环，则该边加入生成树中
        if(!unions(edge[i].u,edge[i].v))
        {
            k++;
            res+=edge[i].r;
        }
        i++;
    }
    return res;
}

///次小生成树
/*
* 次小生成树
* 求最小生成树时，用数组Max[i][j]来表示MST中i到j最大边权
* 求完后，直接枚举所有不在MST中的边，替换掉最大边权的边，更新答案
* 点的编号从0开始
*/
bool vis[N];
int lowc[N];
int pre[N];
int Max[N][N];//Max[i][j]表示在最小生成树中从i到j的路径中的最大边权
bool used[N][N];
int Prim(int cost[][N],int n)
{
    int ans=0;
    memset(vis,false,sizeof(vis));
    memset(Max,0,sizeof(Max));
    memset(used,false,sizeof(used));
    vis[0]=true;
    pre[0]=-1;
    for(int i=1; i<n; i++)
    {
        lowc[i]=cost[0][i];
        pre[i]=0;
    }
    lowc[0]=0;
    for(int i=1; i<n; i++)
    {
        int minc=INF;
        int p=-1;
        for(int j=0; j<n; j++)
            if(!vis[j]&&minc>lowc[j])
            {
                minc=lowc[j];
                p=j;
            }
        if(minc==INF)return -1;
        ans+=minc;
        vis[p]=true;
        used[p][pre[p]]=used[pre[p]][p]=true;
        for(int j=0; j<n; j++)
        {
            if(vis[j])Max[j][p]=Max[p][j]=max(Max[j][pre[p]],lowc[p]);
            if(!vis[j]&&lowc[j]>cost[p][j])
            {
                lowc[j]=cost[p][j];
                pre[j]=p;
            }
        }
    }
    return ans;
}

///单源最短距离
///Dijkstra
//邻接矩阵存图，权值非负，O(n^2), 下标从1开始
int g[N][N],dist[N];
bool vis[N];
int dijkstra(int n,int u,int v)
{
    int minn,i,j,pos;
    memset(vis,false,sizeof(vis));
    for(i=1; i<=n; i++)
        dist[i]=g[u][i];
    dist[u] = 0;  vis[u] = true;
    for(i=1; i<n; i++)
    {
        minn=INF;
        for(j=1; j<=n; j++)
            if(!vis[j]&&dist[j]<minn)
            {
                minn=dist[j];
                pos=j;
            }
        if(minn==INF)break;
        vis[pos]=true;
        //更新权值,与prim不同的地方(起点不变，更新最短距离)
        for(j=1; j<=n; j++)
            if(!vis[j]&&dist[j]>dist[pos]+g[pos][j])
                dist[j]=dist[pos]+g[pos][j];
    }
    return dist[v];
}

///堆优化Dijkstra
//O(ElogE)
struct node
{
    int v;
    int c;
    node(int _v=0,int _c=0):v(_v),c(_c) {}
    bool operator <(const node &r)const
    {
        return c>r.c;
    }
};
struct edge
{
    int v;int cost;
    edge(int _v=0,int _cost=0):v(_v),cost(_cost) {}
};
vector<edge>E[N];
bool vis[N];
int dist[N],f[N];
void addedge(int u,int v,int w)
{
    E[u].push_back(edge(v,w));
}
//点的编号从1开始
void Dijkstra(int n,int start)
{
    memset(vis,false,sizeof(vis));
    for(int i=1;i<=n;i++)dist[i]=INF;
    for(int i=1;i<=n;i++)f[i]=i;
    priority_queue<node>q;
    while(!q.empty())q.pop();
    dist[start]=0.0;
    q.push(node(start,0.0));
    node t;
    while(!q.empty())
    {
        t=q.top();
        q.pop();
        int u=t.v;
        if(vis[u])continue;
        vis[u]=true;
        for(int i=0; i<E[u].size(); i++)
        {
            int v=E[t.v][i].v;
            int cost=E[u][i].cost;
            if(!vis[v]&&dist[v]>dist[u]+cost)
            {
                dist[v]=dist[u]+cost;
                f[v]=u;
                q.push(node(v,dist[v]));
            }
        }
    }
}
void putRoad(int x,int y)
{
    stack<int>st;
    while(y!=x)
    {
        st.push(y);
        y=f[y];
    }
    printf("%d",x);
    while(!st.empty())
    {
        printf(" %d",st.top());
        st.pop();
    }
    printf("\n");
}

///Bellman-Ford
//能判负权边，直接存边，O(n*m);
struct node
{
    int x,y;
    int r;
}edge[M];
int pre[N],dist[N];
bool relax(int x,int y,int r)
{
    if(dist[x]>dist[y]+r)
    {
        dist[x]=dist[y]+r;
        pre[y]=x;
        return true;
    }
    return false;
}
int bellman(int n,int m,int u)
{
    bool flag;
    memset(dist,0x3f,sizeof(dist));
    memset(pre,-1,sizeof(pre));
    dist[u]=0;
    for(int i=1;i<n;i++)
    {
        // 优化，后继节点都已经遍历过无需再松弛
        flag = false;
        for(int j=0;j<m;j++)
            if(relax(edge[j].x,edge[j].y,edge[j].r))
                flag=true;
        if(!flag)
            break;
    }
    // 判负圈
    for(int i=0;i<m;i++)
        if(relax(edge[i].x,edge[i].y,edge[i].r))
            return 0;
    return 1;
}

///SPFA
struct node
{
     int next=-1;
     int to,w;
}edge[2*M];
int head[N],dist[N];
int cnt;
void inserts(int u, int v, int w)
{
    edge[cnt].w = w;
    edge[cnt].to = v;
    edge[cnt].next = head[u];
    head[u] = cnt++;
    edge[cnt].w = w;
    edge[cnt].to = u;
    edge[cnt].next = head[v];
    head[v] = cnt++;
}
void init()
{
    memset(head,-1,sizeof(head));
    cnt=0;
}
void SPFA(int u)
{
    int x,y,w;
    bool vis[N];
    memset(dist,0x3f,sizeof(dist));
    dist[u] = 0;
    memset(vis,false,sizeof(vis));
    queue<int> q;
    while(!q.empty())q.pop();
    q.push(u);
    while(!q.empty())
    {
        x = q.front();  q.pop();
        vis[x] = false;
        for(int i=head[x];i!=-1;i=edge[i].next)
        {
            w = edge[i].w;  y = edge[i].to;
            if(dist[x]+w<dist[y])
            {
                dist[y] = dist[x]+w;
                if(!vis[y])
                    q.push(y);
                vis[y] = true;
            }
        }
    }
}

///全图最短路
///Floyd
int g[N][N];
void init(int n)
{
    memset(g,0x3f,sizeof(g));
    for(int i=1;i<=n;i++)
        g[i][i] = 0;
}
void floyd(int n)
{
    for(int k=1;k<=n;k++)
        for(int i=1;i<=n;i++)
            for(int j=1;j<=n;j++)
                g[i][j] = min(g[i][j],g[i][k]+g[k][j]);
}

///欧拉回路
///fleury
int n,m;
int g[N][N];
stack<int> st;
void dfs(int u)
{
    for(int i=1;i<=n;i++)
        if(g[u][i])
        {
            g[u][i]=g[i][u]=0;
            st.push(i);
            dfs(i);
            break;
        }
}
void fleury(int u)
{
    while(!st.empty())st.pop();
    st.push(u);
    while(!st.empty())
    {
        int flag=1;
        for(int i=1;i<=n;i++)
            if(g[st.top()][i])
            {
                flag=0;
                break;
            }
        if(flag)
        {
            printf("%d ",st.top());
            st.pop();
        }
        else
            dfs(st.top());
    }
}
void solve()
{
    int res=0, u=1;
    for(int i=1;i<=n;i++)
    {
        int sum=0;
        for(int j=1;j<=n;j++)
            sum+=g[i][j];
        if(sum&1)u=i,res++;
    }
    if(res==0 || res==2)fleury(u);
    else printf("Sorry! no euler path.");
    printf("\n");
}

///网络流
///最大流
///EdmondsKarp
//O(V*E^2)
//标记在这条路径上当前节点的前驱,同时标记该节点是否在队列中
int pre[N];
//记录残留网络的容量
int g[N][N];
int BFS(int src,int des,int n)
{
    int i,j;
    int flow[N];
    queue<int> q;
    while(!q.empty())
        q.pop();
    memset(pre,-1,sizeof(pre));
    pre[src] = src;
    flow[src] = INF;
    q.push(src);
    while(!q.empty())
    {
        int index = q.front();
        q.pop();
        if(index == des)
            break;
        for(i=1;i<=n;i++)
            if(i!=src && pre[i]==-1 && g[index][i]>0)
            {
                pre[i] = index;
                flow[i] = min(g[index][i],flow[index]);
                q.push(i);
            }
    }
    if(pre[des] == -1)
        return -1;
    else
        return flow[des];
}
int maxFlow(int src,int des,int n)
{
    int increasement = 0;
    int sumflow = 0;
    while((increasement = BFS(src,des,n)) != -1)
    {
        int k = des;
        while(k != src)
        {
            int last = pre[k];
            g[last][k] -= increasement;
            g[k][last] += increasement;
            k = last;
        }
        sumflow += increasement;
    }
    return sumflow;
}

///dinic
//O(V^2*E)
struct edge{int x,y,next; int c;}e[M];
int tot,head[N],ps[N],dep[N];
void init()
{
    memset(head,-1,sizeof(head));
    tot=0;
}
void addedge(int x,int y,int c)
{
    e[tot].x=x;e[tot].y=y;e[tot].c=c;
    e[tot].next=head[x];head[x]=tot++;
    e[tot].x=y;e[tot].y=x;e[tot].c=0;
    e[tot].next=head[y];head[y]=tot++;
}
int flow(int src,int des,int n)
{
    int tr,res=0;
    int i,j,k,l,r,top;
    while(1)
    {
        //分层标号
        memset(dep,-1,sizeof(dep));
        for(l=dep[ps[0]=src]=0,r=1;l!=r;)
        {
            for(i=ps[l++],j=head[i];j!=-1;j=e[j].next)
                if(e[j].c&&-1==dep[k=e[j].y])
                {
                    dep[k]=dep[i]+1;ps[r++]=k;
                    if(k==des){ l=r; break; }
                }
        }
        if(dep[des]==-1)break;
        //深搜找增广路
        for(i=src,top=0;;)
        {
            if(i==des)
            {
                for(k=0,tr=INF;k<top;++k)
                    if(e[ps[k]].c<tr)tr=e[ps[l=k]].c;
				for(k=0;k<top;++k)
					e[ps[k]].c-=tr,e[ps[k]^1].c+=tr;
				res+=tr;i=e[ps[top=l]].x;
            }
            for(j=head[i];j!=-1;j=e[j].next)
                if(e[j].c&&dep[i]+1==dep[e[j].y])break;
            if(j!=-1)
                ps[top++]=j,i=e[j].y;
            else
            {
                if(!top)break;
                dep[i]=-1;i=e[ps[--top]].x;
            }
        }
    }
    return res;
}

///SAP
int pre[N];
int g[N][N];
int gap[N],dis[N],cur[N];
//0~sum-1
int sap(int src,int des,int sum)
{
    memset(cur,0,sizeof(cur));
    memset(dis,0,sizeof(dis));
    memset(gap,0,sizeof(gap));
    int u=pre[src]=src,maxflow=0,aug=-1;
    gap[0]=sum;
    while(dis[src]<sum)
    {
        int flag=1;
        while(flag)
        {
            flag=0;
            for(int v=cur[u]; v<sum; v++)
            {
                if(g[u][v] && dis[u]==dis[v]+1)
                {
                    if(aug==-1 || aug>g[u][v])aug=g[u][v];
                    pre[v]=u;
                    u=cur[u]=v;
                    if(v==des)
                    {
                        maxflow+=aug;
                        for(u=pre[u]; v!=src; v=u,u=pre[u])
                        {
                            g[u][v]-=aug;
                            g[v][u]+=aug;
                        }
                        aug=-1;
                    }
                    flag=1;
                }
                if(flag)break;
            }
        }
        int mindis=sum-1;
        for(int v=0; v<sum; v++)
            if(g[u][v]&&mindis>dis[v])
            {
                cur[u]=v;
                mindis=dis[v];
            }
        if((--gap[dis[u]])==0)break;
        gap[dis[u]=mindis+1]++;
        u=pre[u];
    }
    return maxflow;
}

///ISAP
struct Edge
{
    int to,next,cap,flow;
} edge[M];
int tol;
int head[N];
int gap[N],dep[N],cur[N];
void init()
{
    tol = 0;
    memset(head,-1,sizeof(head));
}
void addedge(int u,int v,int w,int rw = 0)
{
    edge[tol].to = v;
    edge[tol].cap = w;
    edge[tol].flow = 0;
    edge[tol].next = head[u];
    head[u] = tol++;
    edge[tol].to = u;
    edge[tol].cap = rw;
    edge[tol].flow = 0;
    edge[tol].next = head[v];
    head[v] = tol++;
}
int Q[N];
void BFS(int start,int ends)
{
    memset(dep,-1,sizeof(dep));
    memset(gap,0,sizeof(gap));
    gap[0] = 1;
    int fronts = 0, rear = 0;
    dep[ends] = 0;
    Q[rear++] = ends;
    while(fronts != rear)
    {
        int u = Q[fronts++];
        for(int i = head[u]; i != -1; i = edge[i].next)
        {
            int v = edge[i].to;
            if(dep[v] != -1)continue;
            Q[rear++] = v;
            dep[v] = dep[u] + 1;
            gap[dep[v]]++;
        }
    }
}
int S[N];
int sap(int start,int ends,int sum)
{
    BFS(start,ends);
    memcpy(cur,head,sizeof(head));
    int top = 0;
    int u = start;
    int ans = 0;
    while(dep[start] < sum)
    {
        if(u == ends)
        {
            int Min = INF;
            int inser;
            for(int i = 0; i < top; i++)
                if(Min > edge[S[i]].cap - edge[S[i]].flow)
                {
                    Min = edge[S[i]].cap - edge[S[i]].flow;
                    inser = i;
                }
            for(int i = 0; i < top; i++)
            {
                edge[S[i]].flow += Min;
                edge[S[i]^1].flow -= Min;
            }
            ans += Min;
            top = inser;
            u = edge[S[top]^1].to;
            continue;
        }
        bool flag = false;
        int v;
        for(int i = cur[u]; i != -1; i = edge[i].next)
        {
            v = edge[i].to;
            if(edge[i].cap - edge[i].flow && dep[v]+1 == dep[u])
            {
                flag = true;
                cur[u] = i;
                break;
            }
        }
        if(flag)
        {
            S[top++] = cur[u];
            u = v;
            continue;
        }
        int Min = N;
        for(int i = head[u]; i != -1; i = edge[i].next)
            if(edge[i].cap - edge[i].flow && dep[edge[i].to] < Min)
            {
                Min = dep[edge[i].to];
                cur[u] = i;
            }
        gap[dep[u]]--;
        if(!gap[dep[u]])return ans;
        dep[u] = Min + 1;
        gap[dep[u]]++;
        if(u != start)u = edge[S[--top]^1].to;
    }
    return ans;
}

///最小费用最大流
struct Edge
{
    int to,next,cap,flow,cost;
} edge[M];
int head[N],tol;
int pre[N],dis[N];
bool vis[N];
int N;//节点总个数，节点编号从0~N-1
void init(int n)
{
    N = n;
    tol = 0;
    memset(head,-1,sizeof(head));
}
void addedge(int u,int v,int cap,int cost)
{
    edge[tol].to = v;
    edge[tol].cap = cap;
    edge[tol].cost = cost;
    edge[tol].flow = 0;
    edge[tol].next = head[u];
    head[u] = tol++;
    edge[tol].to = u;
    edge[tol].cap = 0;
    edge[tol].cost = -cost;
    edge[tol].flow = 0;
    edge[tol].next = head[v];
    head[v] = tol++;
}
bool spfa(int s,int t)
{
    queue<int>q;
    for(int i = 0; i < N; i++)
    {
        dis[i] = INF;
        vis[i] = false;
        pre[i] = -1;
    }
    dis[s] = 0;
    vis[s] = true;
    q.push(s);
    while(!q.empty())
    {
        int u = q.front();
        q.pop();
        vis[u] = false;
        for(int i = head[u]; i != -1; i = edge[i].next)
        {
            int v = edge[i].to;
            if(edge[i].cap > edge[i].flow &&
                    dis[v] > dis[u] + edge[i].cost )
            {
                dis[v] = dis[u] + edge[i].cost;
                pre[v] = i;
                if(!vis[v])
                {
                    vis[v] = true;
                    q.push(v);
                }
            }
        }
    }
    if(pre[t] == -1)return false;
    else return true;
}
//返回的是最大流，cost存的是最小费用
int minCostMaxflow(int s,int t,int &cost)
{
    int flow = 0;
    cost = 0;
    while(spfa(s,t))
    {
        int Min = INF;
        for(int i = pre[t]; i != -1; i = pre[edge[i^1].to])
        {
            if(Min > edge[i].cap - edge[i].flow)
                Min = edge[i].cap - edge[i].flow;
        }
        for(int i = pre[t]; i != -1; i = pre[edge[i^1].to])
        {
            edge[i].flow += Min;
            edge[i^1].flow -= Min;
            cost += edge[i].cost * Min;
        }
        flow += Min;
    }
    return flow;
}

///二分图匹配
///匈牙利算法，dfs实现
//二分图匹配：二分图最大配对数
//最小点覆盖：选取最少的点覆盖所有边
//最大独立集：选取最多的点，集合中点之间无连接
//二分图最大匹配==最小点覆盖==总点数-最大独立集
int n,m,cnt;
int f[N];
int vm[N],um[N];
bool vis[N];
vector<int>g[N];
void init()
{
    cnt = 0;
    memset(f,-1,sizeof(f));
    memset(vm,-1,sizeof(vm));
    memset(um,-1,sizeof(um));
    for(int i=0;i<=n+m;i++)
        g[i].clear();
}
void inserts(int u, int v)
{
    g[u].push_back(v);
    g[v].push_back(u);
}
int dfs(int u)
{
    int v;
    for(int i=0;i<g[u].size();i++)
    {
        v = g[u][i];
        if(vis[v])
            continue;
        vis[v] = 1;
        //能直接找到增广路径或者连接该节点的节点有其他增广路径
        if(vm[v] == -1 || dfs(vm[v])!=-1)
        {
            vm[v] = u; um[u] = v;
            return v;
        }
    }
    return -1;
}
//染色
bool dye(int u)
{
    int v;
    for(int i=0;i<g[u].size();i++)
    {
        v = g[u][i];
        if(f[v]==f[u])
            return false;
        if(f[v]!=-1)
            continue;
        f[v] = f[u]^1;
        dye(v);
    }
    return true;
}
bool Dye()
{
    for(int i=1;i<=n;i++)
        if(f[i] == -1)
        {
            f[i] = 0;
            if(!dye(i))
                return false;
        }
    return true;
}
int maxMatch()
{
    Dye();
    int res = 0;
    for(int i=1;i<=n+m;i++)
        if(!f[i])
        {
            memset(vis,0,sizeof(vis));
            if(dfs(i)!=-1)
                res++;
        }
    //取得最大点集覆盖
    return res;
}

///Hopcroft-Karp算法
//O(E*sqrt(V))
//只有u指向v的边
vector<int>g[N];
int um[N],vm[N],n;
int dx[N],dy[N],dis;
bool vis[N];
void inserts(int u, int v)
{
    g[u].push_back(v);
}
bool searchP()
{
    queue<int>q;
    dis=INF;
    memset(dx,-1,sizeof(dx));
    memset(dy,-1,sizeof(dy));
    for(int i=1;i<=n;i++)
        if(um[i]==-1)
        {
            q.push(i);
            dx[i]=0;
        }
    while(!q.empty())
    {
        int u=q.front();q.pop();
        if(dx[u]>dis)  break;
        for(int i=0;i<g[u].size();i++)
        {
            int v = g[u][i];
            if(dy[v]==-1)
            {
                dy[v]=dx[u]+1;
                if(vm[v]==-1)  dis=dy[v];
                else
                {
                    dx[vm[v]]=dy[v]+1;
                    q.push(vm[v]);
                }
            }
        }
    }
    return dis!=INF;
}
bool dfs(int u)
{
    for(int i=0;i<g[u].size();i++)
    {
        int v = g[u][i];
        if(!vis[v]&&dy[v]==dx[u]+1)
        {
            vis[v]=1;
            if(vm[v]!=-1&&dy[v]==dis) continue;
            if(vm[v]==-1||dfs(vm[v]))
            {
                vm[v]=u;um[u]=v;
                return 1;
            }
        }
    }
    return 0;
}
int maxMatch()
{
    int res=0;
    memset(um,-1,sizeof(um));
    memset(vm,-1,sizeof(vm));
    while(searchP())
    {
        memset(vis,0,sizeof(vis));
        for(int i=1;i<=n;i++)
          if(um[i]==-1&&dfs(i))  res++;
    }
    return res;
}

/// KM算法
//复杂度O(nx*nx*ny)
//求最大权匹配
//若求最小权匹配，可将权值取相反数，结果取相反数
const int N = 310;
const int INF = 0x3f3f3f3f;
int nx,ny;//两边的点数
int g[N][N];//二分图描述
int linker[N],lx[N],ly[N];//y中各点匹配状态，x,y中的点标号
int slack[N];
bool visx[N],visy[N];
bool DFS(int x)
{
    visx[x] = true;
    for(int y = 0; y < ny; y++)
    {
        if(visy[y])continue;
        int tmp = lx[x] + ly[y] - g[x][y];
        if(tmp == 0)
        {
            visy[y] = true;
            if(linker[y] == -1 || DFS(linker[y]))
            {
                linker[y] = x;
                return true;
            }
        }
        else if(slack[y] > tmp)
        slack[y] = tmp;
    }
    return false;
}
int KM()
{
    memset(linker,-1,sizeof(linker));
    memset(ly,0,sizeof(ly));
    for(int i = 0;i < nx;i++)
    {
        lx[i] = -INF;
        for(int j = 0;j < ny;j++)
            if(g[i][j] > lx[i])
                lx[i] = g[i][j];
    }
    for(int x = 0;x < nx;x++)
    {
        for(int i = 0;i < ny;i++)
            slack[i] = INF;
        while(true)
        {
            memset(visx,false,sizeof(visx));
            memset(visy,false,sizeof(visy));
            if(DFS(x))break;
            int d = INF;
            for(int i = 0;i < ny;i++)
                if(!visy[i] && d > slack[i])
                    d = slack[i];
            for(int i = 0;i < nx;i++)
                if(visx[i])
                    lx[i] -= d;
            for(int i = 0;i < ny;i++)
            {
                if(visy[i])ly[i] += d;
                else slack[i] -= d;
            }
        }
    }
    int res = 0;
    for(int i = 0;i < ny;i++)
        if(linker[i] != -1)
            res += g[linker[i]][i];
    return res;
}

///强连通分量
///tarjan
//O(n+m)
vector<int>g[N];
stack<int>st;
// 深度优先搜索访问次序, 能追溯到的最早的次序
int dfn[N],low[N];
// 检查是否在栈中, 记录每个点在第几号强连通分量里
int inStack[N],belong[N];
// 索引号，强连通分量个数
int index,cnt;
int n,m;
void init()
{
    for(int i=0;i<N;i++)
        g[i].clear();
    while(!st.empty())st.pop();
	memset(dfn, 0, sizeof(dfn));
	memset(low, 0, sizeof(low));
	memset(inStack, 0, sizeof(inStack));
	index = cnt = 1;
}
void tarjan(int x,int fa)
{
	int i;
	// 刚搜到一个节点时low = dfn
	low[x] = dfn[x] = index;
	index++;
	st.push(x);
	inStack[x] = 1;
	int len = g[x].size();
	for(i=0;i<len;i++)
	{
	    int t=g[x][i];
		if(!dfn[t])
		{
			tarjan(t,x);
			// 回溯的时候改变当前节点的low值
			low[x] = min(low[x], low[t]);
		}
		// 如果新搜索到的节点已经被搜索过而且现在在栈中
		else if(inStack[t])
		{
		    // 更新当前节点的low值，这里的意思是两个节点之间有一条可达边，
		    // 而前面节点已经在栈中，那么后面的节点就可能和前面的节点在一个联通分量中
			low[x] = min(low[x], dfn[t]);
		}
	}
	// 最终退回来的时候 low == dfn ， 没有节点能将根节点更新，那必然就是根节点
	if(low[x] == dfn[x])
	{
		int temp;
		// 一直出栈到此节点， 这些元素是一个强联通分量
		while(!st.empty())
		{
			temp = st.top();
			st.pop();
			belong[temp] = cnt; // 标记强联通分量
		 	inStack[temp] = 0;
		 	if(temp == x)
		 		break;
		}
		cnt++;
	}
}
// tarjan的成果是得到了一个belong数组，记录每个节点分别属于哪个强联通分量
int solve()
{
    for(int i = 1; i <= n; i++)
        if(!dfn[i])
            tarjan(i,i);
    return cnt;
}

///双连通分量
///边双连通分量
//连通分量中不含有桥，O(n+m);
vector<int>g[N];
stack<int>st;
// 深度优先搜索访问次序, 能追溯到的最早的次序
int dfn[N],low[N];
// 检查是否在栈中, 记录每个点在第几号强连通分量里
int inStack[N],belong[N];
// 索引号，双连通分量个数
int index,cnt;
int n,m;
void init()
{
    for(int i=0;i<N;i++)
        g[i].clear();
    while(!st.empty())st.pop();
    memset(dfn, 0, sizeof(dfn));
    memset(low, 0, sizeof(low));
    memset(inStack, 0, sizeof(inStack));
    index = cnt = 1;
}
void tarjan(int x,int fa)
{
    int i;
    // 刚搜到一个节点时low = dfn
    low[x] = dfn[x] = index;
    index++;
    st.push(x);
    inStack[x] = 1;
    int len = g[x].size();
    int mark = 0;
    for(i=0;i<len;i++)
    {
        int t=g[x][i];
        //无向图双连通分量，mark防重边。
        if(!mark && t==fa)
        {
            mark=1;
            continue;
        }
        if(!dfn[t])
        {
            tarjan(t,x);
            // 回溯的时候改变当前节点的low值
            low[x] = min(low[x], low[t]);
        }
        // 如果新搜索到的节点已经被搜索过而且现在在栈中
        else if(inStack[t])
        {
            // 更新当前节点的low值，这里的意思是两个节点之间有一条可达边，
            // 而前面节点已经在栈中，那么后面的节点就可能和前面的节点在一个联通分量中
            low[x] = min(low[x], dfn[t]);
        }
    }

    // 最终退回来的时候 low == dfn ， 没有节点能将根节点更新，那必然就是根节点
    if(low[x] == dfn[x])
    {
        int temp;
        // 一直出栈到此节点， 这些元素是一个双联通分量
        while(!st.empty())
        {
            temp = st.top();
            st.pop();
            belong[temp] = cnt; // 标记双联通分量
            inStack[temp] = 0;
            if(temp == x)
                break;
        }
        cnt++;
    }
}
int solve()
{
    for(int i = 1; i <= n; i++)
        if(!dfn[i])
            tarjan(i,i);
    return cnt;
}

///点双连通分量
//内部不含有割点的点集，割点属于所有包含它的联通分量，
struct node
{
    int x,y;
    node(int a=0, int b=0)
    {
        x=a;y=b;
    }
}tn;
//blocks[]存每个块包含的点，bridge存是桥的边
vector<int>g[N],blocks[N];
vector<node>bridge;
stack<node>st;
// 深度优先搜索访问次序, 能追溯到的最早的次序
int dfn[N],low[N];
bool vis[N];
// 索引号，块的个数
int index,cnt;
int n,m;
void init()
{
    for(int i=0;i<N;i++)
        g[i].clear(),blocks[i].clear();
    bridge.clear();
    while(!st.empty())st.pop();
	memset(dfn, 0, sizeof(dfn));
	memset(low, 0, sizeof(low));
	index = cnt = 1;
}
void judge(int u,int v)
{
    int x,y;
    node temp;
    memset(vis,false,sizeof(vis));
    while(!st.empty())
    {
        temp = st.top();st.pop();
        x=temp.x;y=temp.y;
        if(!vis[y])blocks[cnt].push_back(y),vis[y]=true;
        if(x==u) break;
        if(!vis[x])blocks[cnt].push_back(x),vis[x]=true;
    }
    cnt++;
}
void tarjan(int x,int fa)
{
	low[x] = dfn[x] = index++;
	int len = g[x].size();
	for(int i=0;i<len;i++)
	{
	    int t=g[x][i];
	    if(t==fa)
            continue;
		if(!dfn[t] && dfn[t]<dfn[x])
		{
		    //加入树枝边
		    st.push(node(x,t));
			tarjan(t,x);
			low[x] = min(low[x], low[t]);
			if(dfn[x]<=low[t])
                judge(x,t);
            if(dfn[x]<low[t])
                bridge.push_back(node(x,t));
		}
		else if(dfn[t] < dfn[x])
        {
            //加入后向边
            st.push(node(x,t));
			low[x] = min(low[x], dfn[t]);
        }
	}
}
int solve()
{
    for(int i = 1; i <= n; i++)
        if(!dfn[i])
            tarjan(i,i);
    return cnt;
}

///最近公共祖先(LCA)
///tarjan
struct Tree{
    int head[MAXN];//前向星存图
    int next[MAXM];
    int to[MAXM];
    int val[MAXM];//权值
    int pos;
    Tree()
    {
        clear();
    }
    void clear()
    {
        memset(head,-1,sizeof(head));
        memset(val,0,sizeof(val));
        pos = 0;
    }
    void add(int u,int v,int w)
    {
        val[pos] = w;
        to[pos] = v;
        next[pos] = head[u];
        head[u] = pos++;
    }
}tree;
bool v[MAXN];
int in[MAXN];//入度
int dist[MAXN];//与根节点距离
int father[MAXN];//并查集处理
int n,m,q;
vector<int>query[MAXQ],num[MAXQ];
int ans[MAXQ];

void Init()
{
    memset(v,0,sizeof(v));
    memset(in,0,sizeof(in));
    memset(ans,0,sizeof(ans));
    for(int i=0;i<MAXQ;i++)
    {
        query[i].clear();
        num[i].clear();
    }
    tree.clear();
}
int Find(int x)
{
    if(father[x] != x)
        father[x] = Find(father[x]);
    return father[x];
}
void getDist(int u)//计算节点到根距离
{
    for(int i = tree.head[u];i!=-1;i=tree.next[i])
    {
        dist[tree.to[i]] = dist[u] + tree.val[i];
        getDist(tree.to[i]);
    }
}
void Tarjan(int u)
{
    father[u] = u;//当访问到一个节点的时候，先将其自己形成一个集合
    v[u] = true;//标记访问
    for(int i = tree.head[u];i!=-1;i=tree.next[i])
    {
        Tarjan(tree.to[i]);//递归处理
        father[tree.to[i]] = u;//处理结束后，将子节点集合加到父节点
    }
    for(int i = 0; i < query[u].size(); i++)//便利询问
    {
        if(v[query[u][i]])
        {
            ans[num[u][i]] =Find(query[u][i]);
        }
    }
}
int main()
{
    int x,y,t;
    Init();
    scanf("%d%d",&n,&m);
    for(int i=1;i<=m;i++)
    {
        scanf("%d%d%d",&x,&y,&t);
        tree.add(x,y,t);
        in[y]++;
    }
    scanf("%d",&q);
    for(int i=0;i<q;i++)
    {
        scanf("%d%d",&x,&y);
        query[x].push_back(y);
        query[y].push_back(x);
        num[x].push_back(i);
        num[y].push_back(i);
    }
    for(int i=1;i<=n;i++)
        if(in[i] == 0)
        {
            dist[i]=0;
            getDist(i);
            Tarjan(i);
            break;
        }
    //ans中得到的是最近公共祖先的节点，最近距离就是dist[x]+dist[y]-2*dist[ans[i]];
    for(int i=0;i<q;i++)
        printf("%d\n",ans[i]);
    return 0;
}

///ST-RMQ在线算法
#define N 40010
#define M 25
int n,m;
int flag,sum,ave,ans,res,num;
bool vis[N];
int dp[2*N][M], head[N];
struct node
{
    int u,v,w,next;
}e[2*N];
inline void add(int u ,int v ,int w )
{
    e[num].u = u; e[num].v = v; e[num].w = w;
    e[num].next = head[u]; head[u] = num++;
    u = u^v; v = u^v; u = u^v;
    e[num].u = u; e[num].v = v; e[num].w = w;
    e[num].next = head[u]; head[u] = num++;
}
int ver[2*N],R[2*N],first[N],dir[N];
//ver:节点编号 R：深度 first：点编号位置 dir：距离
void Init()
{
    num = 0;  sum = 0; dir[1] = 0;
    memset(head,-1,sizeof(head));
    memset(vis,false,sizeof(vis));
}
void dfs(int u ,int dep)
{
    vis[u] = true; ver[++sum] = u; first[u] = sum; R[sum] = dep;
    for(int k=head[u]; k!=-1; k=e[k].next)
        if( !vis[e[k].v] )
        {
            int v = e[k].v , w = e[k].w;
            dir[v] = dir[u] + w;
            dfs(v,dep+1);
            ver[++sum] = u; R[sum] = dep;
        }
}
void ST(int n)
{
    for(int i=1;i<=n;i++)
        dp[i][0] = i;
    for(int j=1;(1<<j)<=n;j++)
    {
        for(int i=1;i+(1<<j)-1<=n;i++)
        {
            int a = dp[i][j-1] , b = dp[i+(1<<(j-1))][j-1];
            dp[i][j] = R[a]<R[b]?a:b;
        }
    }
}
//中间部分是交叉的。
int RMQ(int l,int r)
{
    int k=0;
    while((1<<(k+1))<=r-l+1)
        k++;
    int a = dp[l][k], b = dp[r-(1<<k)+1][k]; //保存的是编号
    return R[a]<R[b]?a:b;
}

int LCA(int u ,int v)
{
    int x = first[u] , y = first[v];
    if(x > y) swap(x,y);
    int res = RMQ(x,y);
    return ver[res];
}

int main()
{
    int n,q,u,v,w,lca;
    Init();
    scanf("%d%d",&n,&q);
    for(int i=1; i<n; i++)
    {
        scanf("%d%d%d",&u,&v,&w);
        add(u,v,w);
    }
    dfs(1,1);
    ST(2*n-1);
    while(q--)
    {
        scanf("%d%d",&u,&v);
        lca = LCA(u,v);
        printf("%d\n",dir[u] + dir[v] - 2*dir[lca]);
    }
    return 0;
}

///数论！！！！！
///Fibonacci Number
int a[10]={0,1,1,2,3,5,8,13,21,34}
a[i]=a[i-1]+a[i-2];
a[n]=(pow((1+sqrt(5)),n)-pow((1-sqrt(5)),n))/(pow(2,n)*sqrt(5));

///Greatest Common Divisor 最大公约数,欧几里德算法
long long gcd(long long x,long long y)
{   return y?gcd(y,x%y):x;  }
long long gcd(long long x,long long y)
{
    if(x<y)x^=y,y^=x,x^=y;
    long long t;
    while(y)
        t=y,y=x%y,x=t;
    return x;
}

///Lowest Common Multiple 最小公倍数
long long lcm(long long x,long long y)
{   return x/gcd(x,y)*y;    }

///扩展欧几里德算法
//扩展欧几里德求出a*x+b*y=gcd(a,b)的一组解,x0,y0，
//x=x2+b/gcd(a,b)*t,y=y2-a/gcd(a,b)*t,(t为整数)，即为ax+by=c的所有解。
long long exgcd(long long a, long long b,long long &x,long long &y)
{
    if (b == 0)
    { x = 1; y = 0; return a; }
    long long g = exgcd(b, a % b ,x ,y);
    //x1=y2,  y1=x2-a/b*y2
    long long t = x - a / b * y;
    x = y;
    y = t;
    return g;  //return gcd
}
long long solve(long long  a,long long b,long long c)
{
    long long x,y,x0,y0,x1,y1,t = exgcd(a,b,x0,y0);
    if(c%t!=0)return 0;// NO solution;
//    x = x0+b/t; y = y0-a/t; //通解
    x1 = (x*c/t);  y1 = (y*c/t);//求原方程的解
//    x1 = (x0*c/t)%b;   x1 = (x1%(b/t)+b/t)%(b/t);//取x的最小整数解;
    printf("%lld %lld\n",x1,y1);
    return 0;
}

///快速幂运算
//eg:x^22==x^16+x^4+x^2;
////22==10110
long long power(long long x,long long k,long long mod)
{
	long long ans = 1;
	while(k)
    {
		if(k & 1) ans=ans*x%mod;
		x=x*x%mod;
		k >>= 1;
	}
	return ans;
}

///乘积求模
// a * b % n
//例如: b = 1011101那么a * b mod n = (a * 1000000 mod n + a * 10000 mod n + a * 1000 mod n + a * 100 mod n + a * 1 mod n) mod n
long long mod_mul(long long a,long long b,long long n)
{
    long long res = 0;
    while(b)
    {
        if(b&1)    res = (res + a) % n;
        a = (a + a) % n;
        b >>= 1;
    }
    return res;
}

///次方求模
//1:
//a^b % n
long long mod_exp(long long a,long long b,long long n)
{
    long long res = 1;
    while(b)
    {
        if(b&1)    res = mod_mul(res, a, n);
        a = mod_mul(a, a, n);
        b >>= 1;
    }
    return res;
}
//2:
//a^b mod c=(a mod c)^b mod c很容易设计出一个基于二分的递归算法。
//快速幂算法，数论二分
long long mod_exp(long long a,long long b,long long c)
{
    long long res;
    if(b==0)  return 1%c;
    if(b==1)  return a%c;
    //递归调用，采用二分递归算法，,注意这里n/2会带来奇偶性问题
    t = mod_exp(a,b/2,c);
    t = t*t%c;//二分，乘上另一半再求模
    if(b&1)  t=t*a%c;//n是奇数，因为n/2还少乘了一次a
    return t;
}

///质数判断
///简单Prime判断
bool prime(int a)
{
    if(a==0||a==1)return false;
    if(a==2)return true;
    if(a%2==0)return false;
    for(int i=3;i<=sqrt(a);i+=2)
        if(a%i==0)return false;
    return true;
}

///Sieve Prime素数筛选法
#define N 11234567
bool mark[N];
int pri[N/10],cnt;
void SP()
{
    cnt=0;
    memset(mark,true,sizeof(mark));
    mark[0]=mark[1]=false;
    for(int i=2;i<N;i++)
    {
        if(mark[i])
            pri[cnt++]=i;
        for (int j=0;(j<cnt)&&(i*pri[j]<N);j++)
        {
            mark[i*pri[j]]=false;
            if (i%pri[j]==0)
                break;
        }
    }
}

///双筛
#define N 101000
long long  a[N],sum,n,m;
bool mark[N],num[N];//num[i]存n+i是否素数，
void SP()
{
    sum = 0;
    memset(mark,true,sizeof(mark));
    mark[0] = mark[1] = false;
    for(long long i=2;i<=N;i++)
        if(mark[i])
        {
            a[sum++] = i;
            for(long long j=2;i*j<=N;j++)
                mark[i*j] = false;
        }
}
void solve()
{
    SP();
    if(n>m){n^=m;m^=n;n^=m;}
    memset(num,false,sizeof(num));
    for(long long i=0;i<sum;i++)
    {
        long long s = (n-1)/a[i]+1;
        long long t = m/a[i];
        for(long long j=s;j<=t;j++)
            if(j>1)
                num[j*a[i]-n]=true;
    }
}

///Miller-Rabin素数测试算法
//费马小定理：gcd(a,n)==1,a^(n-1)==1(mod n)，n为素数.
//Carmichael：满足费小的合数，341,561,1105,1729……1e9范围内255个。
//二次探测定理：x^2==1(mod n),n为素数,x==1||x==n-1.
#define T 10//随机算法判定次数，N越大，判错概率越小
long long modMul(long long a,long long b,long long n)
{
    long long res = 0;
    while(b)
    {
        if(b&1)    res = (res + a) % n;
        a = (a + a) % n;
        b >>= 1;
    }
    return res;
}
long long modExp(long long x,long long k,long long mod)
{
    long long ans = 1;
    while(k)
    {
        if(k & 1) ans = modMul(ans, x, mod);
        x = modMul(x, x, mod);
        k >>= 1;
    }
    return ans;
}
bool millerRabin(long long n)
{
    if(n == 2 || n == 3 || n == 5 || n == 7 || n == 11)    return true;
    if(n == 1 || !(n%2) || !(n%3) || !(n%5) || !(n%7) || !(n%11))    return false;

    long long x, pre, u;
    int i, j, k = 0;
    u = n - 1;

    while(!(u&1))
    {
        k++;
        u >>= 1;
    }

    srand((long long)time(0));
    for(i = 0; i < T; ++i)
    {
        x = rand()%(n-2) + 2;
        if((x%n) == 0)
            continue;
        x = modExp(x, u, n);
        pre = x;
        for(j = 0; j < k; ++j)
        {
            x = modMul(x,x,n);
            //二次探测判断
            if(x == 1 && pre != 1 && pre != n-1)
                return false;
            pre = x;
        }
        //费小判断
        if(x != 1)
            return false;
    }
    return true;
}

///因子
///唯一分解定理(因子个数)
//sum=(s[0]+1)*(s[1]+1)*(s[2]+1)……
//s[i]为a的一个素因子的个数。
long long getFactor(long long x)
{
    long long sum=1;
    for(int i=0;i<cnt&&pri[i]*pri[i]<=x;i++)
    {
        long long res=0;
        while(x%pri[i]==0&&x)x/=pri[i],res++;
        sum*=(res+1);
    }
    if(x>1)
        sum*=2;
    return sum;
}
///因子和
//sum = (p[0]^(e[0]+1)-1)/(p[0]-1)*(p[1]^(e[1]+1)-1)/(p[1]-1)*……
//x有e[i]个素因子p[i];
long long getFactorSum(long long x)
{
    long long sum=1;
    for(int i=0;i<cnt&&pri[i]*pri[i]<=x;i++)
    {
        long long res=1,tmp=1;
        while(x%pri[i]==0&&x)x/=pri[i],tmp*=pri[i],res+=tmp;
        if(res)sum*=res;
    }
    if(x>1)
        sum*=x;
    return sum;
}
///质因数分解
int num[N],now,all;
void solve(int t)
{
    now = all = 0;
    int tt=t;
    for(int i=0;i<cnt&&pri[i]*pri[i]<=tt;i++)
    {
        if(tt%pri[i]==0)
            num[now++] = pri[i];
        while(tt%pri[i]==0)
            tt/=pri[i],all++;
    }
    if(tt>1)num[now++] = tt, all++;
}
///pollard_rho 算法
long long factor[1000];//质因数分解结果（刚返回时是无序的）
int sum;//质因数的个数。数组小标从0开始
long long gcd(long long a,long long b)
{
    if(a==0)return 1;
    if(a<0) return gcd(-a,b);
    while(b)
    {
        long long t=a%b;
        a=b;
        b=t;
    }
    return a;
}
long long pollardRho(long long x,long long c)
{
    long long i=1,k=2;
    long long x0=rand()%x;
    long long y=x0;
    while(1)
    {
        i++;
        x0=(modMul(x0,x0,x)+c)%x;
        long long d=gcd(y-x0,x);
        if(d!=1&&d!=x) return d;
        if(y==x0) return x;
        if(i==k){y=x0;k+=k;}
    }
}
//对n进行素因子分解
void findFac(long long x)
{
    srand((long long)time(0));
    if(millerRabin(x))//素数
    {
        factor[sum++]=x;
        return;
    }
    long long p=x;
    while(p>=x)
        p=pollardRho(p,rand()%(x-1)+1);
    findFac(p);
    findFac(x/p);
}

///欧拉函数
//在数论，对正整数n，欧拉函数是少于或等于n的数中与n互质的数的数目。
//欧拉定理：a与p互质，a^x==1(mod p)则x==euler[p];
///直接求解欧拉函数
long long euler(long long n)
{
     long long res=n,a=n;
     for(long long i=2;i*i<=a;i++)
         if(a%i==0)
         {
             res=res/i*(i-1);
             while(a%i==0) a/=i;
         }
     if(a>1) res=res/a*(a-1);
     return res;
}

///筛选法打欧拉函数表
int eul[N];
void init()
{
    for(int i=1;i<N;i++) eul[i] = i;
    for(int i=2;i<N;i++)
        if(eul[i]==i)
            for(int j=i;j<N;j+=i)
                eul[j] = eul[j]/i*(i-1);
}

///约瑟夫环
//就是长度为n的环每数m个就删去，最后留下一个。
//找最后一点
for(i=2,res=0;i<=n;i++)
    res=(res+m)%i;

//找保留区间的步长选择
res=0;k=1;
for(i=0;i<n;i++)
{
    res=(res+k-1)%(2*n-i);
    if(res<n)
        i=-1,res=0,k++;
}

///高斯消元
int a[N][N];//增广矩阵
int x[N];//解集
bool free_x[N];//标记是否是不确定的变元
int  gcd(int x,int y)
{   return y?gcd(y,x%y):x;  }
int lcm(int a,int b)
{   return a/gcd(a,b)*b;    }
// 高斯消元法解方程组(Gauss-Jordan elimination).(-2表示有浮点数解，但无整数解，
//-1表示无解，0表示唯一解，大于0表示无穷解，并返回自由变元的个数)
//有equ个方程，var个变元。增广矩阵行数为equ,分别为0到equ-1,列数为var+1,分别为0到var.
int Gauss(int equ,int var)
{   // max_r当前这列绝对值最大的行.
    int i,j,k,max_r,col,ta,tb,LCM,temp,free_x_num,free_index;
    for(int i=0;i<=var;i++)
    {
        x[i]=0;
        free_x[i]=true;
    }
    //转换为上阶梯阵.
    col=0; // 当前处理的列
    for(k = 0;k < equ && col < var;k++,col++)
    {   // 枚举当前处理的行.
        // 找到该col列元素绝对值最大的那行与第k行交换.(为了在除法时减小误差)
        max_r=k;
        for(i=k+1;i<equ;i++)
        {
            if(abs(a[i][col])>abs(a[max_r][col])) max_r=i;
        }
        if(max_r!=k)
        {   // 与第k行交换.
            for(j=k;j<var+1;j++) swap(a[k][j],a[max_r][j]);
        }
        if(a[k][col]==0)
        {   // 说明该col列第k行以下全是0了，则处理当前行的下一列.
            k--;
//            free_x[free_x_num++] = col;//自由变元
            continue;
        }
        for(i=k+1;i<equ;i++)
        {   // 枚举下面要化为0的行
            if(a[i][col]!=0)
            {
                LCM = lcm(abs(a[i][col]),abs(a[k][col]));
                ta = LCM/abs(a[i][col]);
                tb = LCM/abs(a[k][col]);
                if(a[i][col]*a[k][col]<0)tb=-tb;//异号的情况是相加
                for(j=col;j<var+1;j++)
                {
                    a[i][j] = a[i][j]*ta-a[k][j]*tb;
                }
            }
            //0/1转换
//            if(a[i][col] != 0)
//            {
//                for(int j = col;j < var+1;j++)
//                    a[i][j] ^= a[k][j];
//            }
        }
    }
    // 1. 无解的情况: 化简的增广阵中存在(0, 0, ..., a)这样的行(a != 0).
    for (i = k; i < equ; i++)
    { // 对于无穷解来说，如果要判断哪些是自由变元，那么初等行变换中的交换就会影响，则要记录交换.
        if (a[i][col] != 0) return -1;
    }
    // 2. 无穷解的情况: 在var * (var + 1)的增广阵中出现(0, 0, ..., 0)这样的行，即说明没有形成严格的上三角阵.
    // 且出现的行数即为自由变元的个数.
    if (k < var)
    {
        // 首先，自由变元有var - k个，即不确定的变元至少有var - k个.
        //确定自由变元
        for (i = k - 1; i >= 0; i--)
        {
            // 第i行一定不会是(0, 0, ..., 0)的情况，因为这样的行是在第k行到第equ行.
            // 同样，第i行一定不会是(0, 0, ..., a), a != 0的情况，这样的无解的.
            free_x_num = 0; // 用于判断该行中的不确定的变元的个数，如果超过1个，则无法求解，它们仍然为不确定的变元.
            for (j = 0; j < var; j++)
            {
                if (a[i][j] != 0 && free_x[j]) free_x_num++, free_index = j;
            }
            if (free_x_num > 1) continue; // 无法求解出确定的变元.
            // 说明就只有一个不确定的变元free_index，那么可以求解出该变元，且该变元是确定的.
            temp = a[i][var];
            for (j = 0; j < var; j++)
            {
                if (a[i][j] != 0 && j != free_index) temp -= a[i][j] * x[j];
            }
            x[free_index] = temp / a[i][free_index]; // 求出该变元.
            free_x[free_index] = 0; // 该变元是确定的.
        }
        return var - k; // 自由变元有var - k个.
    }
    // 3. 唯一解的情况: 在var * (var + 1)的增广阵中形成严格的上三角阵.
    // 计算出Xn-1, Xn-2 ... X0.
    for (i = var - 1; i >= 0; i--)
    {
        temp = a[i][var];
        for (j = i + 1; j < var; j++)
            if (a[i][j] != 0)
                temp -= a[i][j] * x[j];
        if (temp % a[i][i] != 0) return -2; // 说明有浮点数解，但无整数解.
        x[i] = temp / a[i][i];
        //0/1转换
//        x[i] = a[i][var];
//        for(int j = i+1;j < var;j++)
//            x[i] ^= (a[i][j] && x[j]);
    }
    return 0;
}
int main()
{
    int i,j,equ,var;
    while (scanf("%d %d", &equ, &var) != EOF)
    {
        memset(a, 0, sizeof(a));
        for (i = 0; i < equ; i++)
        {
            for (j = 0; j < var + 1; j++)
            {
                scanf("%d", &a[i][j]);
            }
        }
        int free_num = Gauss(equ,var);
        if (free_num == -1) printf("无解!\n");
        else if (free_num == -2) printf("有浮点数解，无整数解!\n");
        else if (free_num > 0)
        {
            printf("无穷多解! 自由变元个数为%d\n", free_num);
            for (i = 0; i < var; i++)
            {
                if (free_x[i]) printf("x%d 是不确定的\n", i + 1);
                else printf("x%d: %d\n", i + 1, x[i]);
            }
        }
        else
        {
            for (i = 0; i < var; i++)
            {
                printf("x%d: %d\n", i + 1, x[i]);
            }
        }
        printf("\n");
    }
    return 0;
}

///高斯消元处理开关问题
//一维开关之间互相有影响，求始末状态之间改变的方法数；
int main()
{
    int i,j,equ,var;
    int k,kk,xx,yy;
    scanf("%d",&k);
    while(k--)
    {
        scanf("%d",&equ);
        var=equ;
        memset(a, 0, sizeof(a));
        for (i = 0; i < equ; i++)//输入原始状态
            scanf("%d", &b[i]);
        for(i=0;i<equ;i++)
        {
            int t;
            scanf("%d",&t);
            if((t+2+b[i])%2)//该行状态是否需要改变
                a[i][equ]=1;
        }
        while(scanf("%d%d",&xx,&yy)&&(xx!=0||yy!=0))
            a[yy-1][xx-1] = 1;//yy的状态会被xx改变
        for(i=0;i<equ;i++)
            a[i][i]=1;//选择自己一定会改变自己的状态
        int free_num = Gauss(equ,var);//模版求出自由变元数量
        if (free_num == -1) printf("Oh,it's impossible~!!\n");
        else if (free_num > 0)
            printf("%.0lf\n", pow(2.0,free_num));
        else
            printf("1\n");
    }
    return 0;
}

///组合数学！！！！！
///排列组合
//求C(x,y),(x>=y);
long long gcd(long long a,long long b)
{    return b?gcd(b,a%b):a;     }
long long solve(long long x,long long y)
{
    long long c,d;
    long long a[y+1],b[y+1];
    if(y>x-y)
        y=x-y;
    for(long long i=1,j=x;i<=y;i++,j--)
        a[i]=i,b[i]=j;
    for(long long i=1;i<=y;i++)
        for(long long j=1;j<=y;j++)
        {
            if(a[i]<=1)break;
            long long t = gcd(b[j],a[i]);
            a[i]/=t;
            b[j]/=t;
        }
    long long res=1;
    for(long long i=1;i<=y;i++)
        res*=b[i];
    return res;
}

///Lucas定理
//求C(n,m)%p;
long long fac[100005];
long long  getFact(long long  p){
    fac[0]=1;
    for(int i=1;i<=p;i++)
        fac[i]=(fac[i-1]*i)%p;
}
long long power(long long x,long long k,long long mod)
{
	long long ans = 1;
	while(k)
    {
		if(k & 1) ans=ans*x%mod;
		x=x*x%mod;
		k >>= 1;
	}
	return ans;
}
long long  Lucas(long long  n,long long  m,long long  p){
    long long  ret=1;
    while(n&&m){
        long long  a=n%p,b=m%p;
        if(a<b) return 0;
        ret=(ret*fac[a]*power(fac[b]*fac[a-b]%p,p-2,p))%p;
        n/=p;
        m/=p;
    }
    return ret;
}

///全排列
int len;int sum=0;
char a[1000001][101];
void Swap(char *a, char *b)
{
    char t = *a;
    *a = *b;
    *b = t;
}
//k表示当前选取到第几个数,m表示共有多少数.
void AllRange(char *pszStr, int k)
{
    if (k == len)
    {
        strcpy(a[sum++],pszStr);
//        printf("%d %s\n",sum,pszStr);
    }
    else
    {
        for (int i = k; i <= len; i++) //第i个数分别与它后面的数字交换就能得到新的排列
        {
            Swap(pszStr + k, pszStr + i);
            AllRange(pszStr, k + 1);
            Swap(pszStr + k, pszStr + i);
        }
    }
}
int main()
{
    char szTextStr[] = "123456789";
    len=strlen(szTextStr) - 1;
    sum=0;
    AllRange(szTextStr, 0);
    printf("%d\n",sum);
    return 0;
}

///错排公式
a[n]=(n-1)*(a[n-1]+a[n-2]);//n封信全部装错的可能

///母函数：
///整数划分
//将整数n 分成最多n 份, 且每份不能为空, 任意两种分法不能相同
#define N 130
long long a[N],b[N],c[N];
void Init()
{
    memset(a,0,sizeof(a));
    memset(c,0,sizeof(c));
    int i,j,k;
    c[1]=1;
    for(i=0;i<=N;i++)
        a[i] = 1;   /*第一个多项式：g(x, 1) = x^0 + x^1 + x^2 + x^3 +  */
    for(i=2;i<=N;i++)
    {
        memset(b,0,sizeof(b));
        for(j=0;j<=N;j++) // 把第k个多项式依次乘到之前结果的每一位上
            for(k=0;k+j<=N;k+=i)/*第k个多项式：g(x, k) = x^0 + x^(k) + x^(2k) + x^(3k) +  */
                b[j+k]+=a[j];
        c[i] = b[i];
        memcpy(a,b,sizeof(b));
    }
}

///康托展开
int  fac[] = {1,1,2,6,24,120,720,5040,40320}; //i的阶乘为fac[i]
/*  康托展开.{1...n}的全排列由小到大有序，s[]为第几个数  */
int KT(int n, int s[])
{
    int i, j, t, sum;
    sum = 0;
    for (i=0; i<n; i++)
    {
        t = 0;
        for (j=i+1; j<n; j++)
            if (s[j] < s[i])
                t++;
        sum += t*fac[n-i-1];
    }
    return sum+1;
}

///逆康托展开
int  fac[] = {1,1,2,6,24,120,720,5040,40320}; //i的阶乘为fac[i]
/*  康托展开的逆运算. {1...n}的全排列，中的第k个数为s[]  */
void invKT(int n, int k, int s[])
{
    int i, j, t, vst[8]={0};
    k--;
    for (i=0; i<n; i++)
    {
        t = k/fac[n-i-1];
        k %= fac[n-i-1];
        for (j=1; j<=n; j++)
            if (!vst[j])
            {
                if(t==0)break;
                t--;
            }
        s[i] = j;
        vst[j] = 1;
    }
}

///Catalan Number
//int num[10]={1,1,2,5,14,42,132,429,1430,4862};
long long num[N];
void Init()
{
    memset(num,0,sizeof(num));
    num[0] = num[1] = 1;
    for(long long i=2;i<=N;i++)
        for(long long j=0;j<i;j++)
            num[i] += num[j]*num[i-j-1];
}
/*Application:
1) 将n + 2 边形沿弦切割成n 个三角形的不同切割数
2) n + 1 个数相乘, 加满括号的不同方法数
3) n 个节点的不同形状的二叉树数
4) 从n * n 方格的左上角移动到右下角不升路径数的一半
5) 还有2n个人买票的问题，有n个人有5元钱，n个人有10元钱，票价为5元
   求共有多少种排队方式可以保证对每一个拿十元的人都有5元钱能找给他
6) n个数依次入栈，每个数只能入栈出栈一次，有多少种出栈序列
*/

///Stirling Number(Second Kind)
//S(n, m)表示含n 个元素的集合划分为m 个集合的情况数或者是n 个有标号的球放到m 个无标号的盒子中, 要求无一为空, 其不同的方案数
a[n][m]=a[n-1][m-1]+m*a[n-1][m];

///容斥原理
//大小为n1,n2,n3的三个集合，有多少种组合
//二集合中第i个不能与a[i]个一集合的项组合，不能与b[i]个三集合的项组合
int a[N],b[N];
long long solve(int n2,int n2,int n3)
{
    long long sum=n1*n2*n3;
    for(int i=1;i<=n2;i++)
    {
        sum-=a[i]*n3;
        sum-=b[i]*n1;
        sum+=(a[i]*b[i]);//容斥原理
    }
    return sum;
}
//求小于n的数中能够整除a数组中某个数的数有多少个。
void dfs(int now,int step,int flag)
{
    sum+=(n/step)*flag;
    for(int i=now+1;i<m;i++)
        dfs(i,lcm(step,a[i]),-flag);
}

///矩阵
struct Matrix
{
    long long ma[N][N];
    long long n,m;
};
Matrix matMul(Matrix m1, Matrix m2, long long mod)
{
    Matrix ans;
    memset(ans.ma,0,sizeof(ans.ma));
    for(int i=0;i<m1.n;i++)
        for(int j=0;j<m2.m;j++)
            for(int k=0;k<m1.m;k++)
                ans.ma[i][j]=(ans.ma[i][j]+m1.ma[i][k]*m2.ma[k][j]+mod)%mod;
    ans.n=m1.n; ans.m=m2.m;
    return ans;
}
Matrix matPow(Matrix m1, long long k, long long mod)
{
    Matrix ans;
    for(int i=0;i<m1.n;i++)
        for(int j=0;j<m1.m;j++)
            ans.ma[i][j] = (i==j);
    ans.n=ans.m=m1.n;
    while(k)
    {
        if(k&1)ans = matMul(ans,m1,mod);
        m1 = matMul(m1,m1,mod);
        k>>=1;
    }
    return ans;
}

///计算几何！！！！！
const double eps = 1e-8;
const double PI = acos(-1.0);
int sgn(double x)
{
	if (fabs(x) < eps)return 0;
	if (x < 0)return -1;
	else return 1;
}
///1.1 Point
struct Point
{
	double x,y;
	Point() {}
	Point(double x,double y):x(x),y(y){}
	Point operator -(const Point & b)const
	{
		return Point(x - b.x,y - b.y);
	}
	bool operator ==(const Point & b)const
	{
		return x==b.x && y==b.y;
	}
	//两点间距离
    double operator &(const Point & b)const
    {
        return sqrt((*this-b)*(*this-b));
    }
	//*叉积
	//1.判断向量方向, >0 P在Q顺时针方向 ==0 同向或反向
	//2.求面积, 叉积的长度 |a×b| 可以解释成以a和b为邻边的平行四边形的面积
	double operator ^(const Point & b)const
	{
		return x*b.y - y*b.x;
	}
	//*点积
	//a*b = |a|*|b|*cos<a,b> (0<= (<a,b>) <= PI)
	double operator *(const Point & b)const
	{
		return x*b.x + y*b.y;
	}
	//*比较(先x后y) //极角排序需要最低的一点(改为先y后x)
	bool operator < (const Point & b) const
	{
	    if(x!=b.x)  return x<b.x;
	    if(y!=b.y)  return y<b.y;
	}
	//*绕原点旋转角度B（弧度值）后x,y的值
	void transXY(double B)
	{
		double tx = x,ty = y;
		x = tx*cos(B) - ty*sin(B);
		y = tx*sin(B) + ty*cos(B);
	}
};
///1.2 Line
struct Line
{
	Point s,e;
	Line() {}
	Line(Point s,Point e):s(s),e(e){}
	bool operator ==(const Line & b)const
	{
		return s==b.s && e==b.e;
	}
	//*获取Line的向量
    Point getV()
    {
        return e-s;
    }
	//两直线相交求交点
	//返回为（INF，INF）表示直线重合, (-INF,-INF) 表示平行, (x,y)是相交的交点
	Point operator &(const Line & b)const
	{
		Point res = s;
		if (sgn((s-e)^(b.s-b.e)) == 0)
		{
			if (sgn((s-b.e)^(b.s-b.e)) == 0)
				return Point(INF,INF);//重合
			else return Point(-INF,-INF);//平行
		}
		double t = ((s-b.s)^(b.s-b.e))/((s-e)^(b.s-b.e));
		res.x += (e.x-s.x)*t;
		res.y += (e.y-s.y)*t;
		return res;
	}
	//*判断线段相交
    bool operator ^ (const Line & b) const
    {
        return
            max(s.x,e.x) >= min(b.s.x,b.e.x) &&
            max(b.s.x,b.e.x) >= min(s.x,e.x) &&
            max(s.y,e.y) >= min(b.s.y,b.e.y) &&
            max(b.s.y,b.e.y) >= min(s.y,e.y) &&
            sgn((b.s-e)^(s-e))*sgn((b.e-e)^(s-e)) <= 0 &&
            sgn((s-b.e)^(b.s-b.e))*sgn((e-b.e)^(b.s-b.e)) <= 0;
    }
	//*判断直线l1和线段l2是否相交
    bool operator % (const Line & b) const
    {
        return sgn((b.s-e)^(s-e))*sgn((b.e-e)^(s-e)) <= 0;
    }
};
///1.3 点与直线关系
//点到直线距离
//返回为result,是点到直线最近的点
Point PtoL(Point P,Line L)
{
	Point result;
	double t = ((P-L.s)*(L.e-L.s))/((L.e-L.s)*(L.e-L.s));
	result.x = L.s.x + (L.e.x-L.s.x)*t;
	result.y = L.s.y + (L.e.y-L.s.y)*t;
	return result;
}

//点到线段的距离
//返回点到线段最近的点
Point PtoS(Point P,Line L)
{
	Point res;
	double t = ((P-L.s)*(L.e-L.s))/((L.e-L.s)*(L.e-L.s));
	if (t >= 0 && t <= 1)
	{
		res.x = L.s.x + (L.e.x - L.s.x)*t;
		res.y = L.s.y + (L.e.y - L.s.y)*t;
	}
	else
	{
		if( (P&L.s) < (P&L.e))	res = L.s;
		else                    res = L.e;
	}
	return res;
}

//判断点在线段上
bool onSeg(Point P,Line L)
{
	return
	    sgn((L.s-P)^(L.e-P)) == 0 &&
	    sgn((P.x - L.s.x) * (P.x - L.e.x)) <= 0 &&
	    sgn((P.y - L.s.y) * (P.y - L.e.y)) <= 0;
}

///1.4 多边形
//计算多边形面积
//点的编号从0~n-1
double Area(Point p[],int n)
{
	double res = 0;
	for (int i = 0; i < n; i++)
		res += (p[i]^p[(i+1)%n])/2;
	return fabs(res);
}

//判断点在凸多边形内
//点形成一个凸包，而且按逆时针排序（如果是顺时针把里面的<0改为>0）
//点的编号:0~n-1
//返回值:
//-1:点在凸多边形外
//0:点在凸多边形边界上
//1:点在凸多边形内
int inConvexPoly(Point a,Point p[],int n)
{
	for (int i = 0; i < n; i++)
	{
		if (sgn((p[i]-a)^(p[(i+1)%n]-a)) < 0)return -1;
		else if (OnSeg(a,Line(p[i],p[(i+1)%n])))return 0;
	}
	return 1;
}

//判断点在任意多边形内
//射线法， poly[]的顶点数要大于等于3,点的编号0~n-1
//返回值
//-1:点在任意多边形外
//0:点在任意多边形边界上
//1:点在任意多边形内
int inPoly(Point p,Point poly[],int n)
{
	int cnt;
	Line ray,side;
	cnt = 0;
	ray.s = p;
	ray.e.y = p.y;
	ray.e.x = -100000000000.0;//-INF,注意取值防止越界
	for (int i = 0; i < n; i++)
	{
		side.s = poly[i];
		side.e = poly[(i+1)%n];
		if (OnSeg(p,side))return 0;
//如果平行轴则不考虑
		if (sgn(side.s.y - side.e.y) == 0)
			continue;

		if (OnSeg(side.s,ray))
		{
			if (sgn(side.s.y - side.e.y) > 0)cnt++;
		}
		else if (OnSeg(side.e,ray))
		{
			if (sgn(side.e.y - side.s.y) > 0)cnt++;
		}
		else if (ray^side)
			cnt++;
	}
	if (cnt % 2 == 1)return 1;
	else return -1;
}

//*判断凸多边形
//允许共线边
//点可以是顺时针给出也可以是逆时针给出
//点的编号1~n-1
bool isConvex(Point poly[],int n)
{
	bool s[3]={0};
	for (int i = 0; i < n; i++)
	{
		s[sgn( (poly[(i+1)%n]-poly[i])^(poly[(i+2)%n]-poly[i]) )+1] = true;
		if (s[0] && s[2])return false;
	}
	return true;
}

///2、凸包
//*求凸包， Graham算法
// 点的编号0~n-1
// 返回凸包结果Stack[0~top-1]为凸包的“编号”
const int N = 10100;
Point hull[N];
int Stack[N],top;
//相对于hull[pos]的极角排序
//*极角排序需要最低的一点开始
bool _cmp(Point p1,Point p2)
{
    const int pos = 0;
	double tmp = (p1-hull[pos])^(p2-hull[pos]);
	if (sgn(tmp) > 0)return true;
	else if (sgn(tmp) == 0 && sgn((p1&hull[pos]) - (p2&hull[pos])) <= 0)
		return true;
	else return false;
}
void Graham(int n)
{
	Point p0;
	int k = 0;
	p0 = hull[0];
//找最下边的一个点
	for(int i = 1; i < n; i++)
	{
		if ( (p0.y > hull[i].y) || (p0.y == hull[i].y && p0.x > hull[i].x) )
		{
			p0 = hull[i];
			k = i;
		}
	}
	swap(hull[k],hull[0]);
	sort(hull+1,hull+n,_cmp);
	if (n == 1)
	{
		top = 1;
		Stack[0] = 0;
		return;
	}
	if (n == 2)
	{
		top = 2;
		Stack[0] = 0;
		Stack[1] = 1;
		return ;
	}
	Stack[0] = 0;
	Stack[1] = 1;
	top = 2;
	for (int i = 2; i < n; i++)
	{
		while (top > 1 && sgn((hull[Stack[top-1]]-hull[Stack[top-2]])^(hull[i]-hull[Stack[top-2]])) <=0)
			top--;
		Stack[top++] = i;
	}
}

///3、平面最近点对
int n;
Point p[N];
Point tmpt[N];
bool cmpy(Point a,Point b)
{
	return a.y < b.y;
}
double dfs(int left,int right)
{
	double d = INF;
	if (left == right)return d;
	if (left + 1 == right)
		return p[left]&p[right];
	int mid = (left+right)/2;
	double d1 = dfs(left,mid);
	double d2 = dfs(mid+1,right);
	d = min(d1,d2);
	int k = 0;
	for (int i = left; i <= right; i++)
	{
		if (fabs(p[mid].x - p[i].x) <= d)
			tmpt[k++] = p[i];
	}
	sort(tmpt,tmpt+k,cmpy);
	for (int i = 0; i <k; i++)
	{
		for (int j = i+1; j < k && tmpt[j].y - tmpt[i].y < d; j++)
		{
			d = min(d,(tmpt[i]&tmpt[j]));
		}
	}
	return d;
}
double Closest_Pair()
{
    sort(p,p+n);
    return Closest_Pair(0,n-1)/2;
}

///4、旋转卡壳
//4.1 求两点间距离平方的最大值
int dist2(Point a,Point b)
{
	return (a-b)*(a-b);
}
int rotating_calipers(Point p[],int n)
{
	int ans = 0;
	Point v;
	int cur = 1;
	for (int i = 0; i < n; i++)
	{
		v = p[i]-p[(i+1)%n];
		while ((v^(p[(cur+1)%n]-p[cur])) < 0)
			cur = (cur+1)%n;
		ans = max(ans,max(dist2(p[i],p[cur]),dist2(p[(i+1)%n],p[(cur+1)%n])));
	}
	return ans;
}
Point p[N]
int len(int n)
{
    Graham(n);
    for (int i = 0; i < top; i++)p[i] = hull[Stack[i]];
    return rotating_calipers(p,top);
}

//4.2 旋转卡壳计算平面点集最大三角形面积
int rotating_calipers(Point p[],int n)
{
	int ans = 0;
	Point v;
	for (int i = 0; i < n; i++)
	{
		int j = (i+1)%n;
		int k = (j+1)%n;
		while (j != i && k != i)
		{
			ans = max(ans,abs((p[j]-p[i])^(p[k]-p[i])));
			while ( ((p[i]-p[j])^(p[(k+1)%n]-p[k])) < 0 )
				k = (k+1)%n;
			j = (j+1)%n;
		}
	}
	return ans;
}
Point p[N];
double maxArea(int n)
{
    Graham(n);
    for (int i = 0; i < top; i++)p[i] = hull[Stack[i]];
    return (double)rotating_calipers(p,top)/2;
	return 0;
}

//4.3 求解两凸包最小距离
//点p0到线段p1p2的距离
Point PtoS(Point P,Line L)
{
	Point res;
	double t = ((P-L.s)*(L.e-L.s))/((L.e-L.s)*(L.e-L.s));
	if (t >= 0 && t <= 1)
	{
		res.x = L.s.x + (L.e.x - L.s.x)*t;
		res.y = L.s.y + (L.e.y - L.s.y)*t;
	}
	else
	{
		if( (P&L.s) < (P&L.e))	res = L.s;
		else                    res = L.e;
	}
	return res;
}
double pointtoseg(Point p0,Point p1,Point p2)
{
	return (p0&PtoS(p0,Line(p1,p2)));
}
//平行线段p0p1和p2p3的距离
double dispallseg(Point p0,Point p1,Point p2,Point p3)
{
	double ans1 = min(pointtoseg(p0,p2,p3),pointtoseg(p1,p2,p3));
	double ans2 = min(pointtoseg(p2,p0,p1),pointtoseg(p3,p0,p1));
	return min(ans1,ans2);
}
//得到向量a1a2和b1b2的位置关系
double Get_angle(Point a1,Point a2,Point b1,Point b2)
{
	return (a2-a1)^(b1-b2);
}
double rotating_calipers(Point p[],int np,Point q[],int nq)
{
	int sp = 0, sq = 0;
	for (int i = 0; i < np; i++)
		if (sgn(p[i].y - p[sp].y) < 0)
			sp = i;
	for (int i = 0; i < nq; i++)
		if (sgn(q[i].y - q[sq].y) > 0)
			sq = i;
	double tmp;
	double ans = (p[sp]&q[sq]);
	for (int i = 0; i < np; i++)
	{
		while (sgn(tmp = Get_angle(p[sp],p[(sp+1)%np],q[sq],q[(sq+1)%nq])) < 0)
			sq = (sq+1)%nq;
		if (sgn(tmp) == 0)
			ans = min(ans,dispallseg(p[sp],p[(sp+1)%np],q[sq],q[(sq+1)%nq]));
		else ans = min(ans,pointtoseg(q[sq],p[sp],p[(sp+1)%np]));
		sp = (sp+1)%np;
	}
	return ans;
}
double solve(Point p[],int n,Point q[],int m)
{
	return min(rotating_calipers(p,n,q,m),rotating_calipers(q,m,p,n));
}
Point p[MAXN],q[MAXN];
double len(int n,int m)
{
    for (int i = 0; i < n; i++)scanf("%lf%lf",&hull[i].x,&hull[i].y);
    Graham(n);
    for (int i = 0; i < top; i++)p[i] = hull[i];
    n = top;
    for (int i = 0; i < m; i++)scanf("%lf%lf",&hull[i].x,&hull[i].y);
    Graham(m);
    for (int i = 0; i < top; i++)q[i] = hull[i];
    m = top;
    return solve(p,n,q,m);
}

///6、圆
//*过三点求圆心坐标（三角形外心）
Point waixin(Point a,Point b,Point c)
{
	double a1 = b.x - a.x, b1 = b.y - a.y, c1 = (a1*a1 + b1*b1)/2;
	double a2 = c.x - a.x, b2 = c.y - a.y, c2 = (a2*a2 + b2*b2)/2;
	double d = a1*b2 - a2*b1;
	return Point(a.x + (c1*b2 - c2*b1)/d, a.y + (a1*c2 -a2*c1)/d);
}

//*两个圆的公共部分面积
double Area_of_overlap(Point c1,double r1,Point c2,double r2)
{
	double d = dist(c1,c2);
	if (r1 + r2 < d + eps)return 0;
	if (d < fabs(r1 - r2) + eps)
	{
		double r = min(r1,r2);
		return PI*r*r;
	}
	double x = (d*d + r1*r1 - r2*r2)/(2*d);
	double t1 = acos(x / r1);
	double t2 = acos((d - x)/r2);
	return r1*r1*t1 + r2*r2*t2 - d*r1*sin(t1);
}

///三角形公式
//三角形面积公式
p=(a+b+c)/2;//半周长
s=sqrt(p*(p-a)*(p-b)*(p-c));

//三角形内切圆半径公式
r=(2*s)/(a+b+c);
r=(sqrt((a+b+c)*(b+c-a)*(a+c-b)*(a+b-c))/2)/(a+b+c);

//三角形外接圆半径公式
R=(a*b*c)/(4*s);
R=(a*b*c/sqrt((a+b+c)*(b+c-a)*(a+c-b)*(a+b-c)));

//圆内接四边形面积公式
p=(a+b+c+d)/2;
s=sqrt((p-a)*(p-b)*(p-c)*(p-d));

//三分求极值
//三等分求图形函数极值
//求(x,y)到y=a*x^2+b*x+c的最短距离；
int a,b,c,x,y;
double dis(double now)
{
    return sqrt((now-x)*(now-x)+(a*now*now+b*now+c-y)*(a*now*now+b*now+c-y));
}
int main()
{
    scanf("%d%d%d%d%d",&a,&b,&c,&x,&y);
    double l,r,mr,ml;
    l=-200;r=200;
    for(int i=0;i<100;i++)
    {
        ml=(l+l+r)/3;
        mr=(r+r+l)/3;
        if(dis(ml)>dis(mr))
            l=ml;
        else
            r=mr;
    }
    printf("%.3lf\n",dis(l));
    return 0;
}

///博弈论！！！！！
///巴什博弈：
//二人轮流取一堆n个石子一次取1到m个，问谁先取完
//必败态：n=k(m+1)，每轮共取m+1个。
if(n%(m+1)==0)printf("second win\n");

///威佐夫博弈：
//二人轮流取2堆a，b个石子，一次取其中一堆任意个或两堆相同数量个，问谁先取完
//必败态：a=k(1+sqrt(5))/2,b=a+k;即a为黄金分割数向下取整，
int ep1=(sqrt(5.0)-1.0)/2.0,ep2=(sqrt(5.0)+1.0)/2.0;
int k=x*ep1;//求倍数
int tmp1=ep2*k,tmp2=k+tmp1;//判断是否为奇异局势
int tmp3=ep2*(k+1),tmp4=k+1+tmp3;//求倍数时x的小数被抹去可能求值小1；
if(tmp1==x&&tmp2==y||tmp3==x&&tmp4==y)printf("second win\n");

///nim博弈：
//取n堆石子，一次从一堆中取任意个
//必败态：a1^a2^a3^an=0,即各堆数量按位异或为0，转换为二进制之后每位相同数均为偶数。
int a[N];
int nim(int n)
{
    int cnt=0;
    for(int i=0;i<n;i++)
        cnt^=a[i];
    if(cnt==0)return 0;
    return 1;
}
//当取最后一个者输时：
int a[N];
int nim(int n)
{
    int cnt=0,now=0;
    for(int i=0;i<n;i++)
    {
        cnt^=a[i];
        if(a[i]>1)now++;
    }
    if(n%2==0&&now==0 || cnt&&now)return 1;
    return 0;
}

///k倍减法博弈：
//二人轮流取一堆石子，后一人取得数量不得超多迁移人数量的k倍；
//k=1时：
//转换为二进制取最后一个1，则后一人无法取到下一个1，必胜(若只有一个1必败)
//k=2时：
//即斐波那契博弈：必败态：斐波纳契数列；
//将非斐波纳契数写成两个不连续的斐波纳契数之和，取后一个数。
//面对非斐波纳契数时必胜；
//k>2时：
//类似于斐波纳契博弈，根据k自行构造数列，
//逆向思维，a[0~i]数列最大可以构造数为b[i],则a[i+1]=b[i]+1,b[i]=a[i]+b[t](a[t]*k<a[i]);
a[0]=b[0]=1;
while (a[i]<n)//构造数列
{
    i++;
    a[i]=b[i-1]+1;
    while (a[j+1]*k<a[i])
        j++;
    if (a[j]*k<a[i])
        b[i]=a[i]+b[j];
    else
        b[i]=a[i];
}

///sg函数：
//sg(x)=mex(f(y)|y是x的后继），即取后继节点集合中没有的最小非负整数
//必败态：sg(x)=0
//由sg函数可求组合游戏，即sum=sg(1)^sg(2)^sg(n),参考nim博弈。
//f[]：可以取走的石子个数
//sg[]:0~n的SG函数值
//mex[]:mex{}
int n,m,ans;
int f[N],sg[M];
int SG_dfs(int x)
{
    int i;
    if(sg[x]!=-1)
        return sg[x];
    bool mex[N];
    memset(mex,0,sizeof(mex));
    for(i=0;i<n&&x>=f[i];i++)
    {
        if(sg[x-f[i]]==-1)
            SG_dfs(x-f[i]);
        mex[sg[x-f[i]]]=1;
    }
    int e;
    for(i=0;;i++)
        if(!mex[i])
            return sg[x]=i;
}
void solve()
{
    for(int i=0;i<n;i++)
        scanf("%d",&f[i]);
    sort(f,f+n);

    memset(sg,-1,sizeof(sg));
    sg[0]=0;

    scanf("%d",&m);
    ans=0;
    while(m--)
    {
        scanf("%d",&t);
        sg[t]=SG_dfs(t);
        ans^=sg[t];
    }
    if(ans)return 1;
    else return 0;
}
//f[]：可以取走的石子个数
//sg[]:0~n的SG函数值
//mex[]:mex{}
int n;
int f[N],sg[M],mex[N];
void getSG(int n)
{
    int i,j;
    memset(sg,0,sizeof(sg));
    for(i=0;i<n;i++)
    {
        memset(mex,0,sizeof(mex));
        for(j=0;f[j]<=i;j++)
            mex[sg[i-f[j]]]=1;
        for(j=0;j<=n;j++)    //求mes{}中未出现的最小的非负整数
        {
            if(mex[j]==0)
            {
                sg[i]=j;
                break;
            }
        }
    }
}
void solve()
{
    //给出f[]；
    scanf("%d",&n);
    for(int i=0;i<n;i++)
        scanf("%d",&f[i]);
    getSG(N-1);
}

///header
//#include <bits/stdc++.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <limits.h>
#include <iostream>
#include <sstream>
#include <string>
#include <stack>
#include <queue>
#include <vector>
#include <set>
#include <map>
#include <bitset>
#include <algorithm>

#define PI acos(-1.0)
#define INF 0x3f3f3f3f
#define MOD 1000000007
#define EPS 1e-6
#define N 112345
#define root 1 , n , 1
#define lson l , m , rt << 1
#define rson m + 1 , r , rt << 1 | 1
using namespace std;
struct node
{
    int x,y;
    friend bool operator < (node a, node b)
    {
        return a.x > b.x;
    }
};
int n,m,res,sum,flag;
int main()
{
    int i,j,k,kk,t,x,y,z;
    clock_t start,finish;
    #ifndef ONLINE_JUDGE
        freopen("test.txt","r",stdin);
    #endif
    start = clock();
//    scanf("%d",&k);
//    kk=0;
//    while(k--)
    while(scanf("%d",&n)!=EOF&&n)
    {
    }
    finish = clock();
    printf("time=%lfs\n",(double)(finish-start)/1000);
    return 0;
}
