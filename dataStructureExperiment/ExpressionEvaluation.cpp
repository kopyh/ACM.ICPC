//kopyh
#include<iostream>
#include<cstdio>
#include<string>
#define EPS 1e-6
using namespace std;

class stack
{
private:
    struct node
    {
        double num;
        char symbol;
        node *next;
    };
    node *topNode=NULL;
    int sum=0;
public:
    void init()
    {
        topNode=NULL;
        sum=0;
    }
    bool empty()
    {
        return sum==0;
    }
    void push(double num, char symbol='#')
    {
        node *a = new(node);
        a->num = num;
        a->symbol = symbol;
        a->next = topNode;
        topNode = a;
        sum++;
    }
    node top()
    {
        return *topNode;
    }
    void pop()
    {
        node *t = topNode;
        topNode = topNode->next;
        sum--;
        delete(t);
    }
    int size()
    {
        return sum;
    }
};

int symbolCheck[10][10]={
{1,1,0,0,0,0,1,1},
{1,1,0,0,0,0,1,1},
{1,1,1,1,1,0,1,1},
{1,1,1,1,1,0,1,1},
{1,1,1,1,1,0,1,1},
{0,0,0,0,0,0,2,2},
{1,1,1,1,1,2,1,1},
{0,0,0,0,0,0,2,2}
};
int symbolCheck2[10][10]={
{1,1,0,0,0,1,0,1},
{1,1,0,0,0,1,0,1},
{1,1,1,1,1,1,0,1},
{1,1,1,1,1,1,0,1},
{1,1,1,1,1,1,0,1},
{1,1,1,1,1,1,2,2},
{0,0,0,0,0,2,0,1},
{0,0,0,0,0,2,0,2}
};
int getSymbol(char c)
{
    if(c=='+')return 0;
    if(c=='-')return 1;
    if(c=='*')return 2;
    if(c=='/')return 3;
    if(c=='%')return 4;
    if(c=='(')return 5;
    if(c==')')return 6;
    if(c=='#')return 7;
    if(c=='.')return 8;
    return 9;
}
double calculate(double x, double y, char c)
{
    if(c=='+')return x+y;
    if(c=='-')return x-y;
    if(c=='*')return x*y;
    if(c=='/')return x/y;
    if(c=='%')return 1.0*((int)(x+EPS)%(int)(y+EPS));
}

double infixCal(string s)
{
    stack stackDigitial,stackSymbol;
    stackDigitial.init();
    stackSymbol.init();
    stackSymbol.push(0,'#');
    char ss[101];
    int i=0,j=0;
    while(i<s.size())
    {
        if(getSymbol(s[i])==9)
        {
            j=0;
            if(s[i]==' '){i++;continue;}
            for(j=0; s[i]!=' ' && getSymbol(s[i])>=8 && i<s.size(); ss[j++]=s[i++]);
            ss[j]='\0';
            double t;
            sscanf(ss,"%lf",&t);
            stackDigitial.push(t);
        }
        else
        {
            int now = symbolCheck[getSymbol(stackSymbol.top().symbol)][getSymbol(s[i])];
            if(now==1)
            {
                double y=stackDigitial.top().num;
                stackDigitial.pop();
                double x=stackDigitial.top().num;
                stackDigitial.pop();
                char ct=stackSymbol.top().symbol;
                stackSymbol.pop();
                stackDigitial.push(calculate(x,y,ct));
            }
            else if(now==0)
            {
                stackSymbol.push(0,s[i]);
                i++;
            }
            else if(now==2)
            {
                stackSymbol.pop();
                i++;
            }
        }
    }
    while(stackSymbol.size()>1)
    {
        double y=stackDigitial.top().num;
        stackDigitial.pop();
        double x=stackDigitial.top().num;
        stackDigitial.pop();
        char ct=stackSymbol.top().symbol;
        stackSymbol.pop();
        stackDigitial.push(calculate(x,y,ct));
    }
    return stackDigitial.top().num;
}
double suffixCal(string s)
{
    stack stackDigitial;
    stackDigitial.init();
    char ss[101];
    int i=0,j=0;
    while(i<s.size())
    {
        if(getSymbol(s[i])==9)
        {
            if(s[i]==' '){i++;continue;}
            for(j=0; s[i]!=' ' && getSymbol(s[i])>=8 && i<s.size(); ss[j++]=s[i++]);
            ss[j]='\0';
            double t;
            sscanf(ss,"%lf",&t);
            stackDigitial.push(t);
        }
        else
        {
            double y=stackDigitial.top().num;
            stackDigitial.pop();
            double x=stackDigitial.top().num;
            stackDigitial.pop();
            stackDigitial.push(calculate(x,y,s[i++]));
        }
    }
    return stackDigitial.top().num;
}
double prefixCal(string s)
{
    stack stackNode;
    stackNode.init();
    stackNode.push(0,'#');
    char sss[101];
    int i=0,j=0;
    while(i<s.size())
    {
        if(getSymbol(s[i])==9)
        {
            if(s[i]==' '){i++;continue;}
            for(j=0; s[i]!=' ' && getSymbol(s[i])>=8 && i<s.size(); sss[j++]=s[i++]);
            sss[j]='\0';
            double t;
            sscanf(sss,"%lf",&t);
            stackNode.push(t);
        }
        else
        {
            char c = s[i++];
            stackNode.push(0,c);
        }
    }
    stack ans;
    ans.init();
    while(stackNode.size()>1)
    {
        if(stackNode.top().symbol=='#')
        {
            ans.push(stackNode.top().num);
            stackNode.pop();
        }
        else
        {
            double x=ans.top().num;
            ans.pop();
            double y=ans.top().num;
            ans.pop();
            ans.push(calculate(x,y,stackNode.top().symbol));
            stackNode.pop();
        }
    }
    return ans.top().num;
}
string infixToSuffix(string s)
{
    stack stackSymbol;
    stackSymbol.init();
    stackSymbol.push(0,'#');
    string ss;
    int i=0,j=0;
    while(i<s.size())
    {
        if(getSymbol(s[i])==9)
        {
            if(s[i]==' '){i++;continue;}
            while(s[i]!=' ' && getSymbol(s[i])>=8 && i<s.size())ss+=s[i++];
            ss+=" ";
        }
        else
        {
            int now = symbolCheck[getSymbol(stackSymbol.top().symbol)][getSymbol(s[i])];
            if(now==1)
            {
                ss+=stackSymbol.top().symbol;
                ss+=" ";
                stackSymbol.pop();
            }
            else if(now==0)
            {
                stackSymbol.push(0,s[i]);
                i++;
            }
            else if(now==2)
            {
                stackSymbol.pop();
                i++;
            }
        }
    }
    while(stackSymbol.size()>1)
    {
        ss+=stackSymbol.top().symbol;
        ss+=" ";
        stackSymbol.pop();
    }
    return ss;
}
string infixToPrefix(string s)
{
    stack stackNode;
    stackNode.init();
    stackNode.push(0,'#');
    char sss[101];
    int i=0,j=0;
    while(i<s.size())
    {
        if(getSymbol(s[i])==9)
        {
            if(s[i]==' '){i++;continue;}
            for(j=0; s[i]!=' ' && getSymbol(s[i])>=8 && i<s.size(); sss[j++]=s[i++]);
            sss[j]='\0';
            double t;
            sscanf(sss,"%lf",&t);
            stackNode.push(t);
        }
        else
        {
            char c = s[i++];
            stackNode.push(0,c);
        }
    }
    stack stackSymbol,ans;
    stackSymbol.init();
    ans.init();
    stackSymbol.push(0,'#');
    while(stackNode.size()>1)
    {
        char c = stackNode.top().symbol;
        if(c == '#')
        {
            ans.push(stackNode.top().num);
            stackNode.pop();
        }
        else
        {
            int now = symbolCheck2[getSymbol(stackSymbol.top().symbol)][getSymbol(c)];
            if(now==1)
            {
                ans.push(0,stackSymbol.top().symbol);
                stackSymbol.pop();
            }
            else if(now==0)
            {
                stackSymbol.push(0,c);
                stackNode.pop();
            }
            else if(now==2)
            {
                stackSymbol.pop();
                stackNode.pop();
            }
        }
    }
    while(stackSymbol.size()>1)
    {
        ans.push(0,stackSymbol.top().symbol);
        stackSymbol.pop();
    }
    string ss;
    while(!ans.empty())
    {
        if(ans.top().symbol=='#')sprintf(sss,"%d",(int)ans.top().num);
        else sss[0]=ans.top().symbol,sss[1]='\0';
        for(int i=0;sss[i]!='\0';i++)ss+=sss[i];
        ans.pop();
        ss+=" ";
    }
    return ss;
}

int main()
{
    string s,s1,s2;
    while(getline(cin,s,'\n'))
    {
        s1=infixToSuffix(s);
        s2=infixToPrefix(s);
        printf("infix:  %s\n",s.c_str());
        printf("answer = %.2lf\n",infixCal(s));
        printf("suffix: %s\n",s1.c_str());
        printf("answer = %.2lf\n",suffixCal(s1));
        printf("prefix: %s\n",s2.c_str());
        printf("answer = %.2lf\n",prefixCal(s2));
        printf("\n");
    }
    return 0;
}
