//kopyh
#include <bits/stdc++.h>
using namespace std;
//双向链表节点存储每个人的编号
struct node
{
    int num;
    node *next, *prior;
    node(int x, node *p = NULL, node *q = NULL):num(x),next(p),prior(q){}
};
//初始化所有人形成一个环
void init(int n, node *start)
{
    int i=2;
    node *tmp = start;
    while(i<=n)
    {
        tmp->next = new node(i++);
        tmp->next->prior = tmp;
        tmp = tmp->next;
    }
    tmp->next = start;
    start->prior = tmp;
}
//按序输出答案
void solve(int n, int m, node *start)
{
    int i=0,j=1;
    while(i<n)
    {
        if(j == m)
        {
            if(i!=0)printf(" -> ");
            printf("%d",start->num);
            start->prior->next = start->next;
            start->next->prior = start->prior;
            start = start->next;
            j = 1;
            i++;
        }
        else
            start = start->next,j++;
    }
    printf("\n");
}
int main()
{
    int n,m;
    node *start = new node(1);
    printf("Please input the number of people and the number of password:\n");
    scanf("%d%d",&n,&m);
    init(n,start);
    printf("Output order is:\n");
    solve(n,m,start);
    return 0;
}
