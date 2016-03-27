//kopyh
#include <bits/stdc++.h>
using namespace std;
//˫������ڵ�洢ÿ���˵ı��
struct node
{
    int num;
    node *next, *prior;
    node(int x, node *p = NULL, node *q = NULL):num(x),next(p),prior(q){}
};
//��ʼ���������γ�һ����
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
//���������
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
