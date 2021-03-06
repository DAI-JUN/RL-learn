## 《强化学习（第二版）》例10.2访问控制队列任务  
这是一个涉及一组k个服务器的访问控制的决策任务，四个不同优先级的客户，到达一个队列。如果让客户访问服务器，客户会根据他们的优先级向服务器支付1、2、4或8的收益，优先级更高的客户支付更多。在每个时刻中，队列头的客户，要么被接受（分配给他一个服务器），要么被拒绝（从队列中移走，服务器收益为0），无论是这两种情况的哪一种，下一个时刻均考虑队列中的下一个客户。队列从不被清空，队列中客户的优先级也是等概率随机分布的。当然，如果没有空闲的服务器，客户就不能被服务，而是被拒绝。每一个忙碌的服务器，在每个时刻终有p=0.06的概率变为空闲，虽然我们刚刚对这个任务给出了确定的描述，但我们假设客户到达和离开的统计量是未知的。客户在每个时刻根据优先级和空闲服务器的数量决定是否接受或拒绝下一个客户，以期最大化无折扣的长期收益。  
### 解法  
在这个例子中，我们考虑一种表格型的解决方法，对于每一个“状态-动作”二元组（状态是队列中空闲服务器的数量和客户的优先级，动作是接受或拒绝），我们都有一个差分动作价值估计。

我们使用半梯度Sarsa算法，k=10，p=0.06

这个算法中α=0.01，β=0.01以及ε=0.1.初始的动作价值和平均价值都是0
### 算法（差分半梯度Sarsa算法）  
算法参数：

价值函数q: S x A → R

步长α，β>0

平均回报R_mean∈R

初始化状态S和动作A

对于每一步循环：

    采取动作A，观察R，S`
    通过q(S`,·)选取A`(ε-贪心策略)
    δ ← R - R_mean + q(S`,A`) - q(S,A)
    R_mean ← R_mean + βδ
    q(S,A) ← q(S,A) + αδ
    S ← S`
    A ← A`
### 代码说明  
类Env 模拟环境

类valuef 为价值函数q

输出为最终策略图
