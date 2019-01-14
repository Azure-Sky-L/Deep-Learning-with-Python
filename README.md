#### 深度学习基础：
- 损失函数：用来衡量网络输出结果的质量

+ 优化器：将损失值作为反馈信号来调节权重

- SVM: 目标是通过在属于两个不同类别的两组数据点之间找到良好决策边界(decision boundary)来解决分类问题

+  SVM 通过两步来寻找决策边界:
 1. 将数据映射到一个新的高维表示,这时决策边界可以用一个超平面来表示(如果数据是二维的,那么超平面就是一条直线)
 2. 尽量让超平面与每个类别最近的数据点之间的距离最大化,从而计算出良好决策边界(分割超平面),这一步叫作间隔最大化(maximizing the margin)

* logitic 不是回归是分类

- 降维(dimensionality reduction)和聚类(clustering)都是众所周知的无监督学习方法

+ 三种经典的评估方法:简单的留出验证、K 折验证,以及带有打乱数据的重复 K 折验证

- 数据预处理的目的是使原始数据更适于用神经网络处理,包括向量化、标准化、处理缺失值和特征提取

+ 防止神经网络过拟合的常用方法包括: 获取更多的训练数据、减小网络容量、添加权重正则化、添加 dropout

-  ![](./webwxgetmsgimg.jpeg)

+ 使用预训练网络有两种方法:特征提取(feature extraction)和微调模型(fine-tuning) 

- 冻结(freeze)一个或多个层是指在训练过程中保持其权重不变。(如果不这么做,那么卷积基之前学到的表示将会在训练过程中被修改。因为其上添加的 Dense 层是随机初始化的,所以非常大的权重更新将会在网络中传播,对之前学到的表示造成很大破坏)
