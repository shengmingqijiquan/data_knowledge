---

title: 机器学习系列之决策树算法（07）：梯度提升树算法XGBoost
date: 2019.12.25
tags: 

	- XGBoost 

categories: 

	- Machine Learning

	- XGBoost 

keywords: XGBoost 
description: XGBoost 

---

## 前言

XGBoost的全称是eXtreme Gradient Boosting，它是经过优化的分布式梯度提升库，旨在高效、灵活且可移植。XGBoost是大规模并行boosting tree的工具，它是目前最快最好的开源 boosting tree工具包，比常见的工具包快10倍以上。在数据科学方面，有大量的Kaggle选手选用XGBoost进行数据挖掘比赛，是各大数据科学比赛的必杀武器；在工业界大规模数据方面，XGBoost的分布式版本有广泛的可移植性，支持在Kubernetes、Hadoop、SGE、MPI、 Dask等各个分布式环境上运行，使得它可以很好地解决工业界大规模数据的问题。本文将从XGBoost的数学原理和工程实现上进行介绍，然后介绍XGBoost的优缺点。

## 数学原理

### 生成一棵树

#### **Boosting Tree回顾**

XGBoost模型是大规模并行boosting tree的工具，它是目前较好的开源boosting tree工具包。因此，在了解XGBoost算法基本原理之前，需要首先了解Boosting Tree算法基本原理。Boosting方法是一类应用广泛且非常有效的统计学习方法。它是基于这样一种思想：对于一个复杂任务来说，将多个专家的判断进行适当的综合所得出的判断，要比任何一个专家单独的判断要好。这种思想整体上可以分为两种：

- **强可学习**：如果存在一个多项式的学习算法能够学习它，并且正确率很高，那么就称为强可学习，直接单个模型就搞定常规问题。就好比专家给出的意见都很接近且都是正确率很高的结果，那么一个专家的结论就可以用了，这种情况非常少见。
- **弱可学习**：如果存在一个多项式的学习算法能够学习它，学习的正确率仅比随机猜测略好，那么就称这个概念是弱可学习的。这种情况是比较常见的。

boosting算法主要是针对弱可学习的分类器来开展优化工作。其关心的问题包括两方面内容：

（1）在每一轮如何改变训练数据的权值和概率分布；

（2）如何将弱分类器组合成一个强分类器，这种思路较好的就是AdaBoost算法，以前在遥感图像地物识别中得到过应用。

![Boosting模型基本流程](https://pic2.zhimg.com/80/v2-35e3fd2bd53b5cbfc2f9596eb2479591_hd.jpg)



Boosting Tree模型采用**加法模型**与**前向分步算法**，而且基模型都是决策树模型。前向分步算法（Forward stage wise additive model）是指在叠加新的基模型的基础上同步进行优化，具体而言，就是每一次叠加的模型都去拟合上一次模型拟合后产生的残差（Residual）。从算法模型解释上来说，Boosting Tree是决策树的加法模型：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+f_%7BM%7D%28x%29+%3D+%5Csum_%7Bm%3D1%7D%5E%7BM%7DT%28x%2C%5Ctheta_%7Bm%7D%29+%5Cend%7Bequation%7D) （1）

上式中M为决策树的数量； ![[公式]](https://www.zhihu.com/equation?tex=T%28x%2C+%5Ctheta_%7Bm%7D%29) 为某个决策树； ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bm%7D) 为对应决策树的参数。

Boosting Tree模型采用前向分步算法，其中假设 ![[公式]](https://www.zhihu.com/equation?tex=f_%7B0%7D%28x%29+%3D+0) ，则第m步的模型是：

![[公式]](https://www.zhihu.com/equation?tex=f_%7Bm%7D%28x%29+%3D+f_%7Bm-1%7D%28x%29%2BT%28x%2C+%5Ctheta_%7Bm%7D%29) （2）

为求解对应的参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bm%7D) ，需要最小化相应损失函数来确定，具体公式如下：

![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bm%7D%5E%7B%27%7D+%3D+arg+%5Cmin_%7B%5Ctheta_%7Bm%7D%7D%5Csum_%7Bi%3D1%7D%5E%7BM%7DL%28y_%7Bi%7D%2C+f_%7Bm-1%7D%28x_%7Bi%7D%29+%2B+T%28x_%7Bi%7D%3B%5Ctheta_%7Bm%7D%29%29) （3）

由前向分步算法得到M棵决策树![[公式]](https://www.zhihu.com/equation?tex=T%28x%2C+%5Ctheta_%7Bm%7D%29) 后，再进行加和，就得到了提升树模型 ![[公式]](https://www.zhihu.com/equation?tex=f_%7BM%7D%28x%29) 。在xgboost论文中提到的一个明显的boosting tree的加和应用案例如图3所示。

![img](https://pic2.zhimg.com/80/v2-fee9ec17376a633196bebbf56c18c2f5_hd.jpg)图2 boosting tree的累加效果示意图

> 相关树模型的参数值求解主要依据于**损失函数**的定义。
>
> 一般来言对于**分类问题**，选择**指数损失函数**作为损失函数时，将形成**AdaBoost模型**；
>
> 对于**回归问题**，损失函数常利用**平方损失函数**。为了扩展Boosting Tree的应用范围，需要构建一种可以广泛适用的残差描述方式来满足于任意损失函数的形式，为解决分类问题的Gradient Boosting Decision Tree算法应运而生。

**[带正则项的Boosting Tree模型和带梯度的Boosting Tree推导过程](https://zhuanlan.zhihu.com/p/90520307)**

#### 目标函数

我们知道 XGBoost 是由 ![[公式]](https://www.zhihu.com/equation?tex=k) 个基模型组成的一个加法运算式：

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%3D%5Csum_%7Bt%3D1%7D%5E%7Bk%7D%5C+f_t%28x_i%29+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=f_k) 为第 ![[公式]](https://www.zhihu.com/equation?tex=k) 个基模型， ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i) 为第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个样本的预测值。

损失函数可由预测值 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i) 与真实值 ![[公式]](https://www.zhihu.com/equation?tex=y_i) 进行表示：

![[公式]](https://www.zhihu.com/equation?tex=L%3D%5Csum_%7Bi%3D1%7D%5En+l%28+y_i%2C+%5Chat%7By%7D_i%29+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=n) 为样本数量。

我们知道模型的预测精度由模型的偏差和方差共同决定，损失函数代表了模型的偏差，想要方差小则需要简单的模型，所以目标函数由模型的**损失函数 ![[公式]](https://www.zhihu.com/equation?tex=L)** 与**抑制模型复杂度的正则项 ![[公式]](https://www.zhihu.com/equation?tex=%5COmega)** 组成，所以我们有：

![[公式]](https://www.zhihu.com/equation?tex=Obj+%3D%5Csum_%7Bi%3D1%7D%5En+l%28%5Chat%7By%7D_i%2C+y_i%29+%2B+%5Csum_%7Bt%3D1%7D%5Ek+%5COmega%28f_t%29+%5C%5C+)

![[公式]](https://www.zhihu.com/equation?tex=%5COmega) 为模型的正则项，由于 XGBoost 支持决策树也支持线性模型，所以这里再不展开描述。

我们知道 boosting 模型是前向加法，以第 ![[公式]](https://www.zhihu.com/equation?tex=t) 步的模型为例，模型对第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个样本 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D) 的预测为：

![[公式]](https://www.zhihu.com/equation?tex=++%5Chat%7By%7D_i%5Et%3D+%5Chat%7By%7D_i%5E%7Bt-1%7D+%2B+f_t%28x_i%29++%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 由第 ![[公式]](https://www.zhihu.com/equation?tex=t-1) 步的模型给出的预测值，是已知常数，![[公式]](https://www.zhihu.com/equation?tex=f_t%28x_i%29) 是我们这次需要加入的新模型的预测值，此时，目标函数就可以写成：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Balign%7D+Obj%5E%7B%28t%29%7D+%26%3D+%5Csum_%7Bi%3D1%7D%5Enl%28y_i%2C+%5Chat%7By%7D_i%5Et%29+%2B+%5Csum_%7Bi%3D1%7D%5Et%5COmega%28f_i%29+%5C%5C++++%26%3D+%5Csum_%7Bi%3D1%7D%5En+l%5Cleft%28y_i%2C+%5Chat%7By%7D_i%5E%7Bt-1%7D+%2B+f_t%28x_i%29+%5Cright%29+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29++%5Cend%7Balign%7D+%5C%5C)

求此时最优化目标函数，就相当于求解 ![[公式]](https://www.zhihu.com/equation?tex=f_t%28x_i%29) 。

> 泰勒公式是将一个在 ![[公式]](https://www.zhihu.com/equation?tex=x%3Dx_0) 处具有 ![[公式]](https://www.zhihu.com/equation?tex=n) 阶导数的函数 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 利用关于 ![[公式]](https://www.zhihu.com/equation?tex=x-x_0) 的 ![[公式]](https://www.zhihu.com/equation?tex=n) 次多项式来逼近函数的方法，若函数 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 在包含 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 的某个闭区间 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%5D) 上具有 ![[公式]](https://www.zhihu.com/equation?tex=n) 阶导数，且在开区间 ![[公式]](https://www.zhihu.com/equation?tex=%28a%2Cb%29) 上具有 ![[公式]](https://www.zhihu.com/equation?tex=n%2B1) 阶导数，则对闭区间 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%5D) 上任意一点 ![[公式]](https://www.zhihu.com/equation?tex=x) 有 ![[公式]](https://www.zhihu.com/equation?tex=%5Cdisplaystyle+f%28x%29%3D%5Csum_%7Bi%3D0%7D%5E%7Bn%7D%5Cfrac%7Bf%5E%7B%28i%29%7D%28x_0%29%7D%7Bi%21%7D%28x-x_0%29%5E+i%2BR_n%28x%29+) ，其中的多项式称为函数在 ![[公式]](https://www.zhihu.com/equation?tex=x_0) 处的泰勒展开式， ![[公式]](https://www.zhihu.com/equation?tex=R_n%28x%29) 是泰勒公式的余项且是 ![[公式]](https://www.zhihu.com/equation?tex=%28x%E2%88%92x_0%29%5En) 的高阶无穷小。

根据泰勒公式我们把函数 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%2B%5CDelta+x%29) 在点 ![[公式]](https://www.zhihu.com/equation?tex=x) 处进行泰勒的二阶展开，可得到如下等式：

![[公式]](https://www.zhihu.com/equation?tex=f%28x%2B%5CDelta+x%29+%5Capprox+f%28x%29+%2B+f%27%28x%29%5CDelta+x+%2B+%5Cfrac12+f%27%27%28x%29%5CDelta+x%5E2++%5C%5C)

我们把 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 视为 ![[公式]](https://www.zhihu.com/equation?tex=x) ， ![[公式]](https://www.zhihu.com/equation?tex=f_t%28x_i%29) 视为 ![[公式]](https://www.zhihu.com/equation?tex=%5CDelta+x) ，故可以将目标函数写为：

![[公式]](https://www.zhihu.com/equation?tex=Obj%5E%7B%28t%29%7D+%3D+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+l%28y_i%2C+%5Chat%7By%7D_i%5E%7Bt-1%7D%29+%2B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29+%5C%5C)

其中 ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bi%7D) 为损失函数的一阶导， ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bi%7D) 为损失函数的二阶导，**注意这里的导是对 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 求导**。

我们以平方损失函数为例：

![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bi%3D1%7D%5En+%5Cleft%28y_i+-+%28%5Chat%7By%7D_i%5E%7Bt-1%7D+%2B+f_t%28x_i%29%29+%5Cright%29%5E2++%5C%5C)

则：

![[公式]](https://www.zhihu.com/equation?tex=++%5Cbegin%7Balign%7D++++++g_i+%26%3D+%5Cfrac%7B%5Cpartial+%28%5Chat%7By%7D%5E%7Bt-1%7D+-+y_i%29%5E2%7D%7B%5Cpartial+%7B%5Chat%7By%7D%5E%7Bt-1%7D%7D%7D+%3D+2%28%5Chat%7By%7D%5E%7Bt-1%7D+-+y_i%29+%5C%5C++++++h_i+%26%3D%5Cfrac%7B%5Cpartial%5E2%28%5Chat%7By%7D%5E%7Bt-1%7D+-+y_i%29%5E2%7D%7B%7B%5Chat%7By%7D%5E%7Bt-1%7D%7D%7D+%3D+2++++%5Cend%7Balign%7D++%5C%5C)

由于在第 ![[公式]](https://www.zhihu.com/equation?tex=t) 步时 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_i%5E%7Bt-1%7D) 其实是一个已知的值，所以 ![[公式]](https://www.zhihu.com/equation?tex=l%28y_i%2C+%5Chat%7By%7D_i%5E%7Bt-1%7D%29) 是一个常数，其对函数的优化不会产生影响，因此目标函数可以写成：

![[公式]](https://www.zhihu.com/equation?tex=+Obj%5E%7B%28t%29%7D+%5Capprox+%5Csum_%7Bi%3D1%7D%5En+%5Cleft%5B+g_if_t%28x_i%29+%2B+%5Cfrac12h_if_t%5E2%28x_i%29+%5Cright%5D+%2B+%5Csum_%7Bi%3D1%7D%5Et++%5COmega%28f_i%29+%5C%5C)

所以**我们只需要求出每一步损失函数的一阶导和二阶导的值（由于前一步的 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D%5E%7Bt-1%7D) 是已知的，所以这两个值就是常数），然后最优化目标函数，就可以得到每一步的 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) ，最后根据加法模型得到一个整体模型。**

#### 基于决策树的目标函数

损失函数可由预测值 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_%7Bi%7D) 与真实值 ![[公式]](https://www.zhihu.com/equation?tex=y_%7Bi%7D) 进行表示：

![[公式]](https://www.zhihu.com/equation?tex=L+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bl%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%29%7D+%5C%5C)

其中， ![[公式]](https://www.zhihu.com/equation?tex=n) 为样本的数量。

我们知道模型的预测精度由模型的偏差和方差共同决定，损失函数代表了模型的偏差，想要方差小则需要在目标函数中添加正则项，用于防止过拟合。所以目标函数由模型的损失函数 ![[公式]](https://www.zhihu.com/equation?tex=L) 与抑制模型复杂度的正则项 ![[公式]](https://www.zhihu.com/equation?tex=%5COmega) 组成，目标函数的定义如下：

![[公式]](https://www.zhihu.com/equation?tex=Obj+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bl%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%29%7D+%2B+%5Csum_%7Bi%3D1%7D%5E%7Bt%7D%7B%5COmega%28f_%7Bi%7D%29%7D+%5C%5C)

其中，![[公式]](https://www.zhihu.com/equation?tex=+%5Csum_%7Bi%3D1%7D%5E%7Bt%7D%7B%5COmega%28f_%7Bi%7D%29%7D) 是将全部 ![[公式]](https://www.zhihu.com/equation?tex=t) 棵树的复杂度进行求和，添加到目标函数中作为正则化项，用于防止模型过度拟合。

由于XGBoost是boosting族中的算法，所以遵从前向分步加法，以第 ![[公式]](https://www.zhihu.com/equation?tex=t) 步的模型为例，模型对第 ![[公式]](https://www.zhihu.com/equation?tex=i) 个样本 ![[公式]](https://www.zhihu.com/equation?tex=x_%7Bi%7D) 的预测值为：

![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_%7Bi%7D%5E%7B%28t%29%7D+%3D+%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D+%2B+f_%7Bt%7D%28x_%7Bi%7D%29+%5C%5C)

其中， ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D) 是由第 ![[公式]](https://www.zhihu.com/equation?tex=+t-1+) 步的模型给出的预测值，是已知常数， ![[公式]](https://www.zhihu.com/equation?tex=+f_%7Bt%7D%28x_%7Bi%7D%29+) 是这次需要加入的新模型的预测值。此时，目标函数就可以写成：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cbegin%7Baligned%7D+Obj%5E%7B%28t%29%7D+%26%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bl%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%5E%7B%28t%29%7D%29%7D+%2B+%5Csum_%7Bi%3D1%7D%5E%7Bt%7D%7B%5COmega%28f_%7Bi%7D%29%7D+%5C%5C+%26+%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bl%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%2Bf_%7Bt%7D%28x_%7Bi%7D%29%29%7D+%2B+%5Csum_%7Bi%3D1%7D%5E%7Bt%7D%7B%5COmega%28f_%7Bi%7D%29%7D+%5C%5C+%26%3D%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7Bl%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%2Bf_%7Bt%7D%28x_%7Bi%7D%29%29%7D+%2B+%5COmega%28f_%7Bt%7D%29+%2Bconstant++%5Cend%7Baligned%7D+%5Cend%7Bequation%7D+%5C%5C)

注意上式中，只有一个变量，那就是第 ![[公式]](https://www.zhihu.com/equation?tex=t) 棵树![[公式]](https://www.zhihu.com/equation?tex=f_%7Bt%7D%28x_%7Bi%7D%29) ，其余都是已知量或可通过已知量可以计算出来的。细心的同学可能会问，上式中的第二行到第三行是如何得到的呢？这里我们将正则化项进行拆分，由于前![[公式]](https://www.zhihu.com/equation?tex=t-1) 棵树的结构已经确定，因此前![[公式]](https://www.zhihu.com/equation?tex=+t-1+) 棵树的复杂度之和可以用一个常量表示，如下所示：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Bequation%7D+%5Cbegin%7Baligned%7D+%5Csum_%7Bi%3D1%7D%5E%7Bt%7D%7B%5COmega%28f_%7Bi%7D%29%7D+%26%3D%5COmega%28f_%7Bt%7D%29+%2B+%5Csum_%7Bi%3D1%7D%5E%7Bt-1%7D%7B%5COmega%28f_%7Bi%7D%29%7D+%5C%5C+%26%3D+%5COmega%28f_%7Bt%7D%29+%2B+constant+%5Cend%7Baligned%7D+%5Cend%7Bequation%7D+%5C%5C)

#### **泰勒公式展开**

泰勒公式是将一个在 ![[公式]](https://www.zhihu.com/equation?tex=x%3Dx_%7B0%7D) 处具有![[公式]](https://www.zhihu.com/equation?tex=+n) 阶导数的函数![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 利用关于![[公式]](https://www.zhihu.com/equation?tex=%28x-x_%7B0%7D%29) 的 ![[公式]](https://www.zhihu.com/equation?tex=n) 次多项式来逼近函数的方法。若函数![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 在包含 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B0%7D) 的某个闭区间 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%5D) 上具有 ![[公式]](https://www.zhihu.com/equation?tex=n) 阶导数，且在开区间 ![[公式]](https://www.zhihu.com/equation?tex=%28a%2Cb%29) 上具有 ![[公式]](https://www.zhihu.com/equation?tex=n%2B1+) 阶导数，则对闭区间 ![[公式]](https://www.zhihu.com/equation?tex=%5Ba%2Cb%5D) 上任意一点 ![[公式]](https://www.zhihu.com/equation?tex=x) 有：

![[公式]](https://www.zhihu.com/equation?tex=f%28x%29+%3D+%5Csum_%7Bi%3D0%7D%5E%7Bn%7D%7B%5Cfrac%7Bf%5E%7B%28i%29%7D%28x_%7B0%7D%29%7D%7Bi%21%7D%7D%28x-x_%7B0%7D%29%5E%7Bi%7D%2BR_%7Bn%7D%28x%29+%5C%5C)

其中的多项式称为函数在 ![[公式]](https://www.zhihu.com/equation?tex=x_%7B0%7D) 处的泰勒展开式，![[公式]](https://www.zhihu.com/equation?tex=R_%7Bn%7D%28x%29) 是泰勒公式的余项且是 ![[公式]](https://www.zhihu.com/equation?tex=%28x-x_%7B0%7D%29%5E%7Bn%7D) 的高阶无穷小。

根据泰勒公式，把函数 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%2B%5CDelta+x%29) 在点 ![[公式]](https://www.zhihu.com/equation?tex=x) 处进行泰勒的二阶展开，可得如下等式：

![[公式]](https://www.zhihu.com/equation?tex=f%28x%2B%5CDelta+x%29+%5Capprox+f%28x%29%2Bf%27%28x%29%5CDelta+x+%2B+%5Cfrac%7B1%7D%7B2%7D+f%27%27%28x%29%5CDelta+x%5E%7B2%7D+%5C%5C)

回到XGBoost的目标函数上来， ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) 对应损失函数 ![[公式]](https://www.zhihu.com/equation?tex=l%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%2Bf_%7Bt%7D%28x_%7Bi%7D%29%29) ， ![[公式]](https://www.zhihu.com/equation?tex=x) 对应前 ![[公式]](https://www.zhihu.com/equation?tex=t-1+) 棵树的预测值 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D) ，![[公式]](https://www.zhihu.com/equation?tex=%5CDelta+x) 对应于我们正在训练的第 ![[公式]](https://www.zhihu.com/equation?tex=t) 棵树 ![[公式]](https://www.zhihu.com/equation?tex=f_%7Bt%7D%28x_%7Bi%7D%29) ，则可以将损失函数写为：

![[公式]](https://www.zhihu.com/equation?tex=l%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%2Bf_%7Bt%7D%28x_%7Bi%7D%29%29+%3D+l%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%29+%2B+g_%7Bi%7Df_%7Bt%7D%28x_%7Bi%7D%29+%2B+%5Cfrac%7B1%7D%7B2%7Dh_%7Bi%7Df_%7Bt%7D%5E%7B2%7D%28x_%7Bi%7D%29+%5C%5C)

其中， ![[公式]](https://www.zhihu.com/equation?tex=g_%7Bi%7D) 为损失函数的一阶导， ![[公式]](https://www.zhihu.com/equation?tex=h_%7Bi%7D) 为损失函数的二阶导，注意这里的求导是对 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D) 求导。

我们以平方损失函数为例：

![[公式]](https://www.zhihu.com/equation?tex=l%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%29+%3D+%28y_%7Bi%7D-%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%29%5E%7B2%7D+%5C%5C)

则：

![[公式]](https://www.zhihu.com/equation?tex=g_%7Bi%7D+%3D+%5Cfrac%7B%5Cpartial+l%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%29%7D%7B%5Cpartial+%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D+%7D+%3D++-2%28y_%7Bi%7D-%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%29+%5C%5C)

![[公式]](https://www.zhihu.com/equation?tex=h_%7Bi%7D+%3D+%5Cfrac%7B%5Cpartial+%5E%7B2%7D+l%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%29%7D%7B%5Cpartial+%28%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%29%5E%7B2%7D+%7D+%3D++2+%5C%5C)

将上述的二阶展开式，带入到XGBoost的目标函数中，可以得到目标函数的近似值：

![[公式]](https://www.zhihu.com/equation?tex=Obj%5E%7B%28t%29%7D+%5Csimeq+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Bl%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%5E%7Bt-1%7D%29%2Bg_%7Bi%7Df_%7Bt%7D%28x_%7Bi%7D%29%2B%5Cfrac%7B1%7D%7B2%7Dh_%7Bi%7Df_%7Bt%7D%5E%7B2%7D%28x_%7Bi%7D%29%5D%7D+%2B+%5COmega%28f_%7Bt%7D%29%2Bconstant+%5C%5C)

由于在第 ![[公式]](https://www.zhihu.com/equation?tex=t) 步时 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D) 其实是一个已知的值，所以 ![[公式]](https://www.zhihu.com/equation?tex=l%28y_%7Bi%7D%2C%5Chat%7By%7D_%7Bi%7D%5E%7B%28t-1%29%7D%29+) 是一个常数，其对函数的优化不会产生影响。因此，去掉全部的常数项，得到目标函数为：

![[公式]](https://www.zhihu.com/equation?tex=Obj%5E%7B%28t%29%7D+%5Csimeq+%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%5Bg_%7Bi%7Df_%7Bt%7D%28x_%7Bi%7D%29%2B%5Cfrac%7B1%7D%7B2%7Dh_%7Bi%7Df_%7Bt%7D%5E%7B2%7D%28x_%7Bi%7D%29%5D%7D%2B%5COmega%28f_%7Bt%7D%29+%5C%5C)

所以我们只需要求出每一步损失函数的一阶导和二阶导的值（由于前一步的 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D%5E%7B%28t-1%29%7D) 是已知的，所以这两个值就是常数），然后最优化目标函数，就可以得到每一步的 ![[公式]](https://www.zhihu.com/equation?tex=f%28x%29) ，最后根据加法模型得到一个整体模型。



### 一棵树的生长细节

#### 分裂结点

在实际训练过程中，当建立第 t 棵树时，XGBoost采用贪心法进行树结点的分裂：

从树深为0时开始：

- 对树中的每个叶子结点尝试进行分裂；
- 每次分裂后，原来的一个叶子结点继续分裂为左右两个子叶子结点，原叶子结点中的样本集将根据该结点的判断规则分散到左右两个叶子结点中；
- 新分裂一个结点后，我们需要检测这次分裂是否会给损失函数带来增益，增益的定义如下：

![img](https://pic4.zhimg.com/80/v2-61e13bb229a8574a8ff9a1f9d8fcc87b_hd.jpg)

如果增益Gain>0，即分裂为两个叶子节点后，目标函数下降了，那么我们会考虑此次分裂的结果。

但是，在一个结点分裂时，可能有很多个分裂点，每个分裂点都会产生一个增益，如何才能寻找到最优的分裂点呢？接下来会讲到。

#### 寻找最佳分裂点

> 在实际训练过程中，当建立第 ![[公式]](https://www.zhihu.com/equation?tex=t) 棵树时，一个非常关键的问题是如何找到叶子节点的最优切分点，XGBoost支持两种分裂节点的方法——**贪心算法**和**近似算法**。

###### 贪心算法

  从树的深度为0开始：

  > 1. 对每个叶节点枚举所有的可用特征；
  > 2. 针对每个特征，把属于该节点的训练样本根据该特征值进行升序排列，通过**线性扫描**的方式来决定该特征的最佳分裂点，并记录该特征的**分裂收益**；
  > 3. 选择收益最大的特征作为分裂特征，用该特征的最佳分裂点作为分裂位置，在该节点上分裂出左右两个新的叶节点，并为每个新节点关联对应的样本集；
  > 4. 回到第1步，递归执行直到满足特定条件为止；

 

 **那么如何计算每个特征的分裂收益呢？**

  假设我们在某一节点完成特征分裂，则分裂前的目标函数可以写为：

  ![[公式]](https://www.zhihu.com/equation?tex=Obj_%7B1%7D+%3D-%5Cfrac12+%5B%5Cfrac%7B%28G_L%2BG_R%29%5E2%7D%7BH_L%2BH_R%2B%5Clambda%7D%5D+%2B+%5Cgamma++%5C%5C)

  分裂后的目标函数为：

  ![[公式]](https://www.zhihu.com/equation?tex=Obj_2+%3D++-%5Cfrac12+%5B+%5Cfrac%7BG_L%5E2%7D%7BH_L%2B%5Clambda%7D+%2B+%5Cfrac%7BG_R%5E2%7D%7BH_R%2B%5Clambda%7D%5D+%2B2%5Cgamma+%5C%5C)

  则对于目标函数来说，分裂后的收益为：

  ![[公式]](https://www.zhihu.com/equation?tex=Gain%3D%5Cfrac12+%5Cleft%5B+%5Cfrac%7BG_L%5E2%7D%7BH_L%2B%5Clambda%7D+%2B+%5Cfrac%7BG_R%5E2%7D%7BH_R%2B%5Clambda%7D+-+%5Cfrac%7B%28G_L%2BG_R%29%5E2%7D%7BH_L%2BH_R%2B%5Clambda%7D%5Cright%5D+-+%5Cgamma+%5C%5C)

  **注意：**该特征收益也可作为特征重要性输出的重要依据。

  **对于每次分裂，我们都需要枚举所有特征可能的分割方案，如何高效地枚举所有的分割呢？**

  假设我们要枚举某个特征所有 ![[公式]](https://www.zhihu.com/equation?tex=x+%3C+a) 这样条件的样本，对于某个特定的分割点 ![[公式]](https://www.zhihu.com/equation?tex=a) 我们要计算 ![[公式]](https://www.zhihu.com/equation?tex=a) 左边和右边的导数和。

  ![img](https://pic2.zhimg.com/80/v2-973173d22eeb508eb1b6f26acbf9f2d1_hd.jpg)

  我们可以发现对于所有的分裂点 ![[公式]](https://www.zhihu.com/equation?tex=a) ，只要做一遍从左到右的扫描就可以枚举出所有分割的梯度和 ![[公式]](https://www.zhihu.com/equation?tex=G_L) 、 ![[公式]](https://www.zhihu.com/equation?tex=G_R) 。然后用上面的公式计算每个分割方案的收益就可以了。

  观察分裂后的收益，我们会发现节点划分不一定会使得结果变好，因为我们有一个引入新叶子的惩罚项，也就是说引入的分割带来的增益如果小于一个阀值的时候，我们可以剪掉这个分割。

上面是一种贪心的方法，每次进行分裂尝试都要遍历一遍全部候选分割点，也叫做全局扫描法。

但当数据量过大导致内存无法一次载入或者在分布式情况下，贪心算法的效率就会变得很低，全局扫描法不再适用。

> 基于此，XGBoost提出了一系列加快寻找最佳分裂点的方案：
>
> - **特征预排序+缓存：**XGBoost在训练之前，预先对每个特征按照特征值大小进行排序，然后保存为block结构，后面的迭代中会重复地使用这个结构，使计算量大大减小。
>
> - **分位点近似法：**对每个特征按照特征值排序后，采用类似分位点选取的方式，仅仅选出常数个特征值作为该特征的候选分割点，在寻找该特征的最佳分割点时，从候选分割点中选出最优的一个。
>
> - **并行查找：**由于各个特性已预先存储为block结构，XGBoost支持利用多个线程并行地计算每个特征的最佳分割点，这不仅大大提升了结点的分裂速度，也极利于大规模训练集的适应性扩展。

###### 近似算法

  贪心算法可以得到最优解，但当数据量太大时则无法读入内存进行计算，近似算法主要针对贪心算法这一缺点给出了近似最优解。

  对于每个特征，只考察分位点可以减少计算复杂度。

  该算法首先根据特征分布的分位数提出候选划分点，然后将连续型特征映射到由这些候选点划分的桶中，然后聚合统计信息找到所有区间的最佳分裂点。

  在提出候选切分点时有两种策略：

  - **Global：**学习每棵树前就提出候选切分点，并在每次分裂时都采用这种分割；
  - **Local：**每次分裂前将重新提出候选切分点。

  直观上来看，Local策略需要更多的计算步骤，而Global策略因为节点已有划分所以需要更多的候选点。

  下图给出不同种分裂策略的AUC变化曲线，横坐标为迭代次数，纵坐标为测试集AUC，eps为近似算法的精度，其倒数为桶的数量。

  ![img](https://pic4.zhimg.com/80/v2-3081183127c025ee9f3a1436bb873b07_hd.jpg)

  从上图我们可以看到， Global 策略在候选点数多时（eps 小）可以和 Local 策略在候选点少时（eps 大）具有相似的精度。此外我们还发现，在eps取值合理的情况下，**分位数策略**可以获得与贪心算法相同的精度。

  近似算法简单来说，就是根据特征 ![[公式]](https://www.zhihu.com/equation?tex=k) 的分布来确定 ![[公式]](https://www.zhihu.com/equation?tex=l) 个候选切分点 ![[公式]](https://www.zhihu.com/equation?tex=S_%7Bk%7D+%3D+%5Cleft%5C%7B+s_%7Bk1%7D%2C+s_%7Bk2%7D%2C...%2C+s_%7Bkl%7D+%5Cright%5C%7D) ，然后根据这些候选切分点把相应的样本放入对应的桶中，对每个桶的 ![[公式]](https://www.zhihu.com/equation?tex=G%2CH) 进行累加。最后在候选切分点集合上贪心查找。该算法描述如下：

  ![img](https://pic1.zhimg.com/80/v2-1fe2882f8ef3b0a80068c57905ceaba0_hd.jpg)

  **算法讲解：**

  - **第一个for循环：**对特征k根据该特征分布的分位数找到切割点的候选集合 ![[公式]](https://www.zhihu.com/equation?tex=S_k%3D%5C%7Bs_%7Bk1%7D%2Cs_%7Bk2%7D%2C...%2Cs_%7Bkl%7D+%5C%7D) 。这样做的目的是提取出部分的切分点不用遍历所有的切分点。其中获取某个特征k的候选切割点的方式叫`proposal`(策略)。XGBoost 支持 Global 策略和 Local 策略。
  - **第二个for循环：**将每个特征的取值映射到由该特征对应的候选点集划分的分桶区间，即 ![[公式]](https://www.zhihu.com/equation?tex=%7Bs_%7Bk%2Cv%7D%E2%89%A5x_%7Bjk%7D%3Es_%7Bk%2Cv%E2%88%921%7D%7D) 。对每个桶区间内的样本统计值 G,H并进行累加，最后在这些累计的统计量上寻找最佳分裂点。这样做的目的是获取每个特征的候选分割点的 G,H值。

  下图给出近似算法的具体例子，以三分位为例：

  ![img](https://pic2.zhimg.com/80/v2-cfecb2f6ad675e6e3bf536562e5c06dd_hd.jpg)

  根据样本特征进行排序，然后基于分位数进行划分，并统计三个桶内的 G,H 值，最终求解节点划分的增益。

#### 停止生长

一棵树不会一直生长下去，下面是一些常见的限制条件。

**(1) 当新引入的一次分裂所带来的增益Gain<0时，放弃当前的分裂。这是训练损失和模型结构复杂度的博弈过程。**

![img](https://pic1.zhimg.com/80/v2-46c88b4258c2b9740d89c87d203ed0c0_hd.jpg)

**(2) 当树达到最大深度时，停止建树，因为树的深度太深容易出现过拟合，这里需要设置一个超参数max_depth。**

**(3) 当引入一次分裂后，重新计算新生成的左、右两个叶子结点的样本权重和。如果任一个叶子结点的样本权重低于某一个阈值，也会放弃此次分裂。**这涉及到一个超参数:最小样本权重和，是指如果一个叶子节点包含的样本数量太少也会放弃分裂，防止树分的太细，这也是过拟合的一种措施。

每个叶子结点的样本权值和计算方式如下：

<img src="https://pic3.zhimg.com/80/v2-4ecca09165ffb7a76123401d2009191a_hd.jpg" alt="img" style="zoom:33%;" />

总结推导过程：

![总结推导过程](https://pic2.zhimg.com/80/v2-def00357a06b469b6144d6acb8ab75a9_hd.jpg)

## 算法工程优化

### 对内存的优化：**列块并行学习**

在树生成过程中，最耗时的一个步骤就是在每次寻找最佳分裂点时都需要对特征的值进行排序。而 XGBoost 在训练之前会根据特征对数据进行排序，然后保存到**块结构**中，并在每个块结构中都采用了稀疏矩阵存储格式（Compressed Sparse Columns Format，CSC）进行存储，后面的训练过程中会重复地使用块结构，可以大大减小计算量。

作者提出通过按特征进行分块并排序，在块里面保存排序后的特征值及对应样本的引用，以便于获取样本的一阶、二阶导数值。具体流程为：

- 整体训练数据可以看做一个 ![[公式]](https://www.zhihu.com/equation?tex=n%5Ctimes+m) 的超大规模稀疏矩阵

- 按照mini-batch的方式横向分割，可以切成很多个“Block”

- 每一个“Block”内部采用一种Compress Sparse Column的稀疏短阵格式，每一列特征分别做好升序排列，便于搜索切分点，整体的时间复杂度有效降低。

- 通过Block的设置，可以采用并行计算，从而提升模型训练速度。

具体方式如图：

![ 列分块的升序排列优化示意图](https://pic2.zhimg.com/80/v2-3a93e4d9940cf6e2e9fd89dfa38dc62d_hd.jpg)

通过顺序访问排序后的块遍历样本特征的特征值，方便进行切分点的查找。此外分块存储后多个特征之间互不干涉，可以使用多线程同时对不同的特征进行切分点查找，即特征的并行化处理。在对节点进行分裂时需要选择增益最大的特征作为分裂，这时各个特征的增益计算可以同时进行，这也是 XGBoost 能够实现分布式或者多线程计算的原因。

### 对**CPU Cache**的优化：缓存优化

针对一个具体的块(block)，其中存储了排序好的特征值，以及指向特征值所属样本的索引指针，算法需要间接地利用索引指针来获得样本的梯度值。列块并行学习的设计可以减少节点分裂时的计算量，在顺序访问特征值时，访问的是一块连续的内存空间，但通过特征值持有的索引（样本索引）访问样本获取一阶、二阶导数时，这个访问操作访问的内存空间并不连续，这样可能造成cpu缓存命中率低，影响算法效率。由于块中数据是按特征值来排序的，当索引指针指向内存中不连续的样本时，无法充分利用CPU缓存来提速。

为了解决缓存命中率低的问题，XGBoost 提出了两种优化思路。

**（1）提前取数（Prefetching）**

对于精确搜索，利用多线程的方式，给每个线程划分一个连续的缓存空间，当training线程在按特征值的顺序计算梯度的累加时，prefetching线程可以提前将接下来的一批特征值对应的梯度加载到CPU缓存中。为每个线程分配一个连续的缓存区，将需要的梯度信息存放在缓冲区中，这样就实现了非连续空间到连续空间的转换，提高了算法效率。

**（2）合理设置分块大小**

对于近似分桶搜索，按行分块时需要准确地选择块的大小。块太小会导致每个线程的工作量太少，切换线程的成本过高，不利于并行计算；块太大导致缓存命中率低，需要花费更多时间在读取数据上。经过反复实验，作者找到一个合理的`block_size`为 ![[公式]](https://www.zhihu.com/equation?tex=2%5E%7B16%7D) ![[公式]](https://www.zhihu.com/equation?tex=2%5E%7B16%7D)。

### 对IO的优化：核外块计算

当数据量非常大时，我们不能把所有的数据都加载到内存中。那么就必须将一部分需要加载进内存的数据先存放在硬盘中，当需要时再加载进内存。这样操作具有很明显的瓶颈，即硬盘的IO操作速度远远低于内存的处理速度，肯定会存在大量等待硬盘IO操作的情况。针对这个问题作者提出了“核外”计算的优化方法。具体操作为，将数据集分成多个块存放在硬盘中，使用一个独立的线程专门从硬盘读取数据，加载到内存中，这样算法在内存中处理数据就可以和从硬盘读取数据同时进行。此外，XGBoost 还用了两种方法来降低硬盘读写的开销：

- **块压缩**（**Block Compression**）。论文使用的是按列进行压缩，读取的时候用另外的线程解压。对于行索引，只保存第一个索引值，然后用16位的整数保存与该block第一个索引的差值。作者通过测试在block设置为 ![[公式]](https://www.zhihu.com/equation?tex=2%5E%7B16%7D) 个样本大小时，压缩比率几乎达到26% ![[公式]](https://www.zhihu.com/equation?tex=%5Csim) 29%。
- **块分区**（**Block Sharding** ）。块分区是将特征block分区存放在不同的硬盘上，以此来增加硬盘IO的吞吐量。


## 优缺点

### 优点

- **精度更高：**GBDT 只用到一阶泰勒展开，而 XGBoost 对损失函数进行了二阶泰勒展开。XGBoost 引入二阶导一方面是为了增加精度，另一方面也是为了能够自定义损失函数，二阶泰勒展开可以近似大量损失函数；
- **灵活性更强：**GBDT 以 CART 作为基分类器，XGBoost 不仅支持 CART 还支持线性分类器，使用线性分类器的 XGBoost 相当于带 L1 和 L2 正则化项的逻辑斯蒂回归（分类问题）或者线性回归（回归问题）。此外，XGBoost 工具支持自定义损失函数，只需函数支持一阶和二阶求导；
- **正则化：**XGBoost 在目标函数中加入了正则项，用于控制模型的复杂度。正则项里包含了树的叶子节点个数、叶子节点权重的 L2 范式。正则项降低了模型的方差，使学习出来的模型更加简单，有助于防止过拟合，这也是XGBoost优于传统GBDT的一个特性。
- **Shrinkage（缩减）：**相当于学习速率。XGBoost 在进行完一次迭代后，会将叶子节点的权重乘上该系数，主要是为了削弱每棵树的影响，让后面有更大的学习空间。传统GBDT的实现也有学习速率；
- **列抽样：**XGBoost 借鉴了随机森林的做法，支持列抽样，不仅能降低过拟合，还能减少计算。这也是XGBoost异于传统GBDT的一个特性；
- **缺失值处理：**对于特征的值有缺失的样本，XGBoost 采用的稀疏感知算法可以自动学习出它的分裂方向；
- **XGBoost工具支持并行：**boosting不是一种串行的结构吗?怎么并行的？注意XGBoost的并行不是tree粒度的并行，XGBoost也是一次迭代完才能进行下一次迭代的（第t次迭代的代价函数里包含了前面t-1次迭代的预测值）。XGBoost的并行是在特征粒度上的。我们知道，决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点），XGBoost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。这个block结构也使得并行成为了可能，在进行节点的分裂时，需要计算每个特征的增益，最终选增益最大的那个特征去做分裂，那么各个特征的增益计算就可以开多线程进行。
- **可并行的近似算法：**树节点在进行分裂时，我们需要计算每个特征的每个分割点对应的增益，即用贪心法枚举所有可能的分割点。当数据无法一次载入内存或者在分布式情况下，贪心算法效率就会变得很低，所以XGBoost还提出了一种可并行的近似算法，用于高效地生成候选的分割点。

### 缺点

- 虽然利用预排序和近似算法可以降低寻找最佳分裂点的计算量，但在节点分裂过程中仍需要遍历数据集；

- 预排序过程的空间复杂度过高，不仅需要存储特征值，还需要存储特征对应样本的梯度统计值的索引，相当于消耗了两倍的内存。

### XGBoost与GBDT的差异

在分析XGBooting优缺点的时候，通过比较该算法与GBDT的差异，即可有较清楚的描述，具体表现在如下方面。

**（1）基分类器的差异**

- GBDT算法只能利用CART树作为基学习器，满足分类应用；
- XGBoost算法除了回归树之外还支持线性的基学习器，因此其一方面可以解决带L1与L2正则化项的逻辑回归分类问题，也可以解决线性回问题。

**（2）节点分类方法的差异**

- GBDT算法主要是利用Gini impurity针对特征进行节点划分；
- XGBoost经过公式推导，提出的weighted quantile sketch（**加权分位数缩略图**）划分方法，依据影响Loss的程度来确定连续特征的切分值。

**（3）模型损失函数的差异**

- 传统GBDT在优化时只用到一阶导数信息；
- xgboost则对代价函数进行了二阶泰勒展开，二阶导数有利于梯度下降的更快更准。

**（4）模型防止过拟合的差异**

- GBDT算法无正则项，可能出现过拟合；
- Xgboost在代价函数里加入了正则项，用于控制模型的复杂度，降低了过拟合的可能性。

**（5）模型实现上的差异**

决策树的学习最耗时的一个步骤就是对特征的值进行排序（因为要确定最佳分割点）。xgboost在训练之前，预先对数据进行了排序，然后保存为block结构，后面的迭代中重复地使用这个结构，大大减小计算量。其能够实现在特征粒度的并行。

## XGBoost代码实现

### **安装XGBoost依赖包**

```python
pip install xgboost
```

### **XGBoost分类和回归**

XGBoost有两大类接口：XGBoost原生接口 和 scikit-learn接口 ，并且XGBoost能够实现分类和回归两种任务。

**（1）基于XGBoost原生接口的分类**

```python
from sklearn.datasets import load_iris
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

# read in the iris data
iris = load_iris()

X = iris.data
y = iris.target

# split train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234565)

# set XGBoost's parameters
params = {
    'booster': 'gbtree',
    'objective': 'multi:softmax',   # 回归任务设置为：'objective': 'reg:gamma',
    'num_class': 3,      # 回归任务没有这个参数
    'gamma': 0.1,
    'max_depth': 6,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

plst = params.items()

dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 500
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)

# 计算准确率
cnt1 = 0
cnt2 = 0
for i in range(len(y_test)):
    if ans[i] == y_test[i]:
        cnt1 += 1
    else:
        cnt2 += 1

print("Accuracy: %.2f %% " % (100 * cnt1 / (cnt1 + cnt2)))

# 显示重要特征
plot_importance(model)
plt.show()
```

**（2）基于Scikit-learn接口的回归**

这里，我们用Kaggle比赛中回归问题：House Prices: Advanced Regression Techniques，地址：[https://www.kaggle.com/c/house-prices-advanced-regression-techniques](https://link.zhihu.com/?target=https%3A//www.kaggle.com/c/house-prices-advanced-regression-techniques) 来进行实例讲解。

该房价预测的训练数据集中一共有81列，第一列是Id，最后一列是label，中间79列是特征。这79列特征中，有43列是分类型变量，33列是整数变量，3列是浮点型变量。训练数据集中存在缺失值。

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

# 1.读文件
data = pd.read_csv('./dataset/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)

# 2.切分数据输入：特征 输出：预测目标变量
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

# 3.切分训练集、测试集,切分比例7.5 : 2.5
train_X, test_X, train_y, test_y = train_test_split(X.values, y.values, test_size=0.25)

# 4.空值处理，默认方法：使用特征列的平均值进行填充
my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)

# 5.调用XGBoost模型，使用训练集数据进行训练（拟合）
# Add verbosity=2 to print messages while running boosting
my_model = xgb.XGBRegressor(objective='reg:squarederror', verbosity=2)  # xgb.XGBClassifier() XGBoost分类模型
my_model.fit(train_X, train_y, verbose=False)

# 6.使用模型对测试集数据进行预测
predictions = my_model.predict(test_X)

# 7.对模型的预测结果进行评判（平均绝对误差）
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
```

### **XGBoost调参**

在上一部分中，XGBoot模型的参数都使用了模型的默认参数，但默认参数并不是最好的。要想让XGBoost表现的更好，需要对XGBoost模型进行参数微调。XGBoost需要调的参数不算多，他们可以分成三个部分：

> **1、General Parameters，即与整个模型属基调相关的参数；**
>
> **2、Booster Parameters，即与单颗树生成有关的参数；**
>
> **3、Learning Task Parameters，与模型调优相关的参数；**

#### **General Parameters**

**1、booster [default=gbtree]**

即xgboost中基学习器类型，有两种选择，分别是树模型（gbtree）和线性模型（linear models）

**2、silent [default=0]**

即控制迭代日志的是否输出，默认输出；

**3、nthread [default to maximum number of threads available if not set]**

即控制模型训练调用机器的核心数，与sklearn中*n_jobs的含义相似；*

#### **Booster parameters**

因为booster有两种类型，常用的一般是树模型，这里只列树模型相关的参数：

**1、eta [default=0.3]** **：学习率**

学习率，这个相当于sklearn中的learning_rate，常见的设置范围在0.01-0.2之间

**2、min_child_weight [default=1]：叶节点的最小权重值**

这个参数与GBM（sklearn）中的“min_samples_leaf”很相似，只不过这里不是样本数，而是权重值，如果样本的权重都是1，这两个参数是等同的；这个值设置较大时，通常树不会太深，可以控制过拟合，但太大时，容易造成欠拟合的现象，具体调参需要cv；

**3、max_depth：树的最大深度**

树的最大深度，含义很直白，控制树的复杂性；通常取值范围在3-10；

**4、max_leaf_nodes：最大叶节点数**

一般这个参数与max_depth二选一控制即可；

**5、gamma [default=0]：分裂收益阈值**

即用来比较每次节点分裂带来的收益，有效控制节点的过度分裂；

这个参数的变化范围受损失函数的选取影响；

**6、max_delta_step [default=0]**

这个参数暂时不是很理解它的作用范围，一般可以忽略它；

**7、subsample [default=1]：采样比例**

与sklearn中的参数一样，即每颗树的生成可以不去全部样本，这样可以控制模型的过拟合；通常取值范围0.5-1；

**8、colsample_bytree [default=1]：特征采样的比例（每棵树）**

即每棵树不使用全部的特征，控制模型的过拟合；

通常取值范围0.5-1；

**9、colsample_bylevel [default=1]**

特征采样的比例（每次分裂）；

这个与随机森林的思想很相似，即每次分裂都不取全部变量；

当7、8的参数设置较好时，该参数可以不用在意；

**10、lambda [default=1]**

L2范数的惩罚系数，叶子结点的分数？；

**11、alpha [default=0]**

L1范数的惩罚系数，叶子结点数？；

**12、scale_pos_weight [default=1]**

这个参数也不是很理解，貌似与类别不平衡的问题相关；

#### **Learning Task Parameters**

**1、objective [default=reg:linear]：目标函数**

通常的选项分别是：binary:logistic，用于二分类，产生每类的概率值；multi:softmax，用于多分类，但不产生概率值，直接产生类别结果；multi:softprob，类似softmax，但产生多分类的概率值；

**2、eval_metric [ default according to objective ]：评价指标**

当你给模型一个验证集时，会输出对应的评价指标值；

一般有：rmse ，均方误差；mae ，绝对平均误差；logloss ，对数似然值；error ，二分类错误率；merror ，多分类错误率；mlogloss ；auc

**3、seed：即随机种子**

## **关于XGBoost若干问题的思考**

### **XGBoost与GBDT的联系和区别有哪些？**

（1）GBDT是机器学习算法，XGBoost是该算法的工程实现。

（2）**正则项：**在使用CART作为基分类器时，XGBoost显式地加入了正则项来控制模型的复杂度，有利于防止过拟合，从而提高模型的泛化能力。

（3）**导数信息：**GBDT在模型训练时只使用了代价函数的一阶导数信息，XGBoost对代价函数进行二阶泰勒展开，可以同时使用一阶和二阶导数。

（4）**基分类器：**传统的GBDT采用CART作为基分类器，XGBoost支持多种类型的基分类器，比如线性分类器。

（5）**子采样：**传统的GBDT在每轮迭代时使用全部的数据，XGBoost则采用了与随机森林相似的策略，支持对数据进行采样。

（6）**缺失值处理：**传统GBDT没有设计对缺失值进行处理，XGBoost能够自动学习出缺失值的处理策略。

（7）**并行化**：传统GBDT没有进行并行化设计，注意不是tree维度的并行，而是特征维度的并行。XGBoost预先将每个特征按特征值排好序，存储为块结构，分裂结点时可以采用多线程并行查找每个特征的最佳分割点，极大提升训练速度。

### **为什么XGBoost泰勒二阶展开后效果就比较好呢？**

（1）**从为什么会想到引入泰勒二阶的角度来说（可扩展性）：**XGBoost官网上有说，当目标函数是MSE时，展开是一阶项（残差）+二阶项的形式，而其它目标函数，如logistic loss的展开式就没有这样的形式。为了能有个统一的形式，所以采用泰勒展开来得到二阶项，这样就能把MSE推导的那套直接复用到其它自定义损失函数上。简短来说，就是为了统一损失函数求导的形式以支持自定义损失函数。至于为什么要在形式上与MSE统一？是因为MSE是最普遍且常用的损失函数，而且求导最容易，求导后的形式也十分简单。所以理论上只要损失函数形式与MSE统一了，那就只用推导MSE就好了。

（2）**从二阶导本身的性质，也就是从为什么要用泰勒二阶展开的角度来说（精准性）：**二阶信息本身就能让梯度收敛更快更准确。这一点在优化算法里的牛顿法中已经证实。可以简单认为一阶导指引梯度方向，二阶导指引梯度方向如何变化。简单来说，相对于GBDT的一阶泰勒展开，XGBoost采用二阶泰勒展开，可以更为精准的逼近真实的损失函数。

### **XGBoost对缺失值是怎么处理的？**

在普通的GBDT策略中，对于缺失值的方法是先手动对缺失值进行填充，然后当做有值的特征进行处理，但是这样人工填充不一定准确，而且没有什么理论依据。而XGBoost采取的策略是先不处理那些值缺失的样本，采用那些有值的样本搞出分裂点，在遍历每个有值特征的时候，尝试将缺失样本划入左子树和右子树，选择使损失最优的值作为分裂点。

### **XGBoost为什么可以并行训练？**

（1）XGBoost的并行，并不是说每棵树可以并行训练，XGBoost本质上仍然采用boosting思想，每棵树训练前需要等前面的树训练完成才能开始训练。

（2）XGBoost的并行，指的是特征维度的并行：在训练之前，每个特征按特征值对样本进行预排序，并存储为Block结构，在后面查找特征分割点时可以重复使用，而且特征已经被存储为一个个block结构，那么在寻找每个特征的最佳分割点时，可以利用多线程对每个block并行计算。

## 20道XGBoost面试题

### 简单介绍一下XGBoost

首先需要说一说GBDT，它是一种基于boosting增强策略的加法模型，训练的时候采用前向分布算法进行贪婪的学习，每次迭代都学习一棵CART树来拟合之前 t-1 棵树的预测结果与训练样本真实值的残差。

XGBoost对GBDT进行了一系列优化，比如损失函数进行了二阶泰勒展开、目标函数加入正则项、支持并行和默认缺失值处理等，在可扩展性和训练速度上有了巨大的提升，但其核心思想没有大的变化。

### XGBoost与GBDT有什么不同

- **基分类器**：XGBoost的基分类器不仅支持CART决策树，还支持线性分类器，此时XGBoost相当于带L1和L2正则化项的Logistic回归（分类问题）或者线性回归（回归问题）。
- **导数信息**：XGBoost对损失函数做了二阶泰勒展开，GBDT只用了一阶导数信息，并且XGBoost还支持自定义损失函数，只要损失函数一阶、二阶可导。
- **正则项**：XGBoost的目标函数加了正则项， 相当于预剪枝，使得学习出来的模型更加不容易过拟合。
- **列抽样**：XGBoost支持列采样，与随机森林类似，用于防止过拟合。
- **缺失值处理**：对树中的每个非叶子结点，XGBoost可以自动学习出它的默认分裂方向。如果某个样本该特征值缺失，会将其划入默认分支。
- **并行化**：注意不是tree维度的并行，而是特征维度的并行。XGBoost预先将每个特征按特征值排好序，存储为块结构，分裂结点时可以采用多线程并行查找每个特征的最佳分割点，极大提升训练速度。

### XGBoost为什么使用泰勒二阶展开

- **精准性**：相对于GBDT的一阶泰勒展开，XGBoost采用二阶泰勒展开，可以更为精准的逼近真实的损失函数
- **可扩展性**：损失函数支持自定义，只需要新的损失函数二阶可导。

### XGBoost为什么可以并行训练

- XGBoost的并行，并不是说每棵树可以并行训练，XGB本质上仍然采用boosting思想，每棵树训练前需要等前面的树训练完成才能开始训练。
- XGBoost的并行，指的是特征维度的并行：在训练之前，每个特征按特征值对样本进行预排序，并存储为Block结构，在后面查找特征分割点时可以重复使用，而且特征已经被存储为一个个block结构，那么在寻找每个特征的最佳分割点时，可以利用多线程对每个block并行计算。

### XGBoost为什么快

- **分块并行**：训练前每个特征按特征值进行排序并存储为Block结构，后面查找特征分割点时重复使用，并且支持并行查找每个特征的分割点
- **候选分位点**：每个特征采用常数个分位点作为候选分割点
- **CPU cache 命中优化**： 使用缓存预取的方法，对每个线程分配一个连续的buffer，读取每个block中样本的梯度信息并存入连续的Buffer中。
- **Block 处理优化**：Block预先放入内存；Block按列进行解压缩；将Block划分到不同硬盘来提高吞吐

### XGBoost防止过拟合的方法

XGBoost在设计时，为了防止过拟合做了很多优化，具体如下：

- **目标函数添加正则项**：叶子节点个数+叶子节点权重的L2正则化
- **列抽样**：训练的时候只用一部分特征（不考虑剩余的block块即可）
- **子采样**：每轮计算可以不使用全部样本，使算法更加保守
- **shrinkage**: 可以叫学习率或步长，为了给后面的训练留出更多的学习空间

### XGBoost如何处理缺失值

XGBoost模型的一个优点就是允许特征存在缺失值。对缺失值的处理方式如下：

- 在特征k上寻找最佳 split point 时，不会对该列特征 missing 的样本进行遍历，而只对该列特征值为 non-missing 的样本上对应的特征值进行遍历，通过这个技巧来减少了为稀疏离散特征寻找 split point 的时间开销。

  

- 在逻辑实现上，为了保证完备性，会将该特征值missing的样本分别分配到左叶子结点和右叶子结点，两种情形都计算一遍后，选择分裂后增益最大的那个方向（左分支或是右分支），作为预测时特征值缺失样本的默认分支方向。

  

- 如果在训练中没有缺失值而在预测中出现缺失，那么会自动将缺失值的划分方向放到右子结点。

  

<img src="https://mmbiz.qpic.cn/mmbiz_png/90dLE6ibsg0fkqnx5yOhtlvx8dFgk1DvVfp2pmTsZ0yX0A2usH3afam4cJb7lQNIJGb3N2VZicclrfoRqM6MHhtQ/640?wx_fmt=png&amp;tp=webp&amp;wxfrom=5&amp;wx_lazy=1&amp;wx_co=1" alt="img" style="zoom:33%;" />

find_split时，缺失值处理的伪代码

### XGBoost中叶子结点的权重如何计算出来

XGBoost目标函数最终推导形式如下：

![img](https://mmbiz.qpic.cn/mmbiz_png/90dLE6ibsg0fDfLgXV02BLFJ9eaFEJB0ERQaHDopzOeSvCyaPGicmHqArjzlJYDejcTs9YJoAFdAqwyVrdpUPZQA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



利用一元二次函数求最值的知识，当目标函数达到最小值Obj*时，每个叶子结点的权重为wj*。

具体公式如下：

![img](https://mmbiz.qpic.cn/mmbiz_png/90dLE6ibsg0fDfLgXV02BLFJ9eaFEJB0EURBYpwF4xF4x2lLh7BroeKUjRqk17VXpkZqPEjaskia4kiazjs9nyg0A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### XGBoost中的一棵树的停止生长条件

- 当新引入的一次分裂所带来的增益Gain<0时，放弃当前的分裂。这是训练损失和模型结构复杂度的博弈过程。
- 当树达到最大深度时，停止建树，因为树的深度太深容易出现过拟合，这里需要设置一个超参数max_depth。
- 当引入一次分裂后，重新计算新生成的左、右两个叶子结点的样本权重和。如果任一个叶子结点的样本权重低于某一个阈值，也会放弃此次分裂。这涉及到一个超参数:最小样本权重和，是指如果一个叶子节点包含的样本数量太少也会放弃分裂，防止树分的太细。

### RF和GBDT的区别

**相同点：**

- 都是由多棵树组成，最终的结果都是由多棵树一起决定。

**不同点：**

- **集成学习**：RF属于bagging思想，而GBDT是boosting思想
- **偏差-方差权衡**：RF不断的降低模型的方差，而GBDT不断的降低模型的偏差
- **训练样本**：RF每次迭代的样本是从全部训练集中有放回抽样形成的，而GBDT每次使用全部样本
- **并行性**：RF的树可以并行生成，而GBDT只能顺序生成(需要等上一棵树完全生成)
- **最终结果**：RF最终是多棵树进行多数表决（回归问题是取平均），而GBDT是加权融合
- **数据敏感性**：RF对异常值不敏感，而GBDT对异常值比较敏感
- **泛化能力**：RF不易过拟合，而GBDT容易过拟合



### XGBoost如何处理不平衡数据

对于不平衡的数据集，例如用户的购买行为，肯定是极其不平衡的，这对XGBoost的训练有很大的影响，XGBoost有两种自带的方法来解决：

第一种，如果你在意AUC，采用AUC来评估模型的性能，那你可以通过设置scale_pos_weight来平衡正样本和负样本的权重。例如，当正负样本比例为1:10时，scale_pos_weight可以取10；

第二种，如果你在意概率(预测得分的合理性)，你不能重新平衡数据集(会破坏数据的真实分布)，应该设置max_delta_step为一个有限数字来帮助收敛（基模型为LR时有效）。

原话是这么说的：

```python
For common cases such as ads clickthrough log, the dataset is extremely imbalanced. This can affect the training of xgboost model, and there are two ways to improve it.  If you care only about the ranking order (AUC) of your prediction      Balance the positive and negative weights, via scale_pos_weight      Use AUC for evaluation  If you care about predicting the right probability      In such a case, you cannot re-balance the dataset      In such a case, set parameter max_delta_step to a finite number (say 1) will help convergence
```

那么，源码到底是怎么利用**scale_pos_weight**来平衡样本的呢，是调节权重还是过采样呢？请看源码：

```python
if (info.labels[i] == 1.0f)  w *= param_.scale_pos_weight
```

可以看出，应该是增大了少数样本的权重。

除此之外，还可以通过上采样、下采样、SMOTE算法或者自定义代价函数的方式解决正负样本不平衡的问题。

### 比较LR和GBDT，说说什么情景下GBDT不如LR

先说说LR和GBDT的区别：

- LR是线性模型，可解释性强，很容易并行化，但学习能力有限，需要大量的人工特征工程
- GBDT是非线性模型，具有天然的特征组合优势，特征表达能力强，但是树与树之间无法并行训练，而且树模型很容易过拟合；

当在高维稀疏特征的场景下，LR的效果一般会比GBDT好。原因如下：

先看一个例子：

> 假设一个二分类问题，label为0和1，特征有100维，如果有1w个样本，但其中只要10个正样本1，而这些样本的特征 f1的值为全为1，而其余9990条样本的f1特征都为0(在高维稀疏的情况下这种情况很常见)。
>
> 我们都知道在这种情况下，树模型很容易优化出一个使用f1特征作为重要分裂节点的树，因为这个结点直接能够将训练数据划分的很好，但是当测试的时候，却会发现效果很差，因为这个特征f1只是刚好偶然间跟y拟合到了这个规律，这也是我们常说的过拟合。

那么这种情况下，如果采用LR的话，应该也会出现类似过拟合的情况呀：y = W1*f1 + Wi*fi+….，其中 W1特别大以拟合这10个样本。为什么此时树模型就过拟合的更严重呢？

仔细想想发现，因为现在的模型普遍都会带着正则项，而 LR 等线性模型的正则项是对权重的惩罚，也就是 W1一旦过大，惩罚就会很大，进一步压缩 W1的值，使他不至于过大。但是，树模型则不一样，树模型的惩罚项通常为叶子节点数和深度等，而我们都知道，对于上面这种 case，树只需要一个节点就可以完美分割9990和10个样本，一个结点，最终产生的惩罚项极其之小。

这也就是为什么在高维稀疏特征的时候，线性模型会比非线性模型好的原因了：**带正则化的线性模型比较不容易对稀疏特征过拟合。**

### XGBoost中如何对树进行剪枝

- 在目标函数中增加了正则项：使用叶子结点的数目和叶子结点权重的L2模的平方，控制树的复杂度。
- 在结点分裂时，定义了一个阈值，如果分裂后目标函数的增益小于该阈值，则不分裂。
- 当引入一次分裂后，重新计算新生成的左、右两个叶子结点的样本权重和。如果任一个叶子结点的样本权重低于某一个阈值（最小样本权重和），也会放弃此次分裂。
- XGBoost 先从顶到底建立树直到最大深度，再从底到顶反向检查是否有不满足分裂条件的结点，进行剪枝。

### XGBoost如何选择最佳分裂点？

XGBoost在训练前预先将特征按照特征值进行了排序，并存储为block结构，以后在结点分裂时可以重复使用该结构。

因此，可以采用特征并行的方法利用多个线程分别计算每个特征的最佳分割点，根据每次分裂后产生的增益，最终选择增益最大的那个特征的特征值作为最佳分裂点。

如果在计算每个特征的最佳分割点时，对每个样本都进行遍历，计算复杂度会很大，这种全局扫描的方法并不适用大数据的场景。XGBoost还提供了一种直方图近似算法，对特征排序后仅选择常数个候选分裂位置作为候选分裂点，极大提升了结点分裂时的计算效率。

### XGBoost的Scalable性如何体现

- **基分类器的scalability**：弱分类器可以支持CART决策树，也可以支持LR和Linear。
- **目标函数的scalability**：支持自定义loss function，只需要其一阶、二阶可导。有这个特性是因为泰勒二阶展开，得到通用的目标函数形式。
- **学习方法的scalability**：Block结构支持并行化，支持 Out-of-core计算。

### XGBoost如何评价特征的重要性

我们采用三种方法来评判XGBoost模型中特征的重要程度：



```
 官方文档：（1）weight - the number of times a feature is used to split the data across all trees. （2）gain - the average gain of the feature when it is used in trees. （3）cover - the average coverage of the feature when it is used in trees.
```



- **weight** ：该特征在所有树中被用作分割样本的特征的总次数。
- **gain** ：该特征在其出现过的所有树中产生的平均增益。
- **cover** ：该特征在其出现过的所有树中的平均覆盖范围。

> 注意：覆盖范围这里指的是一个特征用作分割点后，其影响的样本数量，即有多少样本经过该特征分割到两个子节点。

### XGBooost参数调优的一般步骤

首先需要初始化一些基本变量，例如：

- max_depth = 5
- min_child_weight = 1
- gamma = 0
- subsample, colsample_bytree = 0.8
- scale_pos_weight = 1

**(1) 确定learning rate和estimator的数量**

learning rate可以先用0.1，用cv来寻找最优的estimators

**(2) max_depth和 min_child_weight**

我们调整这两个参数是因为，这两个参数对输出结果的影响很大。我们首先将这两个参数设置为较大的数，然后通过迭代的方式不断修正，缩小范围。

max_depth，每棵子树的最大深度，check from range(3,10,2)。

min_child_weight，子节点的权重阈值，check from range(1,6,2)。

如果一个结点分裂后，它的所有子节点的权重之和都大于该阈值，该叶子节点才可以划分。

**(3) gamma**

也称作最小划分损失`min_split_loss`，check from 0.1 to 0.5，指的是，对于一个叶子节点，当对它采取划分之后，损失函数的降低值的阈值。

- 如果大于该阈值，则该叶子节点值得继续划分
- 如果小于该阈值，则该叶子节点不值得继续划分

**(4) subsample, colsample_bytree**

subsample是对训练的采样比例

colsample_bytree是对特征的采样比例

both check from 0.6 to 0.9

**(5) 正则化参数**

alpha 是L1正则化系数，try 1e-5, 1e-2, 0.1, 1, 100

lambda 是L2正则化系数

**(6) 降低学习率**

降低学习率的同时增加树的数量，通常最后设置学习率为0.01~0.1

### XGBoost模型如果过拟合了怎么解决

当出现过拟合时，有两类参数可以缓解：

第一类参数：用于直接控制模型的复杂度。包括`max_depth,min_child_weight,gamma` 等参数

第二类参数：用于增加随机性，从而使得模型在训练时对于噪音不敏感。包括`subsample,colsample_bytree`

还有就是直接减小`learning rate`，但需要同时增加`estimator` 参数。

### 为什么XGBoost相比某些模型对缺失值不敏感

对存在缺失值的特征，一般的解决方法是：

- 离散型变量：用出现次数最多的特征值填充；
- 连续型变量：用中位数或均值填充；

一些模型如SVM和KNN，其模型原理中涉及到了对样本距离的度量，如果缺失值处理不当，最终会导致模型预测效果很差。

而树模型对缺失值的敏感度低，大部分时候可以在数据缺失时时使用。原因就是，一棵树中每个结点在分裂时，寻找的是某个特征的最佳分裂点（特征值），完全可以不考虑存在特征值缺失的样本，也就是说，如果某些样本缺失的特征值缺失，对寻找最佳分割点的影响不是很大。

XGBoost对缺失数据有特定的处理方法，[详情参考上篇文章第7题](http://mp.weixin.qq.com/s?__biz=Mzg2MjI5Mzk0MA==&mid=2247484181&idx=1&sn=8d0e51fb0cb974f042e66659e1daf447&chksm=ce0b59cef97cd0d8cf7f9ae1e91e41017ff6d4c4b43a4c19b476c0b6d37f15769f954c2965ef&scene=21#wechat_redirect)。

因此，对于有缺失值的数据在经过缺失处理后：

- 当数据量很小时，优先用朴素贝叶斯
- 数据量适中或者较大，用树模型，优先XGBoost
- 数据量较大，也可以用神经网络
- 避免使用距离度量相关的模型，如KNN和SVM

### XGBoost和LightGBM的区别

![img](https://mmbiz.qpic.cn/mmbiz_png/90dLE6ibsg0cassUTLvbQlGic1CW6ialKxxJ2S8XI3VokUBf5TBOSDG8zb6gZXe0q63b4TyDlDPCX9G6cPXlmR4cw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

（1）树生长策略：XGB采用`level-wise`的分裂策略，LGB采用`leaf-wise`的分裂策略。XGB对每一层所有节点做无差别分裂，但是可能有些节点增益非常小，对结果影响不大，带来不必要的开销。Leaf-wise是在所有叶子节点中选取分裂收益最大的节点进行的，但是很容易出现过拟合问题，所以需要对最大深度做限制 。

（2）分割点查找算法：XGB使用特征预排序算法，LGB使用基于直方图的切分点算法，其优势如下：

- 减少内存占用，比如离散为256个bin时，只需要用8位整形就可以保存一个样本被映射为哪个bin(这个bin可以说就是转换后的特征)，对比预排序的exact greedy算法来说（用int_32来存储索引+ 用float_32保存特征值），可以节省7/8的空间。
- 计算效率提高，预排序的Exact greedy对每个特征都需要遍历一遍数据，并计算增益，复杂度为𝑂(#𝑓𝑒𝑎𝑡𝑢𝑟𝑒×#𝑑𝑎𝑡𝑎)。而直方图算法在建立完直方图后，只需要对每个特征遍历直方图即可，复杂度为𝑂(#𝑓𝑒𝑎𝑡𝑢𝑟𝑒×#𝑏𝑖𝑛𝑠)。
- LGB还可以使用直方图做差加速，一个节点的直方图可以通过父节点的直方图减去兄弟节点的直方图得到，从而加速计算

> 但实际上xgboost的近似直方图算法也类似于lightgbm这里的直方图算法，为什么xgboost的近似算法比lightgbm还是慢很多呢？
>
> xgboost在每一层都动态构建直方图， 因为xgboost的直方图算法不是针对某个特定的feature，而是所有feature共享一个直方图(每个样本的权重是二阶导)，所以每一层都要重新构建直方图，而lightgbm中对每个特征都有一个直方图，所以构建一次直方图就够了。

（3）支持离散变量：无法直接输入类别型变量，因此需要事先对类别型变量进行编码（例如独热编码），而LightGBM可以直接处理类别型变量。

（4）缓存命中率：XGB使用Block结构的一个缺点是取梯度的时候，是通过索引来获取的，而这些梯度的获取顺序是按照特征的大小顺序的，这将导致非连续的内存访问，可能使得CPU cache缓存命中率低，从而影响算法效率。而LGB是基于直方图分裂特征的，梯度信息都存储在一个个bin中，所以访问梯度是连续的，缓存命中率高。

（5）LightGBM 与 XGboost 的并行策略不同：

- **特征并行** ：LGB特征并行的前提是每个worker留有一份完整的数据集，但是每个worker仅在特征子集上进行最佳切分点的寻找；worker之间需要相互通信，通过比对损失来确定最佳切分点；然后将这个最佳切分点的位置进行全局广播，每个worker进行切分即可。XGB的特征并行与LGB的最大不同在于XGB每个worker节点中仅有部分的列数据，也就是垂直切分，每个worker寻找局部最佳切分点，worker之间相互通信，然后在具有最佳切分点的worker上进行节点分裂，再由这个节点广播一下被切分到左右节点的样本索引号，其他worker才能开始分裂。二者的区别就导致了LGB中worker间通信成本明显降低，只需通信一个特征分裂点即可，而XGB中要广播样本索引。
- **数据并行** ：当数据量很大，特征相对较少时，可采用数据并行策略。LGB中先对数据水平切分，每个worker上的数据先建立起局部的直方图，然后合并成全局的直方图，采用直方图相减的方式，先计算样本量少的节点的样本索引，然后直接相减得到另一子节点的样本索引，这个直方图算法使得worker间的通信成本降低一倍，因为只用通信以此样本量少的节点。XGB中的数据并行也是水平切分，然后单个worker建立局部直方图，再合并为全局，不同在于根据全局直方图进行各个worker上的节点分裂时会单独计算子节点的样本索引，因此效率贼慢，每个worker间的通信量也就变得很大。
- **投票并行（LGB）**：当数据量和维度都很大时，选用投票并行，该方法是数据并行的一个改进。数据并行中的合并直方图的代价相对较大，尤其是当特征维度很大时。大致思想是：每个worker首先会找到本地的一些优秀的特征，然后进行全局投票，根据投票结果，选择top的特征进行直方图的合并，再寻求全局的最优分割点。

## 参考资料

**XGBoost论文解读：**

【1】Chen T , Guestrin C . XGBoost: A Scalable Tree Boosting System[J]. 2016.

【2】[Tianqi Chen的XGBoost的Slides](https://homes.cs.washington.edu/~tqchen/data/pdf/BoostedTree.pdf)

【3】[对xgboost的理解 - 金贵涛的文章 - 知乎]( https://zhuanlan.zhihu.com/p/75217528)

【4】[CTR预估 论文精读(一)--XGBoost](https://blog.csdn.net/Dby_freedom/article/details/84301725)

【5】[XGBoost论文阅读及其原理 - Salon sai的文章 - 知乎]( https://zhuanlan.zhihu.com/p/36794802)

【6】[XGBoost 论文翻译+个人注释](https://blog.csdn.net/qdbszsj/article/details/79615712)

**XGBoost算法讲解：**

【7】[XGBoost超详细推导，终于有人讲明白了！](https://mp.weixin.qq.com/s/wLE9yb7MtE208IVLFlZNkw)

【8】[终于有人把XGBoost 和 LightGBM 讲明白了，项目中最主流的集成算法！](https://mp.weixin.qq.com/s/LoX987dypDg8jbeTJMpEPQ)

【9】[机器学习算法中 GBDT 和 XGBOOST 的区别有哪些？ - wepon的回答 - 知乎](https://www.zhihu.com/question/41354392/answer/98658997) 

【10】[GBDT算法原理与系统设计简介，wepon](http://wepon.me/files/gbdt.pdf)

**XGBoost实例：**

【11】[Kaggle 神器 xgboost](https://www.jianshu.com/p/7e0e2d66b3d4)

【12】[干货 | XGBoost在携程搜索排序中的应用](https://mp.weixin.qq.com/s/X4K6UFZPxL05v2uolId7Lw)

【13】[史上最详细的XGBoost实战 - 章华燕的文章 - 知乎]( https://zhuanlan.zhihu.com/p/31182879)

【14】[XGBoost模型构建流程及模型参数微调（房价预测附代码讲解） - 人工智能学术前沿的文章 - 知乎]( https://zhuanlan.zhihu.com/p/61150141)

**XGBoost面试题：**

【15】[珍藏版 | 20道XGBoost面试题，你会几个？(上篇)](https://mp.weixin.qq.com/s/_QgnYoW827GDgVH9lexkNA)

【16】[珍藏版 | 20道XGBoost面试题，你会几个？(下篇](https://mp.weixin.qq.com/s/BbelOsYgsiOvwfwYs5QfpQ))

【17】[推荐收藏 | 10道XGBoost面试题送给你](https://mp.weixin.qq.com/s/RSQWx4fH3uI_sjZzAKVyKQ)

【18】[面试题：xgboost怎么给特征评分？](https://mp.weixin.qq.com/s/vjLPVhg_UavZIJrOzu_u1w)

【19】[[校招-基础算法]GBDT/XGBoost常见问题 - Jack Stark的文章 - 知乎]( https://zhuanlan.zhihu.com/p/81368182)

【20】《百面机器学习》诸葛越主编、葫芦娃著，P295-P297。

【21】[灵魂拷问，你看过Xgboost原文吗？ - 小雨姑娘的文章 - 知乎]( https://zhuanlan.zhihu.com/p/86816771)

【22】[为什么xgboost泰勒二阶展开后效果就比较好了呢？ - Zsank的回答 - 知乎](https://www.zhihu.com/question/277638585/answer/522272201) 