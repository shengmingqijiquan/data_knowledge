---

title: 机器学习系列之决策树算法（05）：梯度提升树算法GBDT
date: 2019.12.24 14:08 
tags: 

	- GBDT 

categories: 

	- Machine Learning

	- GBDT

keywords: GBDT
description: GBDT

---

## 1 前言

前面讲述了[《决策树的特征选择》](https://dataquaner.github.io/2019/12/17/机器学习系列之决策树算法（01）：决策树特征选择/)、[《决策树的生成》](https://dataquaner.github.io/2019/12/19/机器学习系列之决策树算法（02）：决策树的生成/)、[《决策树的剪枝》](https://dataquaner.github.io/2019/12/19/机器学习系列之决策树算法（03）：决策树的剪枝/)，熟悉了单棵决策树的的实现细节，在实际应用时，往往采用多棵决策树组合的形式完成目标任务。那么如何组合单棵决策树可以使得模型效果更优呢？目前主要有两种思想：**bagging**和**boosting**，分别对应的典型算法**随机森林**和**Adaboost**、**GBDT**等。

> **Bagging**的思想比较简单，即每一次从原始数据中根据**均匀概率分布有放回的抽取和原始数据大小相同的样本集合**，样本点可能出现重复，然后对每一次产生的训练集构造一个分类器，再对分类器进行组合。典型实现算法**随机森林**
>
> **boosting**的每一次抽样的**样本分布都是不一样的**。每一次迭代，都根据上一次迭代的结果，**增加被错误分类的样本的权重**，使得模型能在之后的迭代中更加注意到难以分类的样本，这是一个**不断学习的过程，也是一个不断提升**的过程，这也就是boosting思想的本质所在。迭代之后，将每次迭代的基分类器进行集成。那么如何进行样本权重的调整和分类器的集成是我们需要考虑的关键问题。典型实现算法是**GBDT**

boosting的思想如下图：

![boosting思想](https://pic4.zhimg.com/80/v2-aca3644ddd56abe1e47c0f45601587c3_hd.jpg)

基于boosting思想的经典算法是**Adaboost**和**GBDT**。关于Adaboost的介绍可以参考《Adaboost算法》，本文重点介绍GBDT。

## 2 什么是GBDT

> GBDT(Gradient Boosting Decision Tree) 是一种迭代的决策树算法，是**回归树**，而不是分类树。该算法由多棵决策树组成，所有树的结论累加起来做最终答案。它在被提出之初就和SVM一起被认为是泛化能力较强的算法。
>
> GBDT的思想使其具有天然优势可以发现多种有区分性的特征以及特征组合。业界中，Facebook使用其来自动发现有效的特征、特征组合，来作为LR模型中的特征，以提高 CTR预估（Click-Through Rate Prediction）的准确性。

GBDT用来做回归预测，调整后也可以用于分类。Boost是"提升"的意思，一般Boosting算法都是一个迭代的过程，每一次新的训练都是为了改进上一次的结果。具体训练过程如下图示意：

![GBDT训练过程](https://pic2.zhimg.com/80/v2-4713a5b63da71ef5afba3fcd3a65299d_hd.jpg)

## 3 GBDT算法原理

GBDT算法的核心思想

> GBDT的核心就在于：**每一棵树学的是之前所有树结论和的残差**，这个残差就是一个加预测值后能得真实值的累加量。即所有弱分类器相加等于预测值，下一个弱分类器去拟合误差函数对预测值的梯度。

> GBDT加入了简单的**数值优化**思想。
>
> **Xgboost**更加有效应用了数值优化。相比于gbdt，最重要是对损失函数变得更复杂。目标函数依然是所有树想加等于预测值。损失函数引入了一阶导数，二阶导数。
>
> 不同于随机森林所有树的预测求均值，GBDT所有的树的预测值加起来是最终的预测值，可以不断接近真实值。

GBDT也是集成学习Boosting家族的成员，但是却和传统的Adaboost有很大的不同。回顾下Adaboost，是利用前一轮迭代弱学习器的误差率来更新训练集的权重，这样一轮轮的迭代下去。GBDT也是迭代，使用了前向分布算法，但是弱学习器限定了只能使用CART回归树模型，同时迭代思路和Adaboost也有所不同。

在GBDT的迭代中，假设我们前一轮迭代得到的强学习器是ft−1(x), 损失函数是L(y,ft−1(x)), 我们本轮迭代的目标是找到一个CART回归树模型的弱学习器ht(x)，让本轮的损失损失L(y,ft(x)=L(y,ft−1(x)+ht(x))最小。也就是说，本轮迭代找到决策树，要让样本的损失尽量变得更小。

GBDT的思想的通俗解释

> 假如有个人30岁，
>
> 第一棵树，我们首先用20岁去拟合，发现损失有10岁，
>
> 第二颗，这时我们用6岁去拟合剩下的损失，发现差距还有4岁，
>
> 第三颗，我们用3岁拟合剩下的差距，差距就只有一岁了。
>
> **三棵树加起来为29岁，距离30最近。**

从上面的例子看这个思想还是蛮简单的，但是有个问题是这个损失的拟合不好度量，损失函数各种各样，怎么找到一种通用的拟合方法呢？

## 4 **负梯度拟合**

在上一节中，我们介绍了GBDT的基本思路，但是没有解决**损失函数拟合方法**的问题。针对这个问题，大牛**Freidman**提出了用损失函数的负梯度来拟合本轮损失的近似值，进而拟合一个CART回归树。第t轮的第i个样本的损失函数的负梯度表示为

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyMP27fEskIYa0Y00VyUqTGZLvXic6rwLTApiaqawpGBqoY1b4zNNTGwAw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

利用(xi,rti)(i=1,2,..m),我们可以拟合一颗CART回归树，得到了第t颗回归树，其对应的叶节点区域Rtj,j=1,2,...,J。其中J为叶子节点的个数。

针对每一个叶子节点里的样本，我们求出使损失函数最小，也就是拟合叶子节点最好的的输出值ctj如下：

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyCHtHNTNtpZHNxboDKqMzy43MyLicZFOt8A46iajZMSHbEAW4UEMeoIhw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

这样就得到了本轮的决策树拟合函数如下：

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8ly8Wty7SEqX3Z7MNpiaArS5uNYUu53sb4dp7TsHQMe5Rraw2ZjtbmH84g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

从而本轮最终得到的强学习器的表达式如下：

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyOz13MCp5uicnZkqmXQpMubJAuFndxSJ7fzycvBicyZdwnDgoez4ZXbBQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

通过损失函数的负梯度来拟合，找到了一种通用的拟合损失误差的办法，这样无轮是分类问题还是回归问题，我们通过其损失函数的负梯度的拟合，就可以用GBDT来解决我们的分类回归问题。区别仅仅在于损失函数不同导致的负梯度不同而已。

传统模型中，我们定义一个固定结构的函数，然后通过样本训练拟合更新该函数的参数，获得最后的最优函数。

GBDT提升树并非如此。它是加法模型，是不定结构的函数，通过不断加入新的子函数来使得模型能更加拟合训练数据，直到最优。函数更新的迭代方式可以写作：![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOHhkSYuFVakKkzs8bV1G1x0kTAtekib1cxFnKxQ6Kic59f53ckjEnM8MQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)。所以如果要更快逼近最优的函数，我们就需要在正确的方向上添加子函数，这个“正确的方向”当然就是损失减少最快的方向。所以我们需要用损失函数![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOvzmypoOy2AgFtciavA7xoa2n0JWZd5X30lGibWLBSYHR4Mp3vQXc24xA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)对函数![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOq82zrPJcdR69oOdqjadV52MHoDXRUA3ickHfwRPMLwD8DJINtj20Fpg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)求导（注意不是对x求导），求得的导数，就是接下来![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOq82zrPJcdR69oOdqjadV52MHoDXRUA3ickHfwRPMLwD8DJINtj20Fpg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)需要弥补的方向。在上式中![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOwoQiciaNbtJwTJcKw0EcEwuEwkBAnh9cp72mIAFhOfXM5Wk86ywWorYg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)就是表示导数的拟合。

导数值跟损失函数的选择有关系。如果选择平方损失误差![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOS3I0b3IJ8V0xxjVob1ol6YvFKklAOsnqa1HlIdFicbPuzsnFfd9hPDg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，那么它的导数就是：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOh5pz5Hy4euGM4ohUKWRQAAQn3z1l3QE7I1OCfrqGnbPo0rGBic8L2Vw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

令人惊喜的是这正是真实值和估计值之间的残差！ 这就是为什么谈到GBDT的时候，很多文章都提到“残差”的拟合，却没有说“梯度”的拟合。其实它们在平方损失误差条件下是一个意思！BTW，上面之所以用了![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOu0blrBa87g5KTo3JKNJG3bISFc303NjWothbmK3SsSs5ibIUn3nIH9g/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)是为了计算方便，常数项并不会影响平方损失误差，以及残差的比较。

现在让我们重新理解这个式子：![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOHhkSYuFVakKkzs8bV1G1x0kTAtekib1cxFnKxQ6Kic59f53ckjEnM8MQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

1）先求取一个拟合函数Fm-1(x)

2）用Fm-1(x)进行预测，计算预测值和实际值的残差

3）为了弥补上面的残差，用一个函数△F(x)来拟合这个残差

4）这样最终的函数就变成了![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOHhkSYuFVakKkzs8bV1G1x0kTAtekib1cxFnKxQ6Kic59f53ckjEnM8MQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，其中Fm-1(x)用来拟合原数据，△F(x)用来拟合残差

5）如果目前还有较大的残差，则循环2)~4)，更新函数到Fm+1(x) , Fm+2(x), .....直到残差满足条件。

针对以上流程，我们用实例来说明

## **5 提升树的生成过程**

有以下数据需要用回归，并要求平方损失误差小于0.2（这0.2就是我们人为设置的最优条件，否则训练可能会无休止地进行下去）时，可以停止建树：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcO5GxKf7OxmtUG46swUJwHNyUFv8VOOpj0ShaibKVlPciaPk7lk6O9l4DA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**第一棵树**

**1） 遍历各个切分点s=1.5,2.5,…,9.5找到平方损失误差最小值的切分点：**

比如s=1.5,分割成了两个子集：![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOQeodYeygMxPDwt2E8fP7ic0rwiatLYvcag6VOas4WOvDxeA7H0bxHxiag/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1) 通过公式![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOXTFAGhmwVnVjIAib047xJn00uibicmZcZjxq5YKy6olDr62Eac625L0tA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)求平方损失误差

而其中![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOAvj5NaNw2gmlQ76SqdDOFpTycMKHRr3cyFlpMnWaJT3eF6HpHE54pQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)为各自子集的平均值![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcO8o2CmsIo8IgPBCRHeMeibaUsrWlfYUn1E72DQOdbq86GibuXhBRQJW2w/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)时，可以使得每个子集的平方损失误差最小。

求平均值为：![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOiaYE1zAZnMhO7xiayKZvQmDWFrepYQWftLc2NB5b6LuXpd289gvQOGFw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)，进而求得平方损失误差为![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOZ1zYYIfBrETSG24JJyLwUYqWqVeIz7TV0l1Rn5DtQCfiaMiaBQUfkjuA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

同样的方法求得其它切分点的平方损失误差，列表入下：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOvkj8fRtjicJNMdXIKRDpUawqevqdwWQSYgiamJTY2hcT7KESYOlW8EnA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

可见，当s=6.5时,![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOxveLkvcSxckAMlbqx4I6xtfnEOd8EL9IjZE8DXwCkpqWicR1aAf1MEw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)为所有切分点里平方损失误差最小的

**2) 选择切分点s=6.5构建第一颗回归树，各分支数值使用**

*![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOCng6RAzOkxCibd02ZnfLXHapLY1rxGWJiaXIA1vVyB5VvtBibiadZCvibkw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)：*

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcO0sGV4htSbcmYA79VtOVKIick610xGZfoOyIWePX3FOT75yn6j551Uog/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**第一轮**过后，我们提升树为:

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOsDfeU5cr9IsT5c1rgN8mnibodsaT6KkiaPB67thj2dZrvvCm7828ocjA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**3) 求提升树拟合数据的残差和平方损失误差：**

提升树拟合数据的残差计算：![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOTRzru4dDaCMSFjszEicw3ibPgTKobG5jQL71BISIUeiciasnFUicqKic4E0A/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

各个点的计算结果：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOQYbumw3ZlTVFWP9ZzXcMkoazH2F75AoQZYWxkdPtpfh8IE9uPibRV4g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

提升树拟合数据的平方损失误差计算：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcO3M9WFJP7O5lJsyW8F5ajrLeOiak5iaRsBylu9NT3uIhSNqC6FRw80rqQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

大于0.2，则还需要继续建树。

**第二棵树**

**4) 确定需要拟合的训练数据为上一棵树的残差：**

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOQYbumw3ZlTVFWP9ZzXcMkoazH2F75AoQZYWxkdPtpfh8IE9uPibRV4g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**5） 遍历各个切分点s=1.5,2.5,…,9.5找到平方损失误差最小值的切分点：**

同样的方法求得其它切分点的平方损失误差，列表入下：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOAnbNhuZUsHQHo0F7GVl0GU3Michbmyia3b8iatic03xJ4mOdMIouPqmRyg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

可见，当s=3.5时,![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOX0uGb3louwe22OkRoQdbd3nk722bLU9WuNnFscqq5H3TlXxrjkfhVg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)为所有切分点里平方损失误差最小的

**6) 选择切分点s=3.5构建第二颗回归树，各分支数值使用**

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOjJLm5bHOFULrY4xFYZZzWlfnENVicnxhQYOiaTd3hOWJJjIgt4cXibPwA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOamWhe3GALv8PnuCOca6w70P1lzYjY25KRYrWjVMwY6Pxn11QSwH6fg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

第二轮过后，我们提升树为:

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOYeITOyhABGetdc4z7STdupyzS9fuFa6v95NjpP5he9nzecPDVKOVLA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**7) 求提升树拟合数据的残差和平方损失误差：**

提升树拟合数据的残差计算：![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOlmY5ic3Nfg5VPRFVSeup8xjGudqGdAzI3j5bNWslcRR62AxV2XRYKqQ/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

各个点的计算结果，同时对比初始值和上一颗树的残差：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOt6HhibiciaSQazVLKP3lv4kLvzLF5wfaV2nuCYohOJEU2VkI4WgnZC74w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

可以看见，随着树的增多，残差一直在减少。

到目前为止，提升树拟合数据的平方损失误差计算：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcO5Rzh6ibyicU5VfBHyhsePVC2wQ8LlJeDn8jWcWicKNiarRWtzSF5uprFhg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

多说一句，这里是从全局提升树的角度去计算损失，其实和上面第5）步中从最后一颗树的角度去计算损失，结果是一样的

目前损失大于0.2的阈值，还需要继续建树

… 

… 


**第六棵树**

到第六颗树的时候，我们已经累计获得了：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOsh5duEdgszicGk8UbiafAta5NnHyMNVLYOAiaT0Ju4DLGKibDYPteXYBUA/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)     ![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOUVO2SLfYFcibbOuMyHjICibYBh2EJrIict8lrtBTZWmfCaBFVE628Lypw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOr1ChtQiaO9CXxNCCGSjUDHrz1iayesKv8vMSCGiavrcJhsRXfSyCicJ30A/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)     ![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcODicRF1aptDuvhgKNadZHkhGOsIre4xibUuG60I4miaUkw5LNfo71gS9xw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

此时提升树为：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOv42KBJrAXur7luickhNHjCagFicZAiaTspyoEaibwhNRz2vr5EjibCjSctg/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

此时用![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOiaxnBL9gfbJ7zfDblcMKKgbwsm6iaTbnfR1MGsva6fJ6ypBUkT9poxUw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)拟合训练数据的平方损失误差为：

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOQR444ODwtuSps7BDcbJGPXg70DVN6u5kHLojyQ9qprb27slQnbDV9A/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

平方损失误差小于0.2的阈值，停止建树。

![img](https://mmbiz.qpic.cn/mmbiz_png/mqaP0ypnYKjPEIdtO1Jevr25pRyOXYcOiaxnBL9gfbJ7zfDblcMKKgbwsm6iaTbnfR1MGsva6fJ6ypBUkT9poxUw/640?wx_fmt=gif&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)为我们最终所求的提升树。

## 6 回归算法

**输入：** 最大迭代次数T, 损失函数L，训练样本集

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyodcqQg8f6LDBSria8Wa2zKKrJ313X0ulTbVHBx2cCNwBqdaQWWrT5ug/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**输出：** 强学习器f(x)

**1）** 初始化弱学习器

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyNt6z0iafAWflN2BF8dBd4nlZNC5icuhiaoyAeqQxmur7BN4SEp7cN3k1w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

**2）**对迭代轮数t=1,2,...T有：



　  **a)** 对样本i=1,2，...m，计算负梯度

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyMP27fEskIYa0Y00VyUqTGZLvXic6rwLTApiaqawpGBqoY1b4zNNTGwAw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

　  **b)** 利用(xi,rti)(i=1,2,..m), 拟合一颗CART回归树,得到第t颗回归树，其对应的叶子节点区域为Rtj,j=1,2,...,J。其中J为回归树t的叶子节点的个数。

　 **c)** 对叶子区域j =1,2,..J,计算最佳拟合值

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyobJ7nYicsGlUQibywvuCuXXYEIo2XNIVF6Qtz5FeQFaMGFCqYnSnYcSQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

   **(d)** 更新强学习器

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyOz13MCp5uicnZkqmXQpMubJAuFndxSJ7fzycvBicyZdwnDgoez4ZXbBQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)



**3）** 得到强学习器f(x)的表达式



![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyB0YrMT8hPj4HkNiacdM1iaBIXQgRP1YKxibibgMcCht1hSJooCuIfxMEfg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## 7 分类算法

GBDT的分类算法从思想上和GBDT的回归算法没有区别，但是由于样本输出不是连续的值，而是离散的类别，导致我们无法直接从输出类别去拟合类别输出的误差。

为了解决这个问题，主要有两个方法，

**1）一个是用指数损失函数，此时GBDT退化为Adaboost算法。**

**2）另一种方法是用类似于逻辑回归的对数似然损失函数的方法。**

也就是说，我们用的是类别的预测概率值和真实概率值的差来拟合损失。本文仅讨论用对数似然损失函数的GBDT分类。而对于对数似然损失函数，我们又有二元分类和多元分类的区别。

### 7.1 二元分类算法

对于二元GBDT，如果用类似于逻辑回归的对数似然损失函数，则损失函数为：

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyor0u8iatLzt15YpczNnbOsoRr1sEr2RvP3jTWs8qQgAGZgrhYKhbiaEw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中y∈{−1,+1}。则此时的负梯度误差为

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyiaKu2ichtrUfyDRkFbwibz1WyxNxLK62ePp2OMyKcGd3eupg2jGptoJzg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

对于生成的决策树，我们各个叶子节点的最佳残差拟合值为

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyzmvLpfZt9ACruqtJct6Mdic2x1ibt92yducrmTWtCO5qg8XvDDmBIsLg/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由于上式比较难优化，我们一般使用近似值代替

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyd3gAujuSrODXLfhr0ODJ2QJ7r1AG9KY4FpMvsP8JHLw06qS4Xx3sUA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

除了负梯度计算和叶子节点的最佳残差拟合的线性搜索，二元GBDT分类和GBDT回归算法过程相同。



### 7.2 **多元分类算法**

多元GBDT要比二元GBDT复杂一些，对应的是多元逻辑回归和二元逻辑回归的复杂度差别。假设类别数为K，则此时我们的对数似然损失函数为：

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lynUXdO0WoeZx3kmacQGpC0vj9ny3ageHT4BcLG4sxJ3PeUwZY2EaTvA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

其中如果样本输出类别为k，则yk=1。第k类的概率pk(x)的表达式为：

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyfGo24kcicXMRRpfZrwUdy5MCEmttkCkL18kBibEryicNmlGiba4wgjBn8g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

集合上两式，我们可以计算出第t轮的第i个样本对应类别l的负梯度误差为

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyjDA8kSib3lX4UPdal2YQNADJibOQdec9gyRod0oXd01h37WWet50QfXA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

对于生成的决策树，我们各个叶子节点的最佳残差拟合值为

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lytoTyickEpRJluwyooIWm65M2vbM8yVf2LRwMibMUacYzx3N9EHRN3UtA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

由于上式比较难优化，我们一般使用近似值代替

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyyw8DyXqxF36pAWYHEh7AdB0BHzy8OJvBCBaygzicdUUeclNrLheGN3w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

除了负梯度计算和叶子节点的最佳残差拟合的线性搜索，多元GBDT分类和二元GBDT分类以及GBDT回归算法过程相同。

## 8 **正则化**

和Adaboost一样，我们也需要对GBDT进行正则化，防止过拟合。

GBDT的正则化主要有三种方式。

**第一种是和Adaboost类似的正则化项**，即**步长(learning rate)**。定义为ν,对于前面的弱学习器的迭代

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyDzGSAZPzHsznm49bIdewQ5CibDjbjUK37E0BzhIe7Szcr0lRTx8Oib3A/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

如果我们加上了正则化项，则有

![img](https://mmbiz.qpic.cn/mmbiz_png/KdayOo3PqHCMaFT1BjrnWicmQzJOrs8lyhicpZnfvsEAUeINAqfLibtT0qNzVxy6LpDmbh6oBibmWOJhQgKNSXGK0g/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

ν的取值范围为0<ν≤1。对于同样的训练集学习效果，较小的ν意味着我们需要更多的弱学习器的迭代次数。通常我们用步长和迭代最大次数一起来决定算法的拟合效果。

**第二种正则化的方式是通过子采样比例（subsample）。**取值为(0,1]。注意这里的子采样和随机森林不一样，随机森林使用的是放回抽样，而这里是不放回抽样。如果取值为1，则全部样本都使用，等于没有使用子采样。如果取值小于1，则只有一部分样本会去做GBDT的决策树拟合。选择小于1的比例可以减少方差，即防止过拟合，但是会增加样本拟合的偏差，因此取值不能太低。推荐在[0.5, 0.8]之间。**使用了子采样的GBDT有时也称作随机梯度提升树(Stochastic Gradient Boosting Tree, SGBT)**。由于使用了子采样，程序可以通过采样分发到不同的任务去做boosting的迭代过程，最后形成新树，从而减少弱学习器难以并行学习的弱点。 

**第三种是对于弱学习器即CART回归树进行正则化剪枝。**在决策树原理篇里我们已经讲过，这里就不重复了

## 9 总结

GDBT本身并不复杂，不过要吃透的话需要对集成学习的原理，决策树原理和各种损失函树有一定的了解。由于GBDT的卓越性能，只要是研究机器学习都应该掌握这个算法，包括背后的原理和应用调参方法。目前GBDT的算法比较好的库是xgboost。当然scikit-learn也可以。

**优点**

**1)** 可以灵活处理各种类型的数据，包括连续值和离散值。

**2)** 在相对少的调参时间情况下，预测的准备率也可以比较高。这个是相对SVM来说的。

**3）**使用一些健壮的损失函数，对异常值的鲁棒性非常强。比如 Huber损失函数和Quantile损失函数。

**缺点**

**1)** 由于弱学习器之间存在依赖关系，难以并行训练数据。不过可以通过自采样的SGBT来达到部分并行。

