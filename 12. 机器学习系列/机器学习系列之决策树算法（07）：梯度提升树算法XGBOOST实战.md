---

title: 机器学习系列之决策树算法（07）：梯度提升树算法XGBoost实战
date: 2019.12.26
tags: 

	- XGBoost 

categories: 

	- Machine Learning

	- XGBoost 

keywords: XGBoost 
description: XGBoost 

---

# 1 前言

上一篇从数据原理角度深入介绍了XGBoost的实现原理及优化，参考《[梯度提升树算法XGBoost](https://dataquaner.github.io/2019/12/25/机器学习系列之决策树算法（07）：梯度提升树算法XGBOOST/)》。本篇主要介绍XGBoost的工程实战，参数调优等内容。

> 学习一个算法实战，一般按照以下几步，第一步能够基于某个平台、某种语言构建一个模型，第二步是能够优化一个模型 。我们将学习以下内容
>
> 1. 如果使用xgboost构建分类器
> 2. xgboost 的参数含义，以及如何调参
> 3. xgboost 的如何做cv
> 4. xgboost的可视化

# 2 XGBoost模型构建

## 回归模型

### 准备数据

我们使用**[房价数据](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)** ，做的是一个回归任务，预测房价，分类任务类似。

导入包

```python
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
from sklearn.metrics import mean_absolute_error
```

读入和展示数据

```python
data = pd.read_csv('../input/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])

train_X, test_X, train_y, test_y = train_test_split(X.as_matrix(), y.as_matrix(), test_size=0.25)

my_imputer = Imputer()
train_X = my_imputer.fit_transform(train_X)
test_X = my_imputer.transform(test_X)
print(train_X.shape)
print(test_X.shape)
print(train_y.shape)
print(test_y.shape)
---
##执行结果
(1095, 37)
(365, 37)
(1095,)
(365,)
---
```
### 创建并训练XGBoost模型

随机选取默认参数进行初始化建模

```python
my_model = XGBRegressor()
# Add silent=True to avoid printing out updates with each cycle
my_model.fit(train_X, train_y, verbose=False)
```

### 评估并预测模型

```python
# make predictions
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
```

### 模型调优

XGBoost有一些参数可以显著影响模型的准确性和训练速度。

#### **n_estimators** 

**n_estimators** 指定训练循环次数。在 [欠拟合 vs 过拟合 图表](https://link.zhihu.com/?target=http%3A//i.imgur.com/2q85n9s.png), n_estimators让训练沿着图表向右移动。 值太低会导致欠拟合，这对训练数据和新数据的预测都是不准确的。 太大的值会导致过度拟合，这是对训练数据的准确预测，但对新数据的预测不准确（这是我们关心的）。 通过实际实验来找到理想的n_estimators。 典型值范围为100-1000，但这很大程度上取决于下面讨论的

#### **early_stopping_rounds** 

**early_stopping_rounds** 提供了一种自动查找理想值的方法。 early_stopping_rounds会导致模型在validation score停止改善时停止迭代，即使迭代次数还没有到n_estimators。为**n_estimators**设置一个高值然后使用**early_stopping_rounds**来找到停止迭代的最佳时间是明智的。

存在随机的情况有时会导致validation score无法改善，因此需要指定一个数字，以确定在停止前允许多少轮退化。**early_stopping_rounds = 5**是一个合理的值。 因此，在五轮validation score无法改善之后训练将停止。 以下是early_stopping的代码：

```python
my_model = XGBRegressor(n_estimators=1000)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
```

当使用**early_stopping_rounds**时，需要留出一些数据来检查要使用的轮数。 如果以后想要使所有数据拟合模型，请将**n_estimators**设置为在早期停止运行时发现的最佳值。

#### learning_rate

对于更好的XGBoost模型，这是一个微妙但重要的技巧：

XGBoost模型不是通过简单地将每个组件模型中的预测相加来获得预测，而是在将它们添加之前将每个模型的预测乘以一个小数字。这意味着我们添加到集合中的每个树都不会对最后结果有决定性的影响。在实践中，这降低了模型过度拟合的倾向。

因此，使用一个较大的**n_estimators**值并不会造成过拟合。如果使用early_stopping_rounds，树的数量会被设置成一个合适的值。

通常，较小的learning rate（以及大量的estimators）将产生更准确的XGBoost模型，但是由于它在整个循环中进行更多迭代，因此也将使模型更长时间进行训练。 包含学习率的代码如下：

```python
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model.predict(test_X)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))
```

#### 小结

XGBoost目前是用于在传统数据（也称为表格或结构数据）上构建精确模型的主要算法

```python
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

my_model1 = XGBRegressor()
my_model1.fit(train_X, train_y, verbose=False)
predictions = my_model1.predict(test_X)
print("Mean Absolute Error 1: " + str(mean_absolute_error(predictions, test_y)))



my_model2 = XGBRegressor(n_estimators=1000)
my_model2.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model2.predict(test_X)
print("Mean Absolute Error 2: " + str(mean_absolute_error(predictions, test_y)))

my_model3 = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model3.fit(train_X, train_y,  
             eval_set=[(test_X, test_y)], verbose=False)
predictions = my_model3.predict(test_X)
print("Mean Absolute Error 3: " + str(mean_absolute_error(predictions, test_y)))
```

## 分类模型

以天池竞赛中的[《**快来一起挖掘幸福感！**》](https://tianchi.aliyun.com/competition/entrance/231702/introduction?spm=5176.12281973.1005.1.3dd52448pr3509)中的数据为例，开始一个多分类模型的的实例

#### 导入包

```python
import pandas as pd
from matplotlib import pyplot as plt
import xgboost as xgb
from sklearn.model_selection import learning_curve, train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
```

#### 导入数据

```python
'''  
##         准备训练集和测试集
'''  
data = pd.read_csv('happiness_train_abbr.csv')
y=data['happiness']
data.drop('happiness',axis=1,inplace=True)
data.drop('survey_time',axis=1,inplace=True)#survey_time格式不能直接识别
X=data
```

#### 数据集划分

```python
train_x, test_x, train_y, test_y = train_test_split (X, y, test_size =0.30, early_stopping_rounds=10,random_state = 33)
```

#### XGBoost模型训练

```python
'''  
##         xgboost训练
''' 
params = {'learning_rate': 0.1, 
          'n_estimators': 500, 
          'max_depth': 5, 
          'min_child_weight': 1, 
          'seed': 0, 
          'subsample': 0.8, 
          'colsample_bytree': 0.8,
          'gamma': 0, 
          'reg_alpha': 0, 
          'reg_lambda': 1
         }
#第一次设置300次的迭代，评测的指标是"merror","mlogloss"，这是一个多分类问题。
model = xgb.XGBClassifier(params)
eval_set = [(train_x, train_y), (test_x, test_y)]
model.fit(train_x, train_y, eval_set=eval_set, eval_metric=["merror", "mlogloss"],verbose=True)
predictions = model.predict(test_x)

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))    
accuracy = accuracy_score(test_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

```

#### 模型可视化

```python
'''  
##         可视化训练过程
''' 
results = model.evals_result()
epochs = len(results['validation_0']['merror'])
x_axis = range(0, epochs)
from matplotlib import pyplot
fig, ax = pyplot.subplots(1,2,figsize=(10,5))
ax[0].plot(x_axis, results['validation_0']['mlogloss'], label='Train')
ax[0].plot(x_axis, results['validation_1']['mlogloss'], label='Test')
ax[0].legend()
ax[0].set_title('XGBoost Log Loss')
ax[0].set_ylabel('Log Loss')
ax[0].set_xlabel('epochs')


ax[1].plot(x_axis, results['validation_0']['merror'], label='Train')
ax[1].plot(x_axis, results['validation_1']['merror'], label='Test')
ax[1].legend()
ax[1].set_title('XGBoost Classification Error')
ax[1].set_ylabel('Classification Error')
ax[1].set_xlabel('epochs')
pyplot.show()
```

<img src="C:\Users\liyu25\AppData\Roaming\Typora\typora-user-images\image-20191226184624206.png" alt="模型迭代结果" style="zoom: 80%;" />

实际训练效果，在第146次迭代就停止了，说明最好的效果实在136次左右。根据许多大牛的实践经验，选择**early_stopping_rounds = 10% * n_estimators**。

最终输出模型最佳状态下的结果：

```python
print ("best iteration:",model.best_iteration)
limit = model.best_iteration
predictions = model.predict(test_x,ntree_limit=limit)
print("Mean Absolute Error : " + str(mean_absolute_error(predictions, test_y)))    
accuracy = accuracy_score(test_y, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

## 3 参考资料

[https://www.kaggle.com/dansbecker/xgboost](https://link.zhihu.com/?target=https%3A//www.kaggle.com/dansbecker/xgboost)

[https://blog.csdn.net/lujiandong1/article/details/52777168](https://link.zhihu.com/?target=https%3A//blog.csdn.net/lujiandong1/article/details/52777168)