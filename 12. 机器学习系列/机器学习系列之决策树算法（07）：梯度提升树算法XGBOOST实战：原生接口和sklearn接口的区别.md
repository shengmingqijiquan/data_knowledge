---

title: 机器学习系列之决策树算法（07）：梯度提升树算法XGBoost实战：原生接口和sklearn接口区别
date: 2019.12.27
tags: 

	- XGBoost 

categories: 

	- Machine Learning

	- XGBoost 

keywords: XGBoost 
description: XGBoost 

---

# 1 前言



# 2 官方文档

**[英文官方文档](https://xgboost.readthedocs.io/en/latest/)**

**[中文文档](https://xgboost.apachecn.org/#/xgboost.apachecn.org)**

# 3 sklearn接口

```python
from xgboost.sklearn import XGBClassifier
xgbc = XGBClassifier(n_jobs=-1)  # 新建xgboost sklearn的分类class
# xgboost的sklearn接口默认只使用cpu单线程，设置n_jobs=-1使用所有线程

print("开始xgboost classifier训练")
xgbc.fit(train_vector,np.array(train_label))
# 喂给分类器训练numpy形式的训练特征向量和标签向量
    
print("完成xgboost classifier训练，开始预测")
pre_train_Classifier = xgbc.predict(test_vector)   # 喂给分类器numpy形式的测试特征向量
np.save(os.path.join(model_path,"pre_train_Classifier.npy"),pre_train_Classifier)  # 保存结果
```

xgboost的sklearn接口，可以不经过标签标准化(即将标签编码为0~n_class-1)，直接喂给分类器特征向量和标签向量，使用fit训练后调用predict就能得到预测向量的预测标签，它会在内部调用sklearn.preprocessing.LabelEncoder()将标签在分类器使用时transform，在输出结果时inverse_transform。

**优点：使用简单，无需对标签进行标准化处理，直接得到预测标签；**

**缺点：在模型保存后重新载入，丢失LabelEncoder，不能增量训练只能用一次.**

# 4 xgboost的原生接口

```python
vector_matrix,label_single_new = get_data(data_path) # 获取得到特征矩阵、标签向量
print("标签总数为：%d；数据量总数为：%d"%(len(list(set(label_single_new))),len(vector_matrix)))

# 将标签标准化为0~class number-1,则xgboost概率最大的下标即为该位置数对应的标签
from sklearn import preprocessing
label_coder = preprocessing.LabelEncoder()
label_single_code = label_coder.fit_transform(label_single_new)

# 切割训练集、测试集
from sklearn.model_selection import train_test_split
train_matrix,test_matrix,train_label,test_label = train_test_split(
        vector_matrix,label_single_code,test_size=0.1,random_state=0)

import xgboost as xgb
# 参数设置见 http://www.huaxiaozhuan.com/%E5%B7%A5%E5%85%B7/xgboost/chapters/xgboost_usage.html
params = {
'booster': 'gbtree',
'silent':0,                    # 如果为 0（默认值），则表示打印运行时的信息；如果为 1，则表示不打印这些信息
'objective': 'multi:softprob', # 基于softmax 的多分类模型，但是它的输出是一个矩阵：ndata*nclass，给出了每个样本属于每个类别的概率。
'num_class':len(set(label_single_new)),#指定类别数量
}
dtrain = xgb.DMatrix(train_matrix, label=train_label, nthread=-1)
# xgboost原生接口需要使用DMatrix格式的数据，这里与sklearn接口不同

print("开始xgboost训练")
xgbc = xgb.train(params,dtrain)  # 初始化xgboost分类器，原生接口默认启用全部线程
xgbc.save_model(model_path+save_name+'xgbc_0.9.model') # 保存模型 
# =============================================================================
#     xgbc = xgb.Booster()  # 重新载入模型
#     xgbc.load_model(fname=model_path+save_name+'xgbc_0.9.model')
# =============================================================================

print("xgboost训练完成，得到概率矩阵")
pre_train = xgbc.predict(xgb.DMatrix(train_matrix, nthread=-1))   # 训练数据的预测概率矩阵，启用全部线程
pre_test = xgbc.predict(xgb.DMatrix(test_matrix, nthread=-1))     # 测试数据的预测概率矩阵，启用全部线程
# 概率矩阵各行的数据为各条数据的预测概率，各行数据之和为1；
# 概率矩阵各行的下标即为标准化后的label标签(0~class number-1)

# 数据保存
np.save(model_path+save_name+'pre_train.npy',pre_train)
np.save(model_path+save_name+'train_label.npy',train_label)  
np.save(model_path+save_name+'pre_test.npy',pre_test)
np.save(model_path+save_name+'test_label.npy',test_label)  

# 数据载入
# =============================================================================
# pre_train = np.load(model_path+save_name+'pre_train.npy') 
# train_label = np.load(model_path+save_name+'train_label.npy') 
# pre_test = np.load(model_path+save_name+'pre_test.npy') 
# test_label = np.load(model_path+save_name+'test_label.npy') 
# =============================================================================

# narray_target.argsort(axis=1)，获得按行(排序对象为各行数值)升序后的下标矩阵，axis=0为按列升序;
# np.fliplr(narray_target)获取矩阵的左右翻转，narray_target[::-1]获取矩阵的上下翻转
# narray_target[:,-5:]获取矩阵的后5列;
top_k = 5  # 获取预测概率最大的5个标签
# 获取概率矩阵排序信息，得到按行升序的下标矩阵,切割得到各行的后5个下标,
# 将其左右翻转后，得到各行降序的前5个下标，即标准化后的标签
pre_test_index = np.fliplr(pre_test.argsort(axis=1)[:,-1*top_k:])
pre_test_label = label_coder.inverse_transform(pre_test_index)
# 调用label标准化工具inverse_transform将下标转化为真实标签

pre_train_index = np.fliplr(pre_train.argsort(axis=1)[:,-1*top_k:])
pre_train_label = label_coder.inverse_transform(pre_train_index)        
```

xgboost原生接口，数据需要经过标签标准化(LabelEncoder().fit_transform)、输入数据标准化(xgboost.DMatrix)和输出结果反标签标准化(LabelEncoder().inverse_transform)，训练调用train预测调用predict.

需要注意的是，**xgboost原生接口输出的预测标签概率矩阵各行的下标即为标准化后的label标签(0~class number-1).**

# 5 结论

优先考虑使用原生接口形式，便于模型保存后的复用。