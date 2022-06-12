#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import multiprocessing
import random
from collections import defaultdict
from math import ceil, floor

import numpy as np
import pandas as pd
from numba import jit
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

from tree import Tree
from tool.mdlp import MDLPDiscretizer

from sklearn.utils import shuffle

# 第一次运行需要在cmd（windows）中运行以下命令
"""
pip install git+https://github.com/doctorado-ml/odte
pip install git+https://github.com/doctorado-ml/stree
"""
from stree import Stree
from odte import Odte

class Random_Forest(object):

    def __init__(self, n_estimators: int, estimator, n_jobs=-1, rate=0.7):
        '''
        :description: init
        :param: n_estimators: int: 基训练器数量
        :param: estimator: 基训练器对象
        :param: n_jobs: 并行进程数，{-1, 1, ... max_score}, default: -1: cpu核心数
        :param: rate: float: default=1.0, [0.0, 1.0]
        :return: 
        '''
        self.estimator = estimator
        self.n_estimators = n_estimators
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()
        elif 0 < n_jobs < multiprocessing.cpu_count():
            self.n_jobs = n_jobs
        else:
            self.n_jobs = multiprocessing.cpu_count()
        self.rate = rate
        self.estimator_list = list()

    def _voting(self, data):
        '''
        :description: 投票法(适用于分类)
        :param: data: 
        :return: result: 
        '''
        term = np.transpose(data)  # 转置
        result = list()  # 存储结果

        def vote(df):  # 对每一行做投票
            store = defaultdict()
            for kw in df:
                store.setdefault(kw, 0)
                store[kw] += 1
            return max(store, key=store.get)

        result = list(map(vote, term))  # 获取结果
        return result

    def _underSampling(self, data, number):
        '''
        :description: 随机欠采样函数(未完整测试，可能存在bug)
        :param: data: 训练数据和对应标签组成的数据集合
        :return: result: 
        '''
        data = np.array(data)
        np.random.shuffle(data)  # 打乱data
        # 切片，取总数*rata的个数，删去（1-rate）%的样本
        newdata = data[0:int(data.shape[0]*self.rate), :]
        return newdata

    def _weight_sample_index(self, sample_index: list, k: int, weights: list, filter_sample=[]) -> list:
        '''
        :description: 获取带权有放回采样样本索引（暂时未使用）
        :param: sample_index: list: 样本索引
        :param: k: int: k为抽样的个数
        :param: weights: list: 每个样本抽样概率
        :param: filter_sample: list: 需要过滤的样本索引
        :return: result: 抽样后的样本索引
        '''
        max_weight = max(weights)
        p = list(map(lambda x: x / max_weight, weights))
        size = len(sample_index)
        idx = 0
        result = []
        while True:
            if idx >= k:
                return result
            sample_idx = random.randint(0, size - 1)
            pr = random.random()
            if pr <= p[sample_idx]:
                if sample_index[sample_idx] in filter_sample:
                    continue
                result.append(sample_index[sample_idx])
                idx += 1

    def _weight_sampe(self, x_train, y_train, weights, k,  filter_sample=[]):
        '''
        :description: 带权有放回采样（暂时未使用）
        :param: x_train: 训练数据集
        :param: y_train: 训练数据标签
        :param: k: int: k为抽样的个数
        :param: weights: list: 每个样本抽样概率
        :param: filter_sample: list: 需要过滤的样本索引
        :return: result: 抽样后的样本索引
        '''
        samples = np.column_stack([x_train, y_train])
        samples_index = list(range(samples.shape[0]))
        new_index = self._weight_sample_index(
            samples_index,  k, weights, filter_sample=[])
        new_samples = samples[new_index]
        new_x_train, new_y_train = new_samples[:, :-1], new_samples[:, -1]
        return new_x_train, new_y_train

    @jit
    def _repetitionRandomSampling(self, data, number: int):
        '''
        :description: 简单有放回采样
        :param: data: 训练数据和对应标签组成的数据集合
        :param: number: int:number为抽样的个数
        :return: result: 
        '''
        sample = []
        for i in range(number):
            sample.append(data[random.randint(0, len(data)-1)])
        return sample

    def score(self, y_test, y_predict, average="macro"):
        '''
        :description: 评价函数
        :param: y_test: 真实标签
        :param: y_predict: 预测标签
        :param: average: string: [None, ‘micro’, ‘macro’ (default), ‘samples’, ‘weighted’], 参数意义见：https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html?highlight=precision_score#sklearn.metrics.average_precision_score
        :return: recall, precision ##召回率#查准率
        '''
        recall = recall_score(y_test, y_predict, average=average)  # 召回率
        precision = precision_score(y_test, y_predict, average=average)  # 查准率

        return recall, precision

    def fit(self, x_train, y_train, sample_type="RepetitionRandomSampling"):
        '''
        :description: fit 训练
        :param: x_train: 训练数据集
        :param: y_train: 训练数据集标签
        :param: x_test: 测试数据集
        :param: sample_type: string: 采样方式, 可选"RepetitionRandomSampling" default, "UnderSampling"
        :return:
        '''
        if self.n_jobs != 1:
            self._multi_process_fit(x_train, y_train, sample_type=sample_type)
        else:
            self._single_process_fit(x_train, y_train, sample_type=sample_type)

    def _single_process_fit(self, x_train, y_train, sample_type="RepetitionRandomSampling"):
        '''
        :description: _single_threaded_fit 单进程训练方法
        :param: x_train: 训练数据集
        :param: y_train: 训练数据集标签
        :param: x_test: 测试数据集
        :param: sample_type: string: 采样方式, 可选"RepetitionRandomSampling" default, "UnderSampling"
        :return:
        '''
        result_first = list()
        result_second = list()
        x_train_second = x_train.copy()
        y_train_second = y_train.copy()
        if sample_type == "RepetitionRandomSampling":
            sample_function = self._repetitionRandomSampling
        elif sample_type == "UnderSampling":
            sample_function = self._underSampling
        else:
            sample_function = self._repetitionRandomSampling

        train = np.column_stack([x_train, y_train])
        sample_number = len(train)

        first_train_number = ceil(self.n_estimators * self.rate)

        for i in range(first_train_number):
            estimator = self.estimator
            samples = np.array(sample_function(
                train, sample_number))  # 构建数据集
            x_train = samples[:, 0:-1]
            y_train = samples[:, -1]
            clf = estimator.fit(x_train, y_train)
            if hasattr(clf, "predict"):
                y_train_predict = clf.predict(x_train)
                self.estimator_list.append(clf)
            else:
                y_train_predict = estimator.predict(x_train)
                self.estimator_list.append(estimator)

            x_train_second = np.insert(
                x_train_second, 0, x_train[y_train_predict != y_train], axis=0)
            y_train_second = np.insert(
                y_train_second, 0, y_train[y_train_predict != y_train], axis=0)
            result_first.append(y_train_predict)  # 训练模型 返回每个模型的输出

        second_train_number = floor(self.n_estimators * (1-self.rate))
        second_train = np.column_stack([x_train_second, y_train_second])
        second_sample_number = second_train.shape[0]
        for i in range(second_train_number):
            estimator = self.estimator
            samples = np.array(sample_function(
                second_train, second_sample_number))  # 构建数据集
            x_train = samples[:, 0:-1]
            y_train = samples[:, -1]
            clf = estimator.fit(x_train, y_train)

            if hasattr(clf, "predict"):
                y_train_predict = clf.predict(x_train)
                self.estimator_list.append(clf)
            else:
                y_train_predict = estimator.predict(x_train)
                self.estimator_list.append(estimator)

            result_second.append(y_train_predict)

        return

    def _multi_process_fit(self, x_train, y_train, sample_type="RepetitionRandomSampling"):
        '''
        :description: _multi_process_fit 多进程训练方法
        :param: x_train: 训练数据集
        :param: y_train: 训练数据集标签
        :param: x_test: 测试数据集
        :param: sample_type: string: 采样方式, 可选"RepetitionRandomSampling" default, "UnderSampling"
        :return:
        '''
        samples_first_list = list()
        samples_second_list = list()
        x_train_second = x_train.copy()
        y_train_second = y_train.copy()

        if sample_type == "RepetitionRandomSampling":
            sample_function = self._repetitionRandomSampling
        elif sample_type == "UnderSampling":
            sample_function = self._underSampling
        else:
            sample_function = self._repetitionRandomSampling

        first_train = np.column_stack([x_train, y_train])
        first_sample_number = len(first_train)

        first_train_number = ceil(self.n_estimators * self.rate)
        for i in range(first_train_number):
            samples_first_list.append(np.array(sample_function(
                first_train, first_sample_number)))
        # 多进程
        cores = self.n_jobs
        pool = multiprocessing.Pool(processes=cores)

        pool_list = []
        for sample in samples_first_list:
            pool_list.append(pool.apply_async(self._work, (sample,)))
        
        result_first_list = [xx.get() for xx in pool_list]
        result_first, x_train_second_list, y_train_second_list, self.estimator_list = \
            list(map(lambda x: x[0], result_first_list)),\
            list(map(lambda x: x[1], result_first_list)),\
            list(map(lambda x: x[2], result_first_list)),\
            list(map(lambda x: x[3], result_first_list))
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出
        second_train_number = floor(self.n_estimators * (1-self.rate))
        for x_train_second_item in x_train_second_list:
            x_train_second = np.insert(
                x_train_second, 0, x_train_second_item, axis=0)

        for y_train_second_item in y_train_second_list:
            y_train_second = np.insert(
                y_train_second, 0, y_train_second_item, axis=0)
        second_train = np.column_stack([x_train_second, y_train_second])
        second_sample_number = first_sample_number

        pool = multiprocessing.Pool(processes=cores)
        for i in range(second_train_number):
            samples_second_list.append(np.array(sample_function(
                second_train, second_sample_number)))
        pool_list = []
        for sample in samples_second_list:
            pool_list.append(pool.apply_async(self._work, (sample,)))

        result_second_list = [xx.get() for xx in pool_list]
        result_second = list(map(lambda x: x[0], result_second_list))
        self.estimator_list.extend(
            list(map(lambda x: x[3], result_second_list)))
        pool.close()  # 关闭进程池，不再接受新的进程
        pool.join()  # 主进程阻塞等待子进程的退出
        return

    def _work(self, sample):
        '''
        :description: 多线程工作方法
        :param: sample: 训练数据集
        :return: result eg: [y_train_predict, x_train_second, y_train_second, estimator]
        '''
        x_train = sample[:, 0:-1]
        y_train = sample[:, -1]
        clf = self.estimator.fit(x_train, y_train)
        if hasattr(clf, "predict"):
            y_train_predict = clf.predict(x_train)
            estimator = clf
        else:
            estimator = self.estimator
            y_train_predict = estimator.predict(x_train)

        x_train_second = x_train[y_train_predict != y_train]
        y_train_second = y_train[y_train_predict != y_train]
        result = [y_train_predict, x_train_second, y_train_second, estimator]

        return result

    def predict(self, x):
        '''
        :description: 预测方法
        :param: x: 预测数据集
        :return: 
        '''
        result_list = list()
        for estimator in self.estimator_list:
            result_list.append(estimator.predict(x))
        result = self._voting(result_list)
        return result

def score(y_test, y_predict, average="macro"):
        '''
        :description: 评价函数
        :param: y_test: 真实标签
        :param: y_predict: 预测标签
        :param: average: string: [None, ‘micro’, ‘macro’ (default), ‘samples’, ‘weighted’]
        :return: recall, precision ##召回率#查准率
        '''
        recall = recall_score(y_test, y_predict, average=average)  # 召回率
        precision = precision_score(y_test, y_predict, average=average)  # 查准率

        return recall, precision

def kfold_function(x, y, cols, num, k , n_estimators, max_depth = 10, function="self_rf", average="macro", n_jobs = -1):
    '''
    :description: kFold 方法
    :param: x: 数据集
    :param: y: label
    :param: cols: list： 特征名称
    :param: num: int: for循环次数
    :param: k: int: k阶交叉验证
    :param: n_estimators: int: 基分类器个数
    :param: max_depth: int: 多变量决策树最大深度
    :param: average: str: 评价函数参数
    :param: function: str: ["self_rf", "sk_rf", "s_tree", "odte_rf", "self_s_tree"]，分别使用本项目的随机森林和sklearn自带的随机森林、基于svm节点斜决策树、基于svm节点斜决策树的随机森林、基于svm节点斜决策树的随机森林
    :return: acc 
    '''
    from sklearn.model_selection import KFold
    from sklearn.ensemble import RandomForestClassifier

    acc = []
    for i in range(num):
        accuracy_set = []
        kf = KFold(n_splits=k, shuffle=True, random_state=i)
        for train_index, test_index in kf.split(x, y):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = y[train_index], y[test_index]
            if function == "self_rf":
                estimator = Tree(cols, max_depth=max_depth)  # 基分类器：多变量决策树
                clf = Random_Forest(n_estimators=n_estimators, n_jobs=-1,
                            estimator=estimator, rate=0.7)
                clf.fit(x_train, y_train)
                y_predict = clf.predict(x_test)
                recall, precision = clf.score(y_test, y_predict)
                
            elif function == "self_s_tree":
                estimator = Stree(random_state=i, C=.01)
                clf = Random_Forest(n_estimators=n_estimators, n_jobs=-1,
                            estimator=estimator, rate=1.0)
                clf.fit(x_train, y_train)
                y_predict = clf.predict(x_test)
                recall, precision = clf.score(y_test, y_predict)

            elif function == "s_tree":
                tree = Stree(random_state=i, C=.01)
                tree.fit(x_train, y_train)
                y_predict = tree.predict(x_test)
                _, precision = score(y_test, y_predict)  # 查准率
            
            elif function == "odte_rf":
                odte = Odte(random_state=i, n_jobs = n_jobs, max_features="auto",n_estimators=n_estimators)
                odte.fit(x_train, y_train)
                y_predict = odte.predict(x_test)
                _, precision = score(y_test, y_predict)  # 查准率
            else:
                clf = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1, random_state=i)
                clf.fit(x_train, y_train)
                precision = clf.score(x_test, y_test)
                print("precision:", precision)
            accuracy_set.append(precision)
        print("accuracy_set:", accuracy_set)
        acc_value = np.mean(accuracy_set)
        print("mean accuracy:", acc_value)
        acc.append(acc_value)
    return acc

def test_self():
    """
    test function
    """
    from sklearn import tree
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import shuffle

    le = LabelEncoder()
    base_dir = "/home/jedi/project/0320/"
    df = pd.read_csv("Input/西瓜数据集.csv", index_col=0)
    # df = pd.read_csv(base_dir + "Input/mushrooms.csv")  # 二分类
    new_df = df
    new_df = new_df.reset_index()
    new_df.drop("index", inplace=True, axis=1)
    print(new_df)
    print("-"*30)

    for item in list(new_df.columns):
        new_df[item] = le.fit_transform(new_df[item])
    print(new_df)
    cols = df.columns.tolist()  # 特征名称转list
    feas = cols[1:]
    labels = cols[0]
    data_x = new_df.iloc[:, 1:]
    data_y = new_df.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.1, random_state=120)

    x_train, x_test, y_train, y_test = x_train.values, x_test.values, y_train.values, y_test.values
    # estimator = tree.DecisionTreeClassifier() #　# 基分类器：sklearn 单变量决策树，使用取消注释，并注释下一行代码

    estimator = Tree(cols)  # 基分类器：多变量决策树
    clf = Random_Forest(n_estimators=10, n_jobs=-1,
                        estimator=estimator, rate=0.7)
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_test)
    recall, precision = clf.score(y_test, y_predict=y_predict)

    print("recall:", '\n', recall)
    print("precision", '\n', precision)


def test_sklearn():
    """
    test function (sklearn)
    """
    from sklearn import tree
    from sklearn.ensemble import BaggingClassifier
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import shuffle

    le = LabelEncoder()
    base_dir = "/home/jedi/project/0320/"
    # df = pd.read_csv("Input/西瓜数据集.csv", index_col=0)
    df = pd.read_csv(base_dir + "Input/part_mushrooms.csv")  # 二分类
    new_df = df
    # new_df = pd.concat([df[df["class"] != "p"][:2000], df[df["class"] == "p"][:1900]], )
    new_df = new_df.reset_index()
    new_df.drop("index", inplace=True, axis=1)
    print(new_df)
    print("-"*30)
    # new_df = shuffle(new_df).reset_index(drop=True)
    # print(new_df)

    for item in list(new_df.columns):
        new_df[item] = le.fit_transform(new_df[item])
    print(new_df)
    cols = df.columns.tolist()  # 特征名称转list
    feas = cols[1:]
    labels = cols[0]
    # x = new_df[feas].values
    # y = new_df[labels].values  # 好瓜

    data_x = new_df.iloc[:, 1:]
    data_y = new_df.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.33, random_state=120)
    # estimators = [tree.DecisionTreeClassifier(),AdaBoostClassifier(),tree.DecisionTreeClassifier(max_depth=4)]    #基础模型
    # estimators = [tree.DecisionTreeClassifier()]
    x_train, x_test, y_train, y_test = x_train.values, x_test.values, y_train.values, y_test.values


    # sklearn中 BaggingClassifier
    clf_sklearn = BaggingClassifier(
        base_estimator=tree.DecisionTreeClassifier(), n_estimators=10)
    clf_sklearn.fit(x_train, y_train)
    score = clf_sklearn.predict(x_test)
    recall = recall_score(y_test, score, average="macro")
    precision = precision_score(y_test, score, average="macro")
    print("*******"*10)
    print("sklern.bagging结果")
    print("recall:", '\n', recall)
    print("precision", '\n', precision)


def test_processed_hungarian_kfold(num, k):
    """

    """
    from sklearn.model_selection import KFold
    from sklearn.utils import shuffle

    pd.set_option('display.max_columns', None)
    data = pd.read_csv("/home/jedi/project/0320/Input/processed.hungarian.data", sep=",", header=None)
    data.replace(to_replace='?',value=np.nan,inplace=True) # 替换缺失值
    data.drop([10, 11, 12], axis=1, inplace=True)
    
    
    for col in data.columns:
        data[col] = data[col].astype("float64")

    for col in [3, 4, 7]:
        data[col].fillna(data[col].mean(), inplace=True)# 使用平均数值填充缺失值
    for col in [5, 6, 8]:
        data[col].fillna(data[col].mode()[0], inplace=True)# 使用众数值填充缺失值
    data_x = data.iloc[:, 0:-1]
    data_y = data.iloc[:, -1]
    cols = data_x.columns.tolist()  # 特征名称转list
    print(data_x)
   
    # 离散化
    mdlp = MDLPDiscretizer(
        features=[0, 3, 4, 7], return_intervals=False, return_df=True)
    data_x = mdlp.fit_transform(data_x, data_y)
    print(data_x)
    data_x, data_y = shuffle(data_x, data_y, random_state = 1)
    x, y = data_x.values, data_y.values #pandas DataFrame转numpy array

    acc = kfold_function(x, y, cols, num=1, k=10, function="self_rf", average=None, n_estimators=10)
    print(acc)


if __name__ == "__main__":
    from time import time
    start_time = time()
 
    # test_self()
    # test_sklearn()
    test_processed_hungarian_kfold(2,10)

    print("TIME: ", time()-start_time)
