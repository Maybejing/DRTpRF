#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import warnings
from collections import Counter
from time import time
import random
import numpy as np
import pandas as pd
from numba import jit

warnings.filterwarnings("ignore")  # 不显示警告


class Tree(object):

    def __init__(self,
                 feas: list,
                 privacy_e = 0.1,
                 delete_rate = 0.1,
                 sample_rate=0.95,
                 max_depth=10,
                 name=None,
                 beta=0.9,
                 eps=0.1,
                 criterion="None"):
        '''
        :description:  决策树类
        :param: feas: 类别名（特征名）
        :param: delete_rate: 删除率
        :param: sample_rate: the sample ratio
        :param: max_depth: 树最大深度
        :param: name: 决策树名, 默认"Decision Tree"
        :param: name: beta: float: 取值阈值
        :param: eps: 阈值 # 扩展参数(暂时未使用)
        :param: criterion: 构建决策树算法, 默认"None"：多变量决策树数生成算法 # 扩展参数(暂时未使用)
        :return: 
        '''

        self.privacy_e= privacy_e
        self.tree_ = dict()  # 保存生成的树
        self.feas = feas
        self.delete_rate = delete_rate
        self.sample_rate = sample_rate
        self.beta = beta
        self.eps = eps
        self.max_depth = max_depth
        self.criterion = criterion
        if not name:
            self.name = "Decision Tree"
        else:
            self.name = name

    def fit(self, x, y):
        '''
        :description: 构建决策树方法
        :param: x: 训练数据集
        :param: y: 训练数据集标签
        :return: tree: 生成的决策树
        '''
        current_depth = 0
        self.tree_ = self._build_tree(x, y, current_depth)
        return self.tree_

    def get_score(self, predict_y, y) -> float:
        '''
        :description: 获取预测精度
        :param: predict_y: 预测数据集标签
        :param: y: 真实数据集标签
        :return: score:float
        '''
        assert predict_y.shape == y.shape, "{predict_y}和{y}的shape不相等"
        count = 0
        for predict, true in zip(predict_y, y):
            if predict == true:
                count += 1
        score = count / len(y)
        return score

    def predict(self, x, x_tree=None):
        '''
        :description: 预测方法
        :param: x: 预测数据集
        :param: x_tree: dict: 决策树dict
        :return: 
        '''
        if len(x.shape) == 2:
            rst = []
            for x_ in x:
                rst.append(self.predict(x_))
            rst = np.array(rst)
            return rst

        if x_tree is None:
            x_tree = self.tree_

        if not isinstance(x_tree, dict):
            return x_tree

        values = x_tree.get("values", None)
        if values is None:  # (可能有问题)
            return "生成树异常: values is None"
        left_tree = x_tree.get("left_tree")
        right_tree = x_tree.get("right_tree")

        threshold = values.get("threshold")
        indexs = values.get("indexs")
        w = values.get("w")
        cal_value = np.dot(w.T, x[indexs])
        if cal_value < threshold:
            if isinstance(left_tree, dict):
                return self.predict(x, left_tree)
            else:
                return left_tree
        else:
            if isinstance(right_tree, dict):
                return self.predict(x, right_tree)
            else:
                return right_tree

    def _find_best_split(self, x, y, feas: list, number=10, delete_rate=None, e_current=None ):
        '''
        :description: 寻找最佳超平面方法
        :param: x: numpy.ndarray: 训练数据集 
        :param: y: numpy.ndarray: 训练数据集标签
        :param: feas: 特征名
        :param: number: 取前{number}个特征
        :param: delete_rate: float: 删除率
        :return: max_gini_threshold, max_gini_w, max_gini_feas
        '''
        d_const=len(feas)
        max_gini = -1           # 最大的gini指数
        max_gini_threshold = 0  # 最大的gini指数对应的超平面阈值
        max_gini_w = 0         # 最大的gini指数对应的权重矩阵
        max_gini_feas = list()
        if delete_rate is None:
            delete_rate = self.delete_rate

        # 删除对称不确定性为0的特征
        delete_list = list()
        for col in range(len(feas)):
            if cal_symmetrical_uncertainty(x[:, col], y) == 0.0:
                delete_list.append(col)
        x = np.delete(x, delete_list, 1)
        new_x = x
        new_y = y
        feas = [feas[i] for i in range(len(feas)) if (i not in delete_list)]

        current_list, previous_list = list(range(len(feas))), list()  # 前后两次权重
        save_w = []
        save_boolean_test_threshold = []
        save_max_feas = []
        save_gini = []
        while True:
            if x.size == 0:
                break
            features_weight_index_list, features_weight_items_list, all_features_weight_index_list = self._top_features_by_number(
                new_x, new_y, feas, number)

            current_list, previous_list = [
                i[1] for i in features_weight_items_list], current_list
            temp_x = x[:, features_weight_index_list]
            w = LDA_dimensionality(x=temp_x, y=new_y, k=1)
            # print("typew:",type(w))

            boolean_test_threshold = self._get_boolean_test_threshold(
                x=temp_x, y=new_y, w=w, beta=self.beta)
            left_array, right_array, left_list, right_list = self._split_train_data_by_value(
                x=temp_x, w=w, split_value=boolean_test_threshold)
            # print("typeboolean_test_threshold:", type(boolean_test_threshold))
            #qini=qk
            gini = gini_p(new_y) - gini_c(y[left_list], y[right_list])
            # print("typegini:",type(gini))
            # 权重最小的{len(feas)*delete_rate}个属性
            delete_number = math.ceil(len(feas) * delete_rate)
            delete_index_list = all_features_weight_index_list[-delete_number:]
            new_x = np.delete(new_x, delete_index_list, 1)
            max_feas = list()
            for index in features_weight_index_list:
                max_feas.append(feas[index])
            feas = [feas[i]
                    for i in range(len(feas)) if (i not in delete_index_list)]
            # print("typemax_feas:", type(max_feas))
            #print("save_w:", save_w)


            # if gini > max_gini:
            #     max_gini = gini
            #     max_gini_threshold = boolean_test_threshold
            #     max_gini_w = w
            #     max_gini_feas = max_feas


            # 20210312


            point_current_e = e_current /(11-(math.ceil(10/math.sqrt(d_const))))
            print("point_current_e:",point_current_e)

            #print('test:',find_current_depth)

            list_w = np.array(w).ravel().tolist()
            save_w.append(list_w)


            save_boolean_test_threshold.append(boolean_test_threshold)




            save_max_feas.append(max_feas)

            # print(type(max_feas))

            #print('type',type(gini))
            save_gini.append(gini)
            if len(feas) < number :
                break
        # print("len(save_gini):",len(save_gini))
        j= self._expmechanism(save_gini, len(save_gini), point_current_e, 1)
        max_gini_threshold=save_boolean_test_threshold[j]
        max_gini_w=np.array(save_w[j])
        max_gini_feas=save_max_feas[j]
        # print("save_w",save_w)
        # print("savebool:", save_boolean_test_threshold)
        # print("savefeas:", save_max_feas)
        # print('save_gini:', save_gini)
        print('j:', j)
        # print('others:',save_gini, len(save_gini), point_current_e, 1)
        # print('JJJJJJ:',save_w[j])
        # print("typew:",type(save_w[j]))
        # print('save_boolean_test_threshold[j]:', save_boolean_test_threshold[j])
        # print("typeboolean_test_threshold:", type(save_boolean_test_threshold[j]))
        # print('save_max_feas',save_max_feas[j])
        # print("typemax_feas:", type(save_max_feas[j]))
        #print('savemax_feas', save_max_feas)



        #print("these three :",max_gini_threshold, max_gini_w, max_gini_feas)
        #print('bool:',boolean_test_threshold)
        return max_gini_threshold, max_gini_w, max_gini_feas

    def _expmechanism(self,score, m, epsilon, sensitivity):
        exponents_list = []

        sum = 0

        sum_exp = 0

        for i in range(m):
            expo = 0.5 * (score[i]) * epsilon / sensitivity
            exponents_list.append(math.exp(expo))

        for i in range(m):
            sum = sum + exponents_list[i];
            # print('sum:', sum)

        for i in range(m):
            exponents_list[i] = exponents_list[i] / sum
        r = random.uniform(0, 1)
        print('r:', r)
        for j in range(m):

            sum_exp = sum_exp + exponents_list[j];
            if (sum_exp > r):
                break

        return j

    def _top_features_by_number(self, x, y, feas, number):
        """
        :description: 根据权重选取前{number}个特征
        :param: x: numpy.ndarray: 训练数据集 
        :param: y: numpy.ndarray: 类别
        :param: feas: 特征名
        :param: number: 取前{number}个特征
        :return: features_weight_index_list: list: 权重排前{number}个feature索引
        :return: features_weight_items_list: list: 权重排前{number}个feature(索引, 权重值)
        :return: all_features_weight_index_list: list: 权重排序特征索引
        """
        features_num = x.shape[1]
        features_index_list = list(range(features_num))
        features_weight_dict = dict()

        for feature_index in features_index_list:
            features_weight_dict[feature_index] = self._get_feature_weight(
                x, y,  feature_index, feas)
        # 降序，前{number}个值
        all_features_weight_items_list = sorted(
            features_weight_dict.items(), key=lambda item: item[1], reverse=True)
        features_weight_items_list = all_features_weight_items_list[:number]
        features_weight_index_list = list(
            map(lambda x: x[0], features_weight_items_list))
        all_features_weight_index_list = list(
            map(lambda x: x[0], all_features_weight_items_list))
        return features_weight_index_list, features_weight_items_list, all_features_weight_index_list

    def _get_feature_weight(self, x, y, feature_index, feas) -> float:
        '''
        :description: 获取特征权重 
        :param: x: numpy.ndarray: 数据集
        :param: y: numpy.ndarray: 类别
        :param: feature_index: int: feature index
        :param: feas: list for string: 所有特征名称
        :return: weight: float
        '''
        weight = cal_symmetrical_uncertainty(
            feature=x[:, feature_index], y=y) + self._get_part_weight(x, y, feature_index, feas)

        return weight

    def _get_part_weight(self, x, y, feature_index, feas) -> float:
        '''
        :description: 获取特征权重 
        :param x: numpy.ndarray: 数据集
        :param y: numpy.ndarray: 类别
        :param feature_index: int: feature index
        :param feas: list for string: 所有特征名称
        :return: float
        '''
        N_dict = dict()
        N_dict["N"] = 0
        result = 0.0
        for fea in range(len(feas)):
            if fea == feature_index:
                continue
            result += (cal_symmetrical_uncertainty(feature=x[:, feature_index], y=y)/(cal_symmetrical_uncertainty(feature=x[:, feature_index], y=y) +
                                                                                      cal_symmetrical_uncertainty(feature=x[:, fea], y=y))) * self._pair_score(fea1=x[:, feature_index], fea2=x[:, fea], y=y, N_dict=N_dict)
        if N_dict.get("N") == 0:
            return result
        return result / N_dict.get("N")

    def _pair_score(self, fea1, fea2, y, N_dict):
        '''
        :description: 获取特征权重 PairScore分段函数部分
        :param:  fea1: numpy.ndarray: 特征_i
        :param:  fea2: numpy.ndarray: 特征_j
        :param:  y: numpy.ndarray: 类别
        :param:  N_dict: dict: 计算计数
        :return: 
        '''
        cal_ig_value = cal_ig(fea1=fea1, fea2=fea2, y=y)
        if cal_ig_value > 0:
            N_dict["N"] += 1
            return cal_nig(cal_ig_value, fea1=fea1, fea2=fea2)
        else:
            return 0.0

    def _get_boolean_test_threshold(self, x, y, w, beta=0.9) -> float:
        """
        :description: 获取划分阈值
        :param:  x: numpy.ndarray: 训练集
        :param:  y: numpy.ndarray: 目标值
        :param:  w: numpy.ndarray: 权重矩阵
        :param:  beta: float: 取值阈值
        :return: result: float
        """
        k_max = 0  # 最大的k
        K = len(set(y))  # 类别数
        labels = list(set(y))
        i_min = math.ceil(K/2) - math.floor(K/4)
        i_max = math.ceil(K/2) + math.floor(K/4)

        values_dict = self._get_mean_min_max_dict(x, y, w)

        if i_min == i_max:  # 两类相同时
            k_max = 1
        else:              # 两类不同时
            max_res = 0
            feature_index_list = list(range(i_min, i_max + 1))
            for i in feature_index_list:
                res = beta * (values_dict["mean"][labels[i]]-values_dict["mean"][labels[i-1]]) + (
                    1-beta) * (values_dict["min"][labels[i]]-values_dict["max"][labels[i-1]])
                if res > max_res:
                    k_max = i
        result = (values_dict["min"][labels[k_max]] +
                  values_dict["max"][labels[k_max-1]]) / 2
        return result

    def _get_mean_min_max_dict(self, x, y, w):
        """
        :description: 获取降维后各分类的mean、min、max值组成的dict
        :param:  x: numpy.ndarray: 训练集
        :param:  y: numpy.ndarray: 目标值
        :param:  w: numpy.ndarray: 权重矩阵
        :return: values_dict: dict: eg:分类数目为3时, 结果可能为{'mean': {0: -1.291123624302806, 1: 1.2270701499335643, 2: 2.2697816433956293}, 'min': {0: -1.8681557139603338, 1: 0.5710570896124145, 2: 1.7604990823058344}, 'max': {0: -0.8003757459273807, 1: 1.9400383808484012, 2: 3.220663701450751}
        """
        X_new = np.dot((x), w)
        labels = y.reshape(-1, 1)
        data_df = pd.DataFrame(X_new, columns=["data"])
        data_df["label"] = pd.DataFrame(labels)
        try:
            data_df["data"] = data_df["data"].astype("float64")
        except:
            pass
        groupby_data = data_df.groupby(["label"])
        count_df = pd.concat(
            [groupby_data.mean(), groupby_data.min(), groupby_data.max()], axis=1)
        count_df.columns = ["mean", "min", "max"]

        return count_df.to_dict()

    def _split_train_data_by_value(self, x, w, split_value, index_list=[]):
        """
        :description: 按{split_value}将训练集{x}分为并返回left_array, right_array
        :param:  x: numpy.ndarray: 训练集
        :param:  w: numpy.ndarray: 权重矩阵
        :param:  split_value: float: 取值阈值
        :return: left_array, right_array, left_list, right_list
        """
        if index_list != []:
            cal_x = x[:, index_list]
        else:
            cal_x = x
        sample_length = len(x)
        left_list = list()
        right_list = list()
        left_array = [list(x[0])]  # 占位定型，后面需要删除
        right_array = [list(x[0])]
        for index, sample, cal_sample in zip(range(sample_length), x, cal_x):
            if np.dot(w.T, cal_sample) < split_value:

                left_list.append(index)
                left_array = np.append(left_array, [list(sample)], axis=0)
            else:
                right_list.append(index)
                right_array = np.append(right_array, [list(sample)], axis=0)
        left_array = np.delete(left_array, 0, axis=0)  # 删除首行
        right_array = np.delete(right_array, 0, axis=0)
        return left_array, right_array, left_list, right_list

    def add_noisyCount(self,sensitivety, epsilon):
        beta = sensitivety / epsilon
        u1 = np.random.random()
        u2 = np.random.random()
        if u1 <= 0.5:
            n_value = -beta * np.log(1. - u2)
        else:
            n_value = beta * np.log(u2)
        return n_value

    def _build_tree(self, x, y, current_depth=1, key=None):
        '''
        :description: 构建决策树方法 
        :param: x: numpy.ndarray: 训练数据集 
        :param: y: numpy.ndarray: 训练数据集标签
        :param:  current_depth: 目前的层数
        :return: 
        '''


        feas = list(range(x.shape[1]))
        labels = y

        # return 1: same label(所有样本都属于同一类别)
        if len(set(labels)) == 1:
            return labels[0]

        # max_label 保存labels中出现最多次数的label
        s = 0
        under_d = self.max_depth

        while (under_d >= 1):
            s += (1 / under_d)
            under_d -= 1
        # print('s:',s)

        # dangqiancengshudeyinsiyusuan
        e_current = ((self.privacy_e) / s) * (1 / (self.max_depth - current_depth + 1))
        print('e_curent:', e_current)

        max_label = max([(i, len(list(filter(lambda tmp: tmp == i, labels)))+self.add_noisyCount(1,e_current))
                         for i in set(labels)], key=lambda tmp: tmp[1])[0]
        # print('labels', labels)
        # print("fset(labels):",set(labels))
        # print("max([(i, len(list(filter(lambda tmp: tmp == i, labels))))for i in set(labels)], key=lambda tmp: tmp[1])", max([(i, len(list(filter(lambda tmp: tmp == i, labels))))
        #                  for i in set(labels)], key=lambda tmp: tmp[1]))

        # return 2: 层数大于最大层数
        if current_depth >= self.max_depth:
            return max_label

        # return 3: 当前节点所包含的样本中，绝大多数主语同一类的话，就不继续划分了。比如当前节点有100个样本，其中属于第一类的有98个，属于第二类的有2个，停止划分，并将该结点标记为叶子结点，类别为第一类
        if get_label_max_proportion(y) > self.sample_rate:
            return max_label

        current_depth += 1

        #print('test_current:',current_depth)
        number = math.ceil(math.sqrt(len(feas)))

        # 对称不确定性为0的特征(test)
        delete_list = list()
        for col in range(len(feas)):
            if cal_symmetrical_uncertainty(x[:, col], y) == 0.0:
                delete_list.append(col)
        if len(delete_list) == x.shape[1]:
            return max_label

        max_gini_threshold, max_gini_w, max_gini_feas = self._find_best_split(
            x, y, feas, number=number, delete_rate=None, e_current=e_current)
        left_array, right_array, left_list, right_list = self._split_train_data_by_value(
            x=x, w=max_gini_w, split_value=max_gini_threshold, index_list=max_gini_feas)
        left_y = y[left_list]
        right_y = y[right_list]
        tree = dict()
        value_dict = dict()
        value_dict["threshold"] = max_gini_threshold
        value_dict["indexs"] = max_gini_feas
        value_dict["w"] = max_gini_w
        tree["values"] = value_dict
        # 递归左子树
        #

        if left_list == [] or right_list == []:
            return max_label

        if left_list != []:
            left_tree = self._build_tree(
                x=left_array, y=left_y, current_depth=current_depth, key="left_"+str(current_depth))
            # left_label = "left_tree|{0}|{1}|{2}".format(str(max_gini_threshold)," ".join(map(str, max_gini_feas)),max_gini_w.tolist())
            tree["left_tree"] = left_tree

        # 递归右子树
        #
        if right_list != []:
            right_tree = self._build_tree(
                x=right_array, y=right_y, current_depth=current_depth, key="right_"+str(current_depth))
            # right_label = "right_tree|{0}|{1}|{2}".format(str(max_gini_threshold)," ".join(map(str, max_gini_feas)),max_gini_w.tolist())
            tree["right_tree"] = right_tree

        return tree





def gini_p(y, total_y_number=None) -> float:
    """
    :description: 计算基尼指数
    :param y: numpy.ndarray: 标签
    :param total_y_number: int: 父树标签数量
    :return: gini(y): float: 基尼指数
    """
    if total_y_number is None:
        number = len(y)
    else:
        number = total_y_number
    counter = Counter(y)
    res = 1.0
    for num in counter.values():
        p = num / number
        res -= p**2
    return res


def gini_c(left_y, right_y) -> float:
    """
    :description: 计算划分左右子树后基尼指数
    :param left_y : numpy.ndarray: 左子树标签
    :param right_y: numpy.ndarray: 右子树标签
    :return: gini(y): float: 基尼指数
    """
    left_y_number, right_y_number = len(left_y), len(right_y)
    total_y_number = left_y_number + right_y_number
    res = (left_y_number/total_y_number) * gini_p(left_y, left_y_number) + \
        (right_y_number/total_y_number) * gini_p(right_y, right_y_number)
    return res


@jit
def cal_mutual_information(feature, y):
    '''
    :description: 计算互信息
    :param: feature: 特征
    :param: y: 类别
    :return: mi(feature, y)
    '''
    ent = cal_ent(y) - cal_condition_ent(feature, y)
    # ent = cal_ent(feature) + cal_ent(y) - cal_joint_entropy_2(feature, y)
    return ent


@jit
def cal_symmetrical_uncertainty(feature, y):
    '''
    :description: 计算对称不确定性 
    :param: feature: 特征
    :param: y: 类别
    :return: SU(feature, y)
    '''
    su = (2*cal_mutual_information(feature, y)) / \
        (cal_ent(feature) + cal_ent(y))
    return su


@jit
def cal_ent(x) -> float:
    '''
    :description: 计算香农信息熵
    :param: x: 类别名称值: pandas.core.series.Series
    :return: ent: H(D)=-\sum_{k=1}^K\frac{|C_k|}{|D|}\log_2\frac{|C_k|}{|D|}
    '''
    x_values = list(set(x))
    ent = 0
    for x_value in x_values:
        p = x[x == x_value].shape[0]/x.shape[0]
        ent -= p*np.log2(p)
    return ent


@jit
def cal_condition_ent(x, y):
    '''
    :description: 计算条件熵ent(y|x)
    :param: x: feature: pandas.core.series.Series
    :param: y: class: pandas.core.series.Series
    :return: ent(y|x)
    '''
    ent = 0
    x_values = set(x)
    for x_value in x_values:
        sub_y = y[x == x_value]
        tmp_ent = cal_ent(sub_y)
        p = sub_y.shape[0]/y.shape[0]
        ent += p*tmp_ent
    return ent


def gain(x, y):
    '''
    :description: 计算信息增益
    :param: x: feature: pandas.core.series.Series
    :param: y: class: pandas.core.series.Series
    :return: gain
    '''
    return cal_ent(y) - cal_condition_ent(x, y)


def gain_ratio(x, y):
    '''
    :description: 计算信息增益率gr(y, x)
    :param: x: feature: pandas.core.series.Series
    :param: y: class: pandas.core.series.Series
    :return: gr(y, x)
    '''
    return gain(x, y)/cal_ent(x)


@jit
def cal_joint_entropy_3(fea1, fea2, y) -> float:
    '''
    :description: 计算联合熵，通过特征，三个输入
    :param: fea: 特征 numpy.ndarray
    :param: y: 分类 numpy.ndarray
    :return: joint_entropy
    '''
    result = 0.0
    # 属性以及class不同的值
    fea1_values, fea2_values, CLASS_values = list(
        set(fea1)), list(set(fea2)), list(set(y))
    fea1, fea2, y = pd.DataFrame(
        fea1.reshape(-1, 1)), pd.DataFrame(fea2.reshape(-1, 1)), pd.DataFrame(y.reshape(-1, 1))
    df = pd.concat([fea1, fea2, y], axis=1)
    df.columns = list(range(3))
    for fea1_item in fea1_values:
        for fea2_item in fea2_values:
            for class_item in CLASS_values:
                temp_df = df[(df[0] == fea1_item) & (
                    df[1] == fea2_item) & (df[2] == class_item)]
                if temp_df.empty:
                    continue
                p = temp_df.shape[0] / df.shape[0]
                result -= p*np.log2(p)
    return result


@jit
def cal_joint_entropy_2(fea1, fea2) -> float:
    '''
    :description: 计算联合熵，通过特征，两个输入, H(Y|X)+H(X) = H(X,Y)
    :param: fea: 特征 numpy.ndarray
    :return: joint_entropy
    '''
    result = 0.0
    # 属性不同的值
    fea1_values, fea2_values = list(set(fea1)), list(set(fea2))
    fea1 = pd.DataFrame(fea1.reshape(-1, 1))
    fea2 = pd.DataFrame(fea2.reshape(-1, 1))
    df = pd.concat([fea1, fea2], axis=1)
    df.columns = list(range(2))
    for fea1_item in fea1_values:
        for fea2_item in fea2_values:
            temp_df = df[(df[0] == fea1_item) & (df[1] == fea2_item)]
            if temp_df.empty:
                continue
            p = temp_df.shape[0] / df.shape[0]
            result -= p*np.log2(p)
    return result

# @jit
# def cal_joint_entropy_2(fea1, fea2, z=None):
#     """
#     计算2-3个离散变量的联合信息熵
#     Parameters
#     ----------
#     df: pandas.DataFrame
#     x: str
#         变量名1
#     y: str
#         变量名2
#     z: str, default None
#         变量名3，此变量只允许服从0-1分布的变量
#     Returns
#     -------
#     float
#     """
#     fea1, fea2 = pd.DataFrame(
#         fea1.reshape(-1, 1)), pd.DataFrame(fea2.reshape(-1, 1))
#     df = pd.concat([fea1, fea2], axis=1)
#     df.columns = list(range(df.shape[1]))
#     x = 0
#     y = 1
#     if z is not None:
#         if set(df[z].unique()) != {0, 1}:
#             raise ValueError('variable z be allowed values 0 and 1')
#         xy_joint_freq = pd.crosstab(index=df[x], columns=df[y]).astype(float)
#         count_z1 = pd.crosstab(
#             index=df[x], columns=df[y], values=df[z], aggfunc='sum').fillna(0)
#         count_z0 = xy_joint_freq - count_z1
#         joint_pdf = np.array(
#             [count_z1.values, count_z0.values], dtype=float) / df.shape[0]
#     else:
#         joint_pdf = pd.crosstab(df[x], df[y], normalize=True).values

#     joint_pdf = np.where(joint_pdf == 0, 1e-6, joint_pdf)     # laplace平滑

#     return -(joint_pdf * np.log2(joint_pdf)).sum()

# @jit


def cal_ig(fea1, fea2, y):
    '''
    :description: 计算信息交互增益
    :param: fea: 特征 numpy.ndarray
    :param: y: 分类 numpy.ndarray
    :return: ig(fea1, fea2, y)
    '''
    ig_result = 2*(cal_joint_entropy_2(fea1, fea2) + cal_joint_entropy_2(fea1, y) + cal_joint_entropy_2(
        fea2, y)) - 3*(cal_ent(fea1) + cal_ent(fea2) + cal_ent(y)) - cal_joint_entropy_3(fea1, fea2, y)
    return ig_result


def cal_nig(cal_ig_value, fea1, fea2):
    '''
    :description: 计算信息交互增益(归一化)
    :param: fea: 特征 numpy.ndarray
    :param: y: 分类 numpy.ndarray
    :return: nig(fea1, fea2, y)
    '''
    nig_result = 0.5 + (cal_ig_value /
                        (2*(cal_ent(fea1)+cal_ent(fea2))))
    return nig_result


def get_label_max_proportion(y) -> float:
    """
    :description: 获取占比最大标签的比例
    :param y: numpy.ndarray: 标签集
    :return: result: float: 占比最大标签的比例
    """
    number = len(y)
    counter = max(Counter(y).values())
    result = counter / number
    return result


def LDA_dimensionality(x, y, k=1):
    """
    :description: 使用LDA降维，获取权重矩阵
    :param X: numpy.ndarray: 训练集
    :param y: numpy.ndarray: 标签
    :param k: numpy.ndarray: 降维后维数, default=1
    :return: w: numpy.ndarray: 权重矩阵
    """
    # print("x,", x)
    # print("y,", y)
    label_ = list(set(y))  # 统计标签类型
    X_classify = {}
    # 将数据集归类
    for label in label_:
        X1 = np.array([x[i] for i in range(len(x)) if y[i] == label])
        X_classify[label] = X1
    mju = np.mean(x, axis=0)  # 计算所有属性列的均值
    # 计算不同类的属性列均值
    mju_classify = {}
    for label in label_:
        mju1 = np.mean(X_classify[label], axis=0)
        mju_classify[label] = mju1
    #St = np.dot((x - mju).T, x - mju)
    Sw = np.zeros((len(mju), len(mju)))  # 计算类内散度矩阵
    for i in label_:
        # Sw += np.dot((X_classify[i] - mju_classify[i]).T,
        #              X_classify[i] - mju_classify[i])
        Sw = np.add(np.dot((X_classify[i] - mju_classify[i]).T, X_classify[i] - mju_classify[i]), Sw, casting="unsafe")

    # Sb=St-Sw
    Sb = np.zeros((len(mju), len(mju)))  # 计算类内散度矩阵
    for i in label_:
        # Sb += len(X_classify[i]) * np.dot((mju_classify[i] - mju).reshape(
        #     (len(mju), 1)), (mju_classify[i] - mju).reshape((1, len(mju))))
        Sb = np.add(len(X_classify[i]) * np.dot((mju_classify[i] - mju).reshape(
            (len(mju), 1)), (mju_classify[i] - mju).reshape((1, len(mju)))), Sb)
    if np.linalg.det(Sw) == 0.0:
        eig_vals, eig_vecs = np.linalg.eig(
            np.linalg.pinv(Sw).dot(Sb))  # 计算Sw-1*Sb的特征值和特征矩阵(伪逆矩阵)
    else:
        eig_vals, eig_vecs = np.linalg.eig(
            np.linalg.inv(Sw).dot(Sb))  # 计算Sw-1*Sb的特征值和特征矩阵
    sorted_indices = np.argsort(eig_vals)
    w = eig_vecs[:, sorted_indices[:-k - 1:-1]]  # 提取前k个特征向量
    return w


def test_iris():
    """
    非离散数据不适用
    """
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import LabelEncoder
    from sklearn.utils import shuffle

    iris = load_iris()
    x = iris.data
    y = iris.target
    # 读取数据集
    # cols = iris.feature #特征名称转list
    x = shuffle(x)
    feas = iris.feature_names
    labels = y

    # df = df
    print("RUN...")

    # 划分训练集和测试集
    top_number = 120
    x_train = x[:top_number]
    y_train = y[:top_number]  # 好瓜
    x_test = x[top_number:]
    y_test = y[top_number:]  # 好瓜

    print("x_train.shape", x_train.shape)
    print("y_train.shape", y_train.shape)
    print("x_test.shape", x_test.shape)
    print("y_test.shape", y_test.shape)

    tree = Tree(feas=feas, delete_rate=0.1, eps=0.02,
                sample_rate=0.95, max_depth=10, name=None, criterion="entropy")
    tree.fit(x_train, y_train)
    predict_y = tree.predict(x_test)
    print("predict_y\n{0}\n".format(predict_y))
    print("true_y\n{0}\n".format(y_test))

    print("score:{0}\n".format(tree.get_score(predict_y, y_test)))
    print("tree\n{0}\n".format(tree.tree_))


def test_processed_hungarian():

    from sklearn.model_selection import train_test_split
    from tool.mdlp import MDLPDiscretizer
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
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, test_size=0.2, random_state=1)
    x_train, x_test, y_train, y_test = x_train.values, x_test.values, y_train.values, y_test.values
    # estimator = tree.DecisionTreeClassifier() #　# 基分类器：sklearn 单变量决策树

    estimator = Tree(cols)  # 基分类器：多变量决策树
    
    tree = Tree(feas=cols, delete_rate=0.1, eps=0.02,
                sample_rate=0.95, max_depth=10, name=None, criterion="entropy")
    tree.fit(x_train, y_train)
    predict_y = tree.predict(x_test)
    print("tree\n{0}\n".format(tree.tree_))

    print("predict_y\n{0}\n".format(predict_y))
    print("true_y\n{0}\n".format(y_test))

    print("score:{0}\n".format(tree.get_score(predict_y, y_test)))



if __name__ == '__main__':
    start = time()
    # test_iris()
    test_processed_hungarian()
    print("TIME", time()-start)
