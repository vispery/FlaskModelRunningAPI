import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
import sys, os

from data_analysis_util_api.utils import rules_function_def as rfd
#sys.path.append(os.path.abspath(os.path.join(sys.path[0], '..')))

# 规则筛选模块
def model_rule(dataset, cateValue, deleteValue, lift_down_lmt, badrate_down_lmt, Rule_min_cnt, lift_need, min_bad_add,
               nan_ratio_threshold=0.999, mode_ratio_threshold=0.999, nan_replace_num=-999, train_size=0.7,
               max_leaf_num=6,
               min_woe_box_percent=0.01, min_woe_box_num_min=50, cor_max=0.9):
    """
        定义规则模型处理函数
        输入：
        dataset: 数据集文件存放位置
        cateValue: 标签
        deleteValue：需要删除的无关列
        lift_down_lmt：选取用于两两交叉的规则的提升度阈值
        badrate_down_lmt：坏账率阈值
        Rule_min_cnt：规则需命中的最小样本量
        lift_need：规则需要满足的提升度
        min_bad_add：规则集累加参数
        nan_ratio_threshold：剔除缺失值的阈值，默认为0.999
        mode_ratio_threshold：剔除单一值的阈值，默认为0.999
        nan_replace_num：NAN填充值，默认为-999
        train_size：训练集比例，默认为0.7
        max_leaf_num：分箱的最大箱数，默认值为6
        min_woe_box_percent：叶节点最小样本量比例（仅占非空值的），默认值为0.01
        min_woe_box_num_min：叶节点最小样本量；满足叶节点最小样本量比例，且满足叶节点最小样本量，默认值为50
        cor_max：相关性阈值，默认值为0.9

        输出：dict
        rule_final_table: string 最终规则集csv文件地址
        rule_final_effe_table：string 规则集在测试集上的效果文件地址
    """
    # 返回结果
    rt = {}
    # 定义本次任务的一个标志，暂时用时间来代表
    job_flag = time.strftime("%Y%m%d%H%M%S", time.localtime())

    # 1.处理数据
    df = pd.read_csv(dataset, sep=',', encoding='utf-8')
    # 删除与建模无关的列
    df.drop(columns=deleteValue, inplace=True)
    # 缺失值与重复值处理
    df = rfd.pre_deal(df, nan_ratio_threshold=nan_ratio_threshold, mode_ratio_threshold=mode_ratio_threshold)
    # nan值填充
    df.replace(nan_replace_num, np.nan, inplace=True)
    print(df.head(5))
    # 数据集划分
    Y = df[cateValue]
    df_train, df_test = train_test_split(df, test_size=1 - train_size, random_state=888, stratify=Y)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)

    Data = df.copy()
    X = Data.drop(columns=[cateValue])
    Y = Data.loc[:, cateValue]  # 注意，此处1代表坏客户，0代表好客户
    badrate_real = sum(Y) / len(Y)  # np.shape(Y)[0] returns the number of rows

    # woe分箱
    var_splitpoint = rfd.myWOEbin(X, Y, max_leaf_num=max_leaf_num,
                                  min_woe_box_percent=min_woe_box_percent,
                                  min_woe_box_num_min=min_woe_box_num_min)

    # 映射单变量规则
    left_new = list(set(var_splitpoint.index.values))
    right_new = list(set(var_splitpoint.index.values))
    oth_var = set(var_splitpoint.index.values) - set(left_new)
    oth_var = list(oth_var - set(right_new))

    # 处理分箱结果
    rule_X, rule = rfd.deal_WOEbin(X, Y, left_var=left_new,
                                   right_var=right_new, oth_var=oth_var,
                                   var_splitpoint=var_splitpoint,
                                   lift_need=lift_down_lmt,
                                   badrate_need=badrate_down_lmt)
    print('用于两两交叉的规则个数为{0}'.format(len(rule)))

    # 处理相关性
    rule_cor = pd.DataFrame(np.corrcoef(rule_X.T))  # 计算相关系数矩阵
    rule_cor_choose, rule_X_cor_choose = rfd.dealwithcor(rule_cor, rule,
                                                         rule_X, cor_max=cor_max)

    if_cross, hitnum, badnum, badrate = rfd.rule_cross_rule(rule_cor_choose, rule_X_cor_choose, Y,
                                                            Rule_min_cnt=Rule_min_cnt,
                                                            lift_need=lift_need,
                                                            badrate_real=badrate_real)

    # 提取单变量规则
    badrate_need = badrate_real * lift_need

    # 取出满足条件的单变量规则
    OneVar_choose = np.where((rule.loc[:, "badrate"] > badrate_need) & (rule.loc[:, "hit_num"] > Rule_min_cnt))[0]
    OneVarRule = pd.DataFrame(rule.iloc[OneVar_choose, :])
    OneVarRule_X = pd.DataFrame(rule_X.iloc[:, OneVar_choose])
    OneVarRule_num = len(OneVar_choose)

    # 双变量规则的条数
    TwoVarRule_num = 0
    for i in range(len(if_cross)):
        TwoVarRule_num = sum(if_cross.iloc[:, i]) + TwoVarRule_num
    TwoVarRule_num = int(TwoVarRule_num)

    # 用于存储单、双变量规则
    TwoVarRule = pd.DataFrame(np.zeros([TwoVarRule_num + OneVarRule_num, 9]) * np.nan,
                              columns=["var_1", "down_lmt_1", "up_lmt_1",
                                       "var_2", "down_lmt_2", "up_lmt_2", "badnum", "hitnum", "badrate"])

    sample_num = len(rule_X_cor_choose)  # 总样本量
    TwoVarRule_X = pd.DataFrame(np.zeros([sample_num, (TwoVarRule_num + OneVarRule_num)]))

    # 提取双变量规则
    k = 0
    for i in range(len(rule_cor_choose)):
        temp_i = rule_X_cor_choose.iloc[:, i]
        temp_index = np.where(if_cross.iloc[i, :] > 0)[0]
        temp_num = len(temp_index)
        print(temp_num)
        if temp_num > 0:
            temp_cors_i = pd.DataFrame(np.zeros([sample_num, temp_num]))
            for j in range(temp_num):
                temp = rule_X_cor_choose.iloc[:, temp_index[j]] + temp_i
                temp_cors_i.iloc[np.where(temp == 2)[0], j] = 1
            TwoVarRule_X.iloc[:, k:(k + temp_num)] = np.array(temp_cors_i)
            TwoVarRule.iloc[k:(k + temp_num), 0] = np.array(rule_cor_choose.iloc[[i], 0])
            TwoVarRule.iloc[k:(k + temp_num), 1] = np.array(rule_cor_choose.iloc[[i], 1])
            TwoVarRule.iloc[k:(k + temp_num), 2] = np.array(rule_cor_choose.iloc[[i], 2])
            TwoVarRule.iloc[k:(k + temp_num), 3] = np.array(rule_cor_choose.iloc[temp_index, 0])
            TwoVarRule.iloc[k:(k + temp_num), 4] = np.array(rule_cor_choose.iloc[temp_index, 1])
            TwoVarRule.iloc[k:(k + temp_num), 5] = np.array(rule_cor_choose.iloc[temp_index, 2])
            TwoVarRule.iloc[k:(k + temp_num), 6] = np.array(badnum.iloc[i, temp_index])
            TwoVarRule.iloc[k:(k + temp_num), 7] = np.array(hitnum.iloc[i, temp_index])
            TwoVarRule.iloc[k:(k + temp_num), 8] = np.array(badrate.iloc[i, temp_index])
            k = k + len(temp_index)

    # 单变量规则
    TwoVarRule.iloc[TwoVarRule_num:, 0] = np.array(rule.iloc[OneVar_choose, 0])
    TwoVarRule.iloc[TwoVarRule_num:, 1] = np.array(rule.iloc[OneVar_choose, 1])
    TwoVarRule.iloc[TwoVarRule_num:, 2] = np.array(rule.iloc[OneVar_choose, 2])
    TwoVarRule.iloc[TwoVarRule_num:, 6] = np.array(rule.iloc[OneVar_choose, 3])
    TwoVarRule.iloc[TwoVarRule_num:, 7] = np.array(rule.iloc[OneVar_choose, 4])
    TwoVarRule.iloc[TwoVarRule_num:, 8] = np.array(rule.iloc[OneVar_choose, 5])
    TwoVarRule_X.iloc[:, TwoVarRule_num:] = OneVarRule_X.values

    # 筛选有效规则集
    if_choose_final = rfd.useful_rule_v1(TwoVarRule, TwoVarRule_X, Y,
                                         min_bad_add=min_bad_add,
                                         badrate="badrate")

    rule_final = TwoVarRule.iloc[np.where(if_choose_final.iloc[:, 0] == 1)[0], :]
    rule_X_final = TwoVarRule_X.iloc[:, np.where(if_choose_final.iloc[:, 0] == 1)[0]]

    # 保存rule_final
    rule_final_table_path = 'data_analysis_util_api/output/rule/csv/rule_final_table' + job_flag + '.csv'
    rule_final.to_csv(rule_final_table_path)
    rt['rule_final_table'] = rule_final_table_path

    # 验证规则集的效果
    all_bad = sum(Y)
    sample_num = len(Y)

    order_temp = np.argsort(-np.array(rule_final.loc[:, "badrate"]))

    rule_final_effe = pd.DataFrame(np.zeros([len(rule_final), 8]),
                                   columns=["badnum_acu", "hit_num_acu", "badrate_acu",
                                            "recall_acu", "coverate_acu", "lift_acu", "var_ch_1", "var_ch_2"])

    temp = pd.DataFrame(np.zeros([len(rule_X_final), 1]))
    for i in range(len(order_temp)):
        temp_i = pd.DataFrame(rule_X_final.iloc[:, order_temp[i]])
        temp = pd.DataFrame(temp_i.iloc[:, 0] + temp.iloc[:, 0])
        temp.iloc[np.where(temp.iloc[:, 0] > 0)[0], :] = 1

        rule_final_effe.iloc[i, 0] = np.dot(np.array(temp.iloc[:, 0]), np.array(Y))
        rule_final_effe.iloc[i, 1] = sum(temp.iloc[:, 0])
        rule_final_effe.iloc[i, 2] = rule_final_effe.iloc[i, 0] / rule_final_effe.iloc[i, 1]
        rule_final_effe.iloc[i, 3] = rule_final_effe.iloc[i, 0] / all_bad
        rule_final_effe.iloc[i, 4] = rule_final_effe.iloc[i, 1] / sample_num
        rule_final_effe.iloc[i, 5] = rule_final_effe.iloc[i, 2] / badrate_real

    rule_final_temp = pd.DataFrame(rule_final.iloc[order_temp, :])
    rule_final_temp.index = [i for i in range(len(rule_final))]

    rule_result = pd.concat([rule_final_temp, rule_final_effe], axis=1)
    rule_result = rule_result.loc[:, ['var_1', 'down_lmt_1', 'up_lmt_1', 'var_2', 'down_lmt_2', 'up_lmt_2',
                                      'badnum', 'hitnum', 'badrate', 'badnum_acu', 'hit_num_acu',
                                      'badrate_acu', 'recall_acu', 'coverate_acu', 'lift_acu']]

    # 保存rule_result
    rule_result_table_path = 'data_analysis_util_api/output/rule/csv/rule_result_table' + job_flag + '.csv'
    rule_result.to_csv(rule_result_table_path)
    rt['rule_result_table'] = rule_result_table_path

    return rt
