# -*- coding: utf-8 -*-
"""
Created on Sat May 30 17:57:50 2020

@author: juan.li
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 30 15:52:49 2020

@author: juan.li
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

def psi_stats_score(data_left, data_right, non_computed=None, plot_image=True):
    """Calculate Average PSI value.
    Parameters
    ----------
    data_left: pandas.DataFrame
        The necessary columns score, y, id must be in data.

    data_right: pandas.DataFrame
        The necessary columns score, y, id must be in data.

    non_computed : str or None
        The column name of non-computed scoring indicators.
        'True' means score by non_computed.
        'False' means score by computed.

    plot_image : bool (default True)
        Plot image.

    Returns
    -------
    psi_table : pandas.DataFrame
        The PSI value of score interval.
    """

    """Check columns of data."""
    check_cols = ['score', 'y', 'id']
    if non_computed != None and type(non_computed) == str:
        check_cols += [non_computed]
        data_left = data_left[~data_left[non_computed] == True].copy()
        data_right = data_right[~data_right[non_computed] == True].copy()
    elif non_computed == None:
        pass
    else:
        raise ValueError('non_computed must be a str.')
    for col in check_cols:
        if col not in data_left.columns or col not in data_right.columns:
            raise ValueError('Please check the columns %s of data' % col)

    """Drop NaN by column 'score'."""
    data_left = data_left.loc[data_left['score'].notnull(), check_cols]
    data_right = data_right.loc[data_right['score'].notnull(), check_cols]

    """Discrete score value."""
    break_points = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    data_left.loc[data_left.score < 0, 'score'] = 0
    data_left.loc[data_left.score >= 100, 'score'] = 99
    data_right.loc[data_right.score < 0, 'score'] = 0
    data_right.loc[data_right.score >= 100, 'score'] = 99
    data_left['score'] = pd.cut(data_left['score'], break_points, right=False).values
    data_right['score'] = pd.cut(data_right['score'], break_points, right=False).values

    """Count psi of bad & good sample."""
    count_left = data_left.groupby(['score', 'y']).count()['id'].unstack().fillna(value=0.0)
    count_right = data_right.groupby(['score', 'y']).count()['id'].unstack().fillna(value=0.0)
    count_left['bad_ratio'] = count_left[1] / count_left[1].sum()
    count_right['bad_ratio'] = count_right[1] / count_right[1].sum()
    count_left['good_ratio'] = count_left[0] / count_left[0].sum()
    count_right['good_ratio'] = count_right[0] / count_right[0].sum()
    count_final = pd.merge(count_left, count_right, left_index=True,
                           right_index=True, suffixes=['_left', '_right'])
    count_final['psi_bad'] = (count_left['bad_ratio'] - count_right['bad_ratio']) * \
                             np.log(count_left['bad_ratio'] / count_right['bad_ratio'])
    count_final['psi_good'] = (count_left['good_ratio'] - count_right['good_ratio']) * \
                              np.log(count_left['good_ratio'] / count_right['good_ratio'])

    average_psi = (count_final['psi_bad'].replace([np.inf, np.nan], 0.0).sum() + \
                   count_final['psi_good'].replace([np.inf, np.nan], 0.0).sum()) / 2

    """Plot image"""
    if plot_image == True:
        plot_range = ['bad_ratio_left', 'good_ratio_left',
                      'bad_ratio_right', 'good_ratio_right']
        plot_label = ['Bad Ratio of Test Sample', 'Good Ratio of Test Sample',
                      'Bad Ratio of Train Set', 'Good Ratio of Train Set']
        color = ['blue', 'red', 'green', 'cyan']
        marker = ['s', 'x', 'o', 'v']
        plt.figure()
        for p, l, c, m in zip(plot_range, plot_label, color, marker):
            value = count_final[p].values
            score_range = range(len(count_final.index))
            score_label = ['[0,10)', '[10,20)',
                           '[20,30)', '[30,40)',
                           '[40,50]', '[50,60)',
                           '[60,70)', '[70,80]',
                           '[80,90]', '[90,100)']
            plt.plot(score_range, value, color=c, marker=m,
                     markersize=1, label=l)
        plt.grid()
        plt.legend(loc='upper left')
        plt.xticks(score_range, score_label, rotation=45)
        plt.title('PSI of Score Card')
        plt.ylabel('Ratio')
        plt.tight_layout()
        plt.show()
    print('Average PSI:%f' % average_psi)
    return count_final

def psi_stats_score1(data_left, data_right, non_computed=None, plot_image=True):
    """Calculate Average PSI value.
    Parameters
    ----------
    data_left: pandas.DataFrame
        The necessary columns score, y, id must be in data.

    data_right: pandas.DataFrame
        The necessary columns score, y, id must be in data.

    non_computed : str or None
        The column name of non-computed scoring indicators.
        'True' means score by non_computed.
        'False' means score by computed.

    plot_image : bool (default True)
        Plot image.

    Returns
    -------
    dict
        count_final : pandas.DataFrame
            The PSI value of score interval.
        figure: matplotlib figure objects
            PSI of scorecard.
        average_psi: float
            Average PSI
    """

    """result dict"""
    rt = {}

    """Check columns of data."""
    check_cols = ['score', 'y', 'id']
    if non_computed != None and type(non_computed) == str:
        check_cols += [non_computed]
        data_left = data_left[~data_left[non_computed] == True].copy()
        data_right = data_right[~data_right[non_computed] == True].copy()
    elif non_computed == None:
        pass
    else:
        raise ValueError('non_computed must be a str.')
    for col in check_cols:
        if col not in data_left.columns or col not in data_right.columns:
            raise ValueError('Please check the columns %s of data' % col)

    """Drop NaN by column 'score'."""
    data_left = data_left.loc[data_left['score'].notnull(), check_cols]
    data_right = data_right.loc[data_right['score'].notnull(), check_cols]

    """Discrete score value."""
    break_points = list(range(300,1001,50))
    data_left.loc[data_left.score < 300, 'score'] = 300
    data_left.loc[data_left.score >= 1000, 'score'] = 999
    data_right.loc[data_right.score < 300, 'score'] = 300
    data_right.loc[data_right.score >= 1000, 'score'] = 999
    data_left['score'] = pd.cut(data_left['score'], break_points, right=False).values
    data_right['score'] = pd.cut(data_right['score'], break_points, right=False).values

    """Count psi of bad & good sample."""
    count_left = data_left.groupby(['score', 'y']).count()['id'].unstack().fillna(value=0.0)
    count_right = data_right.groupby(['score', 'y']).count()['id'].unstack().fillna(value=0.0)
    count_left['bad_ratio'] = count_left[1] / count_left[1].sum()
    count_right['bad_ratio'] = count_right[1] / count_right[1].sum()
    count_left['good_ratio'] = count_left[0] / count_left[0].sum()
    count_right['good_ratio'] = count_right[0] / count_right[0].sum()
    count_final = pd.merge(count_left, count_right, left_index=True,
                           right_index=True, suffixes=['_left', '_right'])
    count_final['psi_bad'] = (count_left['bad_ratio'] - count_right['bad_ratio']) * \
                             np.log(count_left['bad_ratio'] / count_right['bad_ratio'])
    count_final['psi_good'] = (count_left['good_ratio'] - count_right['good_ratio']) * \
                              np.log(count_left['good_ratio'] / count_right['good_ratio'])
    rt['count_final'] = count_final

    average_psi = (count_final['psi_bad'].replace([np.inf, np.nan], 0.0).sum() + \
                   count_final['psi_good'].replace([np.inf, np.nan], 0.0).sum()) / 2
    print('Average PSI:%f' % average_psi)
    rt['average_psi'] = average_psi

    """Plot image"""
    if plot_image == True:
        plot_range = ['bad_ratio_left', 'good_ratio_left',
                        'bad_ratio_right', 'good_ratio_right']
        plot_label = ['Bad Ratio of Test Sample', 'Good Ratio of Test Sample',
                        'Bad Ratio of Train Set', 'Good Ratio of Train Set']
        color = ['blue', 'red', 'green', 'cyan']
        marker = ['s', 'x', 'o', 'v']
        figure = plt.figure()
        for p, l, c, m in zip(plot_range, plot_label, color, marker):
            value = count_final[p].values
            score_range = range(len(count_final.index))
            score_label = ['[300,350)', '[350,400)',
                            '[400,450)', '[450,500)',
                            '[500,550]', '[550,600)',
                            '[600,650)', '[650,700]',
                            '[700,750]', '[750,800)',
                            '[800,850]', '[850,900)'
                            '[900,950]', '[950,1000)']
            plt.plot(score_range, value, color=c, marker=m,
                        markersize=1, label=l)
        plt.grid()
        plt.legend(loc='upper left')
        plt.xticks(score_range, score_label, rotation=45)
        plt.title('PSI of Score Card')
        plt.ylabel('Ratio')
        plt.tight_layout()
        plt.show()
        rt['figure'] = figure
    
    '''return'''
    return rt

#类别变量转数值变量
def cate_var_transform(X, Y):
    ##取出数据类型
    d_type = X.dtypes
    object_var = X.iloc[:, np.where(d_type == "object")[0]]
    num_var = X.iloc[:, np.where(d_type != "object")[0]]

    # object_transfer_rule用于记录每个类别变量的数值转换规则
    object_transfer_rule = list(np.zeros([len(object_var.columns)]))

    # object_transform是类别变量数值化转化后的值
    object_transform = pd.DataFrame(np.zeros(object_var.shape),
                                    columns=object_var.columns)

    for i in range(0, len(object_var.columns)):

        temp_var = object_var.iloc[:, i]

        ##除空值外的取值种类
        unique_value = np.unique(temp_var.iloc[np.where(~temp_var.isna())[0]])

        transform_rule = pd.concat([pd.DataFrame(unique_value, columns=['raw data']),
                                    pd.DataFrame(np.zeros([len(unique_value), 2]),
                                                 columns=['transform data', 'bad rate'])], axis=1)
        for j in range(0, len(unique_value)):
            bad_num = len(np.where((Y == 1) & (temp_var == unique_value[j]))[0])
            all_num = len(np.where(temp_var == unique_value[j])[0])

            # 计算badprob
            if all_num == 0:  # 防止all_num=0的情况，报错
                all_num = 0.5
            transform_rule.iloc[j, 2] = 1.0000000 * bad_num / all_num

        # 按照badprob排序，给出转换后的数值
        transform_rule = transform_rule.sort_values(by='bad rate')
        transform_rule.iloc[:, 1] = list(range(len(unique_value), 0, -1))

        # 保存转换规则
        object_transfer_rule[i] = {object_var.columns[i]: transform_rule}
        # 转换变量
        for k in range(0, len(unique_value)):
            transfer_value = transform_rule.iloc[np.where(transform_rule.iloc[:, 0] == unique_value[k])[0], 1]
            object_transform.iloc[np.where(temp_var == unique_value[k])[0], i] = float(transfer_value)
        object_transform.iloc[np.where(object_transform.iloc[:, i] == 0)[0], i] = np.nan

    X_transformed = pd.concat([num_var, object_transform], axis=1)
    return (X_transformed, transform_rule)

#去除缺失率、单一值率过高的变量
def missing_identity_select(data,y='flagy', missing_rate=0.9, identity_rate=0.9):
    null_ratio = data.isnull().sum() / data.shape[0]
    data1 = data.iloc[:,np.where(null_ratio <= missing_rate)[0]]
    identity = data1.drop(columns='flagy').apply(lambda x: x.value_counts().max() / x.size).reset_index(name='identity_rate').rename(columns={'index': 'variable'})
    identity_vars = identity[identity['identity_rate'] <= identity_rate]['variable'].to_list()
    data1 = data1[identity_vars + ['flagy']]
    return data1
#剔除自变量与因变量相关性过低、自变量与自变量相关性高的变量
def delete_corelation(data, y='flagy', y_cor=0.1, x_cor=0.7):
    cor = data.corr().abs()
    cor_y = cor[y]
    y_select = cor_y[cor_y >= y_cor ]
    y_select.sort_values(ascending=False)
    cor_x = cor.loc[y_select.index.values, y_select.index.values]
    cor_x.drop(labels=y, inplace=True)
    cor_x.drop(columns=y, inplace=True)
    drop_index = []
    for i in range(len(cor_x)-1):
        drop_index.extend(np.where(cor_x.iloc[i+1:,i] >= x_cor)[0])
    drop_index = list(set(drop_index))
    drop_name = cor_x.index[drop_index].tolist()
    cor_x.drop(index=drop_name, inplace=True)
    final_vars = cor_x.index.tolist() + [y]
    data = data[final_vars]
    return data
#变量PSI计算
def psi_var(data_train, data_test):
    data_train1 = data_train.copy()
    data_test1 = data_test.copy()
    var_names = data_train.columns.tolist()
    drop = []
    for name in var_names:
        num = data_train[name].unique().shape[0]
        if num <= 6:
            breaks = np.linspace(data_train[name].min()-1, data_train[name].max(), num+1)
        else:
            breaks = np.linspace(data_train[name].min()-1, data_train[name].max(), 6+1)
        data_train1[name] = pd.cut(data_train[name], bins=breaks, right=True)
        data_test1[name] = pd.cut(data_test[name], bins=breaks, right=True)
        data_train1[name] = data_train1[name].astype(str)
        data_test1[name] = data_test1[name].astype(str)
        df1 = data_train1[name].value_counts().rename('train')
        df2 = data_test1[name].value_counts().rename('test')
        df = pd.concat([df1, df2], axis=1)
        df['train_ratio'] = df.train / df.train.sum()
        df['test_ratio'] = df.test / df.test.sum()
        df['psi'] = (df.train_ratio - df.test_ratio) * np.log(df.train_ratio / df.test_ratio)
        df.psi.replace(np.inf, 0)
        if df.psi.sum() > 0.1:
#             del data_train[name]
#             del data_test[name]
            drop.append(name)
        del df1
        del df2
        del df
    print('PSI偏高的被剔除的变量：\n{}'.format(drop))
    return drop
    