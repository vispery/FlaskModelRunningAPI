#import json
import time
import lightgbm as lgb
import pandas as pd
import numpy as np
import json
import sys, os
#import re
#import matplotlib.pylab as plt
#from sklearn import metrics
from sklearn.metrics import mean_squared_error
import scorecardpy as sc
#from sklearn.feature_selection import RFE
#from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
#from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
#import csv
from hyperopt import STATUS_OK
from timeit import default_timer as timer
from hyperopt import hp
from hyperopt.pyll.stochastic import sample
from hyperopt import fmin, tpe, Trials
#import os

from data_analysis_util_api.utils import lgb_func as func


def dataset_scan(dataset):
    '''
        浏览数据集前5条数据
        输入：
            dataset: string 数据集csv文件
        输出：
            1. header: list 数据集的各变量名
            2. rows：list(list) 前5行数据
    '''
    data = pd.read_csv(dataset,nrows=5)
    header = list(data.columns)
    rows = np.array(data).tolist()
    return header,rows


def model_lgb(dataset,flagy,missing_rate=0.95, identity_rate=0.95, train_size=0.7):
    '''
        定义LGB模型处理函数
        输入:
            dataset：string 需要预处理的数据集文件存放位置
            flagy: string 标签列的title
            missing_rate: float 剔除缺失值的阈值，默认值0.95
            identity_rate: float 剔除单一值的阈值，默认值0.95
            train_size: float 训练集比例，默认值0.7

        输出: dict
            train_ks_auc_figure : string 训练集ks-auc统计结果图片地址.
            train_ks： float 训练集KS值.
            train_auc： float 训练集AUC值.
            test_ks_auc_figure : string 测试集KS-AUC指标结果图片地址.
            test_ks： float 测试集KS值.
            test_auc： float 测试集AUC值.
            feature_imp：string 特征重要性排序数据csv文件地址.
            psi_result_table: string PSI分值统计数据csv文件地址.
            psi_result_figure: string PSI统计结果图片地址.
    '''
    #函数返回的dict
    rt = {}
    Father_Catalog = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")
    #定义本次任务的一个标志，暂时用时间来代表
    job_flag = time.strftime("%Y%m%d%H%M%S", time.localtime())
    #读取数据集
    data = pd.read_csv(dataset, sep=',', encoding='utf-8')
    #统一将标签列名称设为flagy
    data.rename(columns={flagy:'flagy'},inplace=True)
    #统计正负样本数据
    data = data[data['flagy'] != 2]
    sample_stat = data.flagy.value_counts()
    #剔除缺失值和单一值过高的变量
    data = func.missing_identity_select(data, 'flagy', missing_rate=missing_rate, identity_rate=identity_rate)
    print('剔除缺失值和单一值过高的变量之后，变量的个数为{0}'.format(data.shape))
    #剔除类别型变量
    d_type = data.dtypes
    print('类别变量的个数为{0}'.format(sum((d_type == "object"))))
    rr = [i for i in data.columns if data[i].dtypes=='object']
    data.drop(rr,axis=1,inplace=True)
    print('剔除类别型变量后变量的个数为{0}'.format(data.shape))
    #划分train/test数据集
    Y = data['flagy']
    data_train, data_test = train_test_split(data, test_size=1-float(train_size), random_state=888, stratify=Y)
    data_train.reset_index(drop=True,inplace=True)
    data_test.reset_index(drop=True,inplace=True)
    Y_train = data_train['flagy']
    X_train = data_train.drop(columns=['flagy'])
    Y_test = data_test['flagy']
    X_test = data_test.drop(columns=['flagy'])

    #初始化模型
    W_train = np.ones(X_train.shape[0])
    W_test = np.ones(X_test.shape[0])
    lgb_train = lgb.Dataset(X_train, Y_train, weight=W_train, free_raw_data=False)
    lgb_eval = lgb.Dataset(X_test, Y_test, reference=lgb_train, weight=W_test, free_raw_data=False)

    #贝叶斯参数搜索
    MAX_EVALS = 100
    N_FOLDS = 5
    #定义超参数优化的目标函数 定义超参数优化（贝叶斯优化——Tree Parzen估计）
    def objective(params, n_folds = 5):
        """Objective function for Gradient Boosting Machine Hyperparameter Optimization"""

        # Keep track of evals
        global ITERATION

        ITERATION += 1

        # Retrieve the subsample if present otherwise set to 1.0
        subsample = params['boosting_type'].get('bagging_fraction', 1.0)

        # Extract the boosting type
        params['boosting_type'] = params['boosting_type']['boosting_type']
        params['bagging_fraction'] = subsample

        # Make sure parameters that need to be integers are integers
        for parameter_name in ['num_leaves', 'bagging_freq', 'min_data_in_leaf', 'max_bin', 'max_depth', 'n_estimators']:
            params[parameter_name] = int(params[parameter_name])
        for parameter_name in ['learning_rate', 'bagging_fraction', 'feature_fraction', 'reg_lambda', 'reg_alpha']:
            params[parameter_name] = round(params[parameter_name], 4)
        start = timer()

        # Perform n_folds cross validation
        lgb_model = lgb.LGBMClassifier(n_jobs = -1,
                                        objective = 'binary', random_state = 50, **params)
        lgb_model.fit(X_train, Y_train)
        pred1 = lgb_model.predict_proba(X_train)[:,1]
        pred2 = lgb_model.predict_proba(X_test)[:,1]
        auc1 = roc_auc_score(Y_train, pred1)
        auc2 = roc_auc_score(Y_test, pred2)
        run_time = timer() - start

        # Extract the best score
        loss = abs(auc1 - auc2) / auc2 -np.log(auc2)

        auc_list = [auc1,auc2]
        # Dictionary with information for evaluation
        return {'loss': loss, 'auc_list': auc_list,'params': params, 'iteration': ITERATION,
                'train_time': run_time, 'status': STATUS_OK}

    #定义搜索域
    space = {
        'boosting_type': hp.choice('boosting_type', [{'boosting_type': 'gbdt', 'bagging_fraction': hp.uniform('gbdt_bagging_fraction', 0.5, 1)}]),
        'num_leaves': hp.quniform('num_leaves', 4, 50, 1),
        'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.2)),
        'bagging_freq': hp.uniform('bagging_freq', 0, 5),
        'feature_fraction': hp.uniform('feature_fraction', 0.5, 1),
        'min_data_in_leaf': hp.uniform('min_data_in_leaf', 10, 100),
        'max_bin': hp.uniform('max_bin', 4, 20),
        'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
        'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
        'importance_type': 'gain',
        'max_depth': hp.uniform('max_depth', 2, 10),
        'n_estimators': hp.uniform('n_estimators', 50, 200)
    }
    # 定义优化函数
    bayes_trials = Trials()
    global  ITERATION
    ITERATION = 0
    # Run optimization
    best = fmin(fn = objective, space = space, algo = tpe.suggest,
                max_evals = MAX_EVALS, trials = bayes_trials, rstate = np.random.RandomState(123))
    #按loss从小到大排序，然后获取最优超参数
    bayes_trials_results = sorted(bayes_trials.results, key = lambda x: x['loss'])
    bayes_trials_results[:2]
    best_bayes_params = bayes_trials_results[0]['params']
    #用最优超参数训练模型
    print('Starting training...')
    start_time = time.time()
    best_bayes_model = lgb.LGBMClassifier(n_jobs = -1,
                                        objective = 'binary', random_state = 50, **best_bayes_params)
    best_bayes_model.fit(X_train, Y_train, eval_metric='auc', verbose=2)
    print('train over')
    end_time = time.time()
    print('训练时间为：', end_time - start_time)

    #AUC和KS值评估
    train_class_pred = best_bayes_model.predict_proba(X_train)[:, 1]
    test_class_pred = best_bayes_model.predict_proba(X_test)[:, 1]
    # get the ks
    fpr_train, tpr_train, thresholds_test = roc_curve(np.array(Y_train), train_class_pred)
    fpr_test, tpr_test, thresholds_test = roc_curve(np.array(Y_test), test_class_pred)
    ks_train = max(tpr_train - fpr_train)
    ks_test = max(tpr_test - fpr_test)
    print("KS(Train): %f" % ks_train)
    print("KS(Test): %f" % ks_test)
    # get the auc
    print("AUC Score(Train): %f" % roc_auc_score(Y_train, train_class_pred))
    print("AUC Score(Test): %f" % roc_auc_score(Y_test, test_class_pred))
    #评估和绘制AUC和KS结果图片,注入返回结果
    train_perf = sc.perf_eva(Y_train, train_class_pred, title="train")
    train_ks_auc_figure_path = 'data_analysis_util_api/output/lgb/figure/train_ks_auc'+job_flag+'.png'
    train_perf['pic'].savefig(train_ks_auc_figure_path)
    rt['train_ks_auc_figure'] = train_ks_auc_figure_path
    rt['train_ks'] = train_perf['KS']
    rt['train_auc'] = train_perf['AUC']
    test_perf = sc.perf_eva(Y_test, test_class_pred, title="test")
    test_ks_auc_figure_path = 'data_analysis_util_api/output/lgb/figure/test_ks_auc'+job_flag+'.png'
    test_perf['pic'].savefig(test_ks_auc_figure_path)
    rt['test_ks_auc_figure'] = test_ks_auc_figure_path
    rt['test_ks'] = test_perf['KS']
    rt['test_auc'] = test_perf['AUC']

    #特征重要性排序
    feature_imp = pd.Series(best_bayes_model.feature_importances_)
    feature_name = pd.Series(X_train.columns)
    feature_df = pd.DataFrame({'feature_name': feature_name, 'element': feature_imp})
    # feature_df = feature_df.reset_index().drop(columns=['index'])
    feature_df = feature_df.sort_values(by='element', ascending=False)
    feature_df = feature_df[feature_df['element'] > 0].reset_index(drop=True) #剔除重要性为0的变量
    feature_df['Gain'] = feature_df['element']/sum(feature_df['element'])
    feature_imp_path = 'data_analysis_util_api/output/lgb/csv/feature_imp'+job_flag+'.csv'
    feature_df.to_csv(feature_imp_path)
    rt['feature_imp'] = feature_imp_path

    ############################ 模型分映射 Score Distribution ###########################
    train_p = train_class_pred.copy()
    test_p = test_class_pred.copy()
    # whole_p = whole_class_pred.copy()
    points0 = 735
    pdo = 140
    odds0 = (data.flagy == 1).sum() / (data.flagy == 0).sum()
    B = pdo / np.log(2)
    A = points0 + B * np.log(odds0)
    train_score = pd.Series(np.around(A - B * np.log(train_p/(1 - train_p))))
    #train_score.to_csv('train_score.csv',encoding='utf_8_sig',index=False)
    test_score = pd.Series(np.around(A - B * np.log(test_p/(1 - test_p))))

    #分组统计 train,test
    #train
    arr1 = np.arange(train_score.shape[0])
    train_score = pd.Series(train_score)
    s1 = pd.Series(arr1)
    train_psi = pd.concat([train_score, s1, Y_train], ignore_index=True, axis=1)
    train_psi.columns = ['score', 'id', 'y']
    #test
    arr1 = np.arange(test_score.shape[0])
    test_score = pd.Series(test_score)
    s1 = pd.Series(arr1)
    test_psi = pd.concat([test_score, s1, Y_test], ignore_index=True, axis=1)
    test_psi.columns = ['score', 'id', 'y']
    psi_result=func.psi_stats_score1(test_psi, train_psi, non_computed=None, plot_image=True)
    #保存psi_result_table
    psi_result_table_path = 'data_analysis_util_api/output/lgb/csv/psi_result_table_'+job_flag+'.csv'
    psi_result['count_final'].to_csv(psi_result_table_path)
    rt['psi_result_table'] = psi_result_table_path
    #保存psi_result_figure
    psi_result_figure_path = 'data_analysis_util_api/output/lgb/figure/psi_result_'+job_flag+'.png'
    psi_result['figure'].savefig(psi_result_figure_path)
    rt['psi_result_figure'] = psi_result_figure_path

    print(rt)
    rt['train_ks_auc_figure'] = Father_Catalog + '/' + rt['train_ks_auc_figure']
    rt['test_ks_auc_figure'] = Father_Catalog + '/' + rt['test_ks_auc_figure']
    rt['feature_imp'] = Father_Catalog + '/' + rt['feature_imp']
    rt['psi_result_table'] = Father_Catalog + '/' + rt['psi_result_table']
    rt['psi_result_figure'] = Father_Catalog + '/' + rt['psi_result_figure']
    print(rt)
    json_str = json.dumps(rt)
    with open('return_data.json', 'w') as json_file:
        json_file.write(json_str)
    #return
    return rt

