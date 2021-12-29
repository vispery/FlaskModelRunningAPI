# -*- coding: utf-8 -*-
"""
Created on Mon May 28 22:44:12 2018

@author: wanghuanhuan
"""

import numpy as np
import pandas as pd
import re
from sklearn import tree
##-----------------------==========定义数据预处理函数===============--------------------------------------#
'''
#X：需要预处理的数据集
#nan_ratio_threshold:非空值比例参数
#mode_ratio_threshold:取值集中度参数
'''
def pre_deal(X, nan_ratio_threshold = 0.99, mode_ratio_threshold = 0.99):
    #删掉缺失值超过99%（可调整）的变量
    count_null = np.where(X.isnull(), 1, 0)
    count_null_sumfactor = sum(count_null) / np.shape(X)[0]
    X = X.iloc[:, np.where(count_null_sumfactor <= nan_ratio_threshold)[0]]
    #删掉非nan同值超过99%（可调整）的变量
    raw_feature = X.columns.values
    if_delete_feature = np.zeros([len(raw_feature), 1])
    for i in range(len(raw_feature)):
        temp = X.iloc[np.where(~X.iloc[:, i].isna())[0],:]
        if_delete_feature[i] = ( temp.groupby(raw_feature[i])[raw_feature[i]].count().max() / len(temp) > mode_ratio_threshold)
        
    X = X.iloc[:, np.where(if_delete_feature == 0)[0]]
    return X
    
#--------------------====================OVER=========================================-----------------#

##---------------------=================定义类别变量数值化转化函数======================---------------------------#
def cate_var_transform(X,Y):
    ##取出数据类型
    d_type = X.dtypes
    object_var = X.iloc[:, np.where(d_type == "object")[0]]
    num_var = X.iloc[:, np.where(d_type != "object")[0]]
    
    #object_transfer_rule用于记录每个类别变量的数值转换规则
    object_transfer_rule = list(np.zeros([len(object_var.columns)])) 
    
    #object_transform是类别变量数值化转化后的值
    object_transform = pd.DataFrame(np.zeros(object_var.shape),
                                    columns=object_var.columns) 
    
    for i in range(0,len(object_var.columns)):
        
        temp_var = object_var.iloc[:, i]
        
        ##除空值外的取值种类
        unique_value=np.unique(temp_var.iloc[np.where(~temp_var.isna() )[0]])
    
        transform_rule=pd.concat([pd.DataFrame(unique_value,columns=['raw data']),
                                       pd.DataFrame(np.zeros([len(unique_value),2]),
                                                    columns=['transform data','bad rate'])],axis=1) 
        for j in range(0,len(unique_value)):
            bad_num=len(np.where( (Y == 1) & (temp_var == unique_value[j]) )[0])
            all_num=len(np.where(temp_var == unique_value[j])[0])
            
            #计算badprob
            if all_num == 0:#防止all_num=0的情况，报错
                all_num=0.5  
            transform_rule.iloc[j,2] = 1.0000000*bad_num/all_num
        
        #按照badprob排序，给出转换后的数值
        transform_rule = transform_rule.sort_values(by='bad rate')
        transform_rule.iloc[:,1]=list(range(len(unique_value),0,-1))
         
        #保存转换规则
        object_transfer_rule[i] = transform_rule
        #转换变量
        for k in range(0,len(unique_value)):
            transfer_value = transform_rule.iloc[np.where(transform_rule.iloc[:,0] == unique_value[k])[0],1]
            object_transform.iloc[np.where(temp_var == unique_value[k])[0],i] = float(transfer_value)
        object_transform.iloc[np.where(object_transform.iloc[:,i] == 0)[0],i] = np.nan 
    
    X_transformed = pd.concat([num_var,object_transform],axis = 1) 
    return(X_transformed,transform_rule)
#--------------------====================OVER=========================================-----------------#

##---------------------=================定义WOE分箱函数======================---------------------------#
'''
#Data_transformed:类型变量转换为数值变量后的数据集
#max_leaf_num:分箱的最大箱数
#min_woe_box_percent:叶节点最小样本量比例（仅占非空值的）
#min_woe_box_num_min:叶节点最小样本量；满足叶节点最小样本量比例，且满足叶节点最小样本量
'''
def myWOEbin(Data_transformed,y,max_leaf_num = 6, min_woe_box_percent = 0.01,min_woe_box_num_min = 100):    
    #每个分箱的数量不得小于总体取值非空样本的1%（可调整
    NonNan_num = np.sum(~Data_transformed.isna(), axis=0)
    min_woe_box_num = pd.Series(np.multiply(NonNan_num, min_woe_box_percent),
                                dtype=int)
    for i in range(0, len(min_woe_box_num)):
        if min_woe_box_num[i] < min_woe_box_num_min:
            min_woe_box_num[i] = min_woe_box_num_min 

    var_num = len(Data_transformed.columns)
    
    #var_splitpoint用于存储每一个变量的分箱截点
    var_splitpoint = list(np.zeros([var_num, 1]))
    
    for i in range(0, var_num):
        #非空的取值才进行决策树最优分箱
        temp_var = Data_transformed.iloc[:, i]
        NonNan_position = np.where(~temp_var.isna())[0]
        Nan_position = np.where(temp_var.isna())[0]
        
        #若没有nan项，则决策树最优分箱的最大箱数为max_leaf_num，否则，决策树最大箱数为max_leaf_num-1（因为缺失值会单独分箱）
        if len(Nan_position) == 0:
             max_leaf=max_leaf_num
        else:
             max_leaf=max_leaf_num-1
             
        #最多分到5个叶子节点
        groupdt = tree.DecisionTreeClassifier(criterion='entropy',
                                              min_samples_leaf=min_woe_box_num[i],
                                              max_leaf_nodes=max_leaf)
        
        groupdt.fit( np.array(Data_transformed.iloc[NonNan_position,i]).reshape(-1,1), y.iloc[NonNan_position])
        dot_data = tree.export_graphviz(groupdt, out_file=None, )
        pattern = re.compile('<= (.*?)\\\\nentropy', re.S)
        split_num = re.findall(pattern, dot_data)
        splitpoint = [float(j) for j in split_num]    
        final_splitpoint=sorted(splitpoint)
        final_splitpoint.append(np.inf)
        var_splitpoint[i]=final_splitpoint
        
    var_splitpoint=pd.Series(var_splitpoint,index=Data_transformed.columns)
    return var_splitpoint
#------------------===========================OVER===============================----------------------#

##-----------------------============定义处理WOE分箱结果的函数==========---------------------------------#
'''
#X:特征数据集
#Y:目标变量
#left_var：需要抹平左边界的字段
#right_var：需要抹平右边界的字段
#oth_var：其他字段
#var_splitpoint:上一步分箱的结果
#lift_need:用于下一步进行箱与箱交叉的箱需满足的提升度 或者
#badrate_need:用于下一步进行箱与箱交叉的箱需满足的提升度
'''
def deal_WOEbin(X,Y,left_var,right_var,oth_var,var_splitpoint,lift_need = 3,badrate_need = 0.3):
    rule_X = []
    rule = []
    badrate = sum(Y) / np.shape(Y)[0]
    ##left_var分箱处理，分箱的下界不是－inf的箱，修改成（－inf，a）
    if(len(left_var) > 0):
        for i in range(len(left_var)):
            temp_X = X.loc[:, left_var[i]]
            for j in range(len(var_splitpoint[left_var[i]])):
                temp_rule_X = pd.Series(np.zeros(len(X)))
                temp_rule_X.iloc[np.where(temp_X < var_splitpoint[left_var[i]][j])[0]] = 1
                temp_badnum = sum(temp_rule_X.iloc[np.where((Y == 1) & (temp_rule_X == 1))[0]])
                if(sum(temp_rule_X) == 0):
                    temp_badrate = 0
                else:
                    temp_badrate = temp_badnum / sum(temp_rule_X)
                # if ( (temp_badrate > lift_need*badrate) or (temp_badrate >badrate_need) ):
                if temp_badrate > lift_need * badrate:
                    temp_rule = [left_var[i],-np.inf,var_splitpoint[left_var[i]][j],temp_badnum,sum(temp_rule_X),temp_badrate]
                    rule_X.append(temp_rule_X)
                    rule.append(temp_rule)

    ##right_var分箱处理，分箱的上界不是inf的箱，修改成［a，inf）
    if(len(right_var)>0):
        for i in range(len(right_var)):
            temp_X = X.loc[:,right_var[i]]
            for j in range(len(var_splitpoint[right_var[i]])):
                if (var_splitpoint[right_var[i]][j] == np.inf):
                    temp_rule_X = pd.Series(np.zeros(len(X)))
                    temp_rule_X.iloc[np.where(~temp_X.isna())[0]] = 1
                    temp_badnum = sum(temp_rule_X.iloc[np.where((Y == 1) & (temp_rule_X == 1))[0]])
                    if(sum(temp_rule_X) == 0):
                        temp_badrate = 0
                    else:
                        temp_badrate = temp_badnum / sum(temp_rule_X)
                    # if ( (temp_badrate > lift_need*badrate) or (temp_badrate >badrate_need) ):
                    if temp_badrate > lift_need * badrate:
                        temp_rule = [right_var[i],-np.inf,np.inf, temp_badnum,sum(temp_rule_X),temp_badrate]
                        rule_X.append(temp_rule_X)
                        rule.append(temp_rule)
                else :
                    temp_rule_X = pd.Series(np.zeros(len(X)))
                    temp_rule_X.iloc[np.where(temp_X >= var_splitpoint[right_var[i]][j])[0]] = 1
                    temp_badnum = sum(temp_rule_X.iloc[np.where( (Y == 1) & (temp_rule_X == 1))[0]] )
                    if(sum(temp_rule_X) == 0):
                        temp_badrate = 0
                    else:
                        temp_badrate = temp_badnum / sum(temp_rule_X)
                    # if ( (temp_badrate > lift_need*badrate) or (temp_badrate >badrate_need) ):
                    if temp_badrate > lift_need * badrate:
                        temp_rule = [right_var[i],var_splitpoint[right_var[i]][j],np.inf,temp_badnum,sum(temp_rule_X),temp_badrate]
                        rule_X.append(temp_rule_X)
                        rule.append(temp_rule)               
    
    ##其他分箱结果处理
    if(len(oth_var)>0):
        for i in range(len(oth_var)):
            temp_X = X.loc[:, oth_var[i]]
            if(len(var_splitpoint[oth_var[i]]) == 1):
                temp_rule_X = pd.Series(np.zeros(len(X)))
                temp_rule_X.iloc[np.where(~temp_X.isna())[0]] = 1
                temp_badnum = sum(temp_rule_X.iloc[np.where((Y == 1) & (temp_rule_X == 1))[0]])
                if(sum(temp_rule_X) == 0):
                    temp_badrate = 0
                else:
                    temp_badrate = temp_badnum / sum(temp_rule_X)
                # if ( (temp_badrate > lift_need*badrate) or (temp_badrate >badrate_need) ):
                if temp_badrate > lift_need * badrate:
                    temp_rule = [oth_var[i],-np.inf,np.inf, temp_badnum,sum(temp_rule_X),temp_badrate]
                    rule_X.append(temp_rule_X)
                    rule.append(temp_rule)
            else:
                for j in range( len(var_splitpoint[oth_var[i]]) ):
                    if ( var_splitpoint[oth_var[i]][j] == np.inf):
                        temp_rule_X = pd.Series(np.zeros(len(X)))
                        temp_rule_X.iloc[np.where(temp_X>=var_splitpoint[oth_var[i]][j-1])[0]] = 1
                        temp_badnum = sum(temp_rule_X.iloc[np.where( (Y == 1) & (temp_rule_X == 1))[0]] )
                        if(sum(temp_rule_X) == 0):
                            temp_badrate =0 
                        else:
                            temp_badrate = temp_badnum / sum(temp_rule_X)
                        # if ( (temp_badrate > lift_need*badrate) or (temp_badrate >badrate_need) ):
                        if temp_badrate > lift_need * badrate:
                            temp_rule = [oth_var[i],var_splitpoint[oth_var[i]][j-1],np.inf, temp_badnum,sum(temp_rule_X),temp_badrate]
                            rule_X.append(temp_rule_X)
                            rule.append(temp_rule)  
                    elif(j == 0):
                        temp_rule_X = pd.Series(np.zeros(len(X)))
                        temp_rule_X.iloc[np.where( temp_X < var_splitpoint[oth_var[i]][j])[0]] = 1
                        temp_badnum = sum(temp_rule_X.iloc[np.where( (Y == 1) & (temp_rule_X == 1))[0]] )
                        if(sum(temp_rule_X) == 0):
                            temp_badrate =0 
                        else:
                            temp_badrate = temp_badnum / sum(temp_rule_X)
                        # if ( (temp_badrate > lift_need*badrate) or (temp_badrate >badrate_need) ):
                        if temp_badrate > lift_need * badrate:
                            temp_rule = [oth_var[i],-np.inf,var_splitpoint[oth_var[i]][j],temp_badnum,sum(temp_rule_X),temp_badrate]
                            rule_X.append(temp_rule_X)
                            rule.append(temp_rule)
                    else:
                        temp_rule_X = pd.Series(np.zeros(len(X)))
                        temp_rule_X.iloc[np.where( (temp_X < var_splitpoint[oth_var[i]][j]) & (temp_X >= var_splitpoint[oth_var[i]][j-1]) )[0]] = 1
                        temp_badnum = sum(temp_rule_X.iloc[np.where( (Y == 1) & (temp_rule_X == 1))[0]] )
                        if(sum(temp_rule_X) == 0):
                            temp_badrate =0 
                        else:
                            temp_badrate = temp_badnum / sum(temp_rule_X)
                        # if ( (temp_badrate > lift_need*badrate) or (temp_badrate >badrate_need) ):
                        if temp_badrate > lift_need * badrate:
                            temp_rule = [oth_var[i],var_splitpoint[oth_var[i]][j-1],var_splitpoint[oth_var[i]][j],temp_badnum,sum(temp_rule_X),temp_badrate]
                            rule_X.append(temp_rule_X)
                            rule.append(temp_rule)
    rule = pd.DataFrame(rule, columns = ["var","dowm_lmt","up_lmt", "badnum","hit_num","badrate"])
    rule_X = pd.DataFrame(rule_X).T
    return rule_X, rule
#----------------------------------================OVER==================--------------------------------------#

#----------------------------------======定义筛选有效规则集函数 version1=======----------------------------#

'''第一步取坏账率最高的规则入有效规则集，按坏账率从高到低加入规则集，每往有效规则集中添加一条，必须满足有效规则集命中的坏人数增加>min_bad_add 
#rule:规则集
#rule_X:样本映射成规则集的0、1矩阵
#Y:目标变量
#min_bad_add:增加坏样本阈值
#badrate:rule规则集中表示坏样本率的字段名称
''' 
def useful_rule_v1(rule,rule_X, Y, min_bad_add = 30, badrate = "badrate"):
    order = np.argsort(-np.array(rule.loc[:,badrate]))
    if_choose = pd.DataFrame(np.zeros([len(rule),1]))
    
    #首次选入坏账率最高的规则
    rule_useful_set = pd.DataFrame(rule_X.iloc[:,order[0]])
    if_choose.iloc[order[0],:] = 1
    
    for i in range(1,len(rule)):
        print(i)
        rule_new = pd.Series( np.zeros([len(rule_X)]) )
        temp = rule_X.iloc[:,order[i]] - rule_useful_set.iloc[:,0]
        rule_new.iloc[np.where(temp == 1)[0]] = 1
        
        new_bad = np.dot(np.array(rule_new),np.array(Y))
        
        if(new_bad>min_bad_add):
            if_choose.iloc[order[i],:] = 1
            temp = rule_X.iloc[:,order[i]] + rule_useful_set.iloc[:,0]
            rule_useful_set.iloc[np.where(temp > 0)[0],0] = 1
    return if_choose

#------------------------======================OVER====================-------------------------------#

#----------------------------------======定义筛选有效规则集函数 version2=======----------------------------#   
'''
#第一步取坏账率最高的规则入有效规则集，每往有效规则集中添加一条，从满足有效规则集命中的坏人数增加>min_bad_add，选择使得规则集的坏账率最高的规则
#rule:规则集
#rule_X:样本映射成规则集的0、1矩阵
#Y:目标变量
#min_bad_add:增加坏样本阈值
#badprob:rule规则集中表示坏样本率的字段名称
'''
def useful_rule_v2(rule,rule_X, Y, min_bad_add = 30):
    order = np.argsort(-np.array(rule.loc[:,"badprob"]))
    rule = rule.iloc[order,:]
    rule_X = rule_X.iloc[:,order]
    
    if_choose = pd.DataFrame(np.zeros([len(rule),1]))
    #首次选入坏账率最高的规则
    rule_useful_set = pd.DataFrame(rule_X.iloc[:,0])
    if_choose.iloc[0,:] = 1
    
    rule_rest = [i for i in range(1,len(rule))]
    
    while(len(rule_rest)>0):
        print(len(rule_rest))
        if_choose_temp = pd.DataFrame(np.zeros([len(rule),1]))
        badrate_temp = pd.DataFrame(np.zeros([len(rule),1]))
        
        useful_bad = sum(np.multiply(np.array(rule_useful_set.iloc[:,0]),np.array(Y)))
        
        for i in rule_rest:
            rule_new = pd.Series( np.zeros([len(rule_X)]) )
            temp = rule_X.iloc[:,i] + rule_useful_set.iloc[:,0]
            rule_new.iloc[np.where(temp > 0)[0]] = 1
            
            badnum_temp = np.dot(np.array(rule_new),np.array(Y))
            
            if(badnum_temp-useful_bad>min_bad_add):
                if_choose_temp.iloc[i,:] = 1
                badrate_temp.iloc[i,:] = badnum_temp/sum(rule_new)
        order_temp = np.argsort(-np.array(badrate_temp.iloc[:,0]))
        
        if_choose.iloc[order_temp[0],:] = 1
        if_choose_temp.iloc[order_temp[0],:] = 0
        
        rule_rest = np.where(if_choose_temp.iloc[:,0] == 1)[0]
        temp = rule_X.iloc[:,order_temp[0]] + rule_useful_set.iloc[:,0]
        rule_useful_set.iloc[np.where(temp > 0)[0],0] = 1
    rule_useful= rule.iloc[np.where(if_choose.iloc[:,0]==1)[0],:]
    rule_X_useful = rule_X.iloc[:,np.where(if_choose.iloc[:,0]==1)[0]]
    return rule_useful,rule_X_useful
#------------------------======================OVER====================-------------------------------#   
    
#-----------------------=================定义两两交叉函数===========------------------------------------#    
'''
#rule:规则集
#rule_X:样本映射成规则集的0、1矩阵
#Y:目标变量
#badrate_real:该数据集真实坏样本比例
#Rule_min_cnt:规则需命中的最小样本量
#lift_need:规则需要满足的提升度
'''
def rule_cross_rule(rule,rule_X,Y,badrate_real,Rule_min_cnt = 100,lift_need = 5):
    n = len(rule)
    badrate_need = badrate_real*lift_need
    if_cross = pd.DataFrame(np.zeros([n,n]))
    hitnum = pd.DataFrame(np.zeros([n,n]))
    badnum = pd.DataFrame(np.zeros([n,n]))
    badrate = pd.DataFrame(np.zeros([n,n]))
    
    for i in range(0, n - 1):
        print(i)
        var_i = rule.iloc[i, 0]
        for j in range(i+1, n):
            var_j = rule.iloc[j, 0]
            if(var_i != var_j):
                rule_new = pd.Series(np.zeros([len(rule_X)]))
                temp = rule_X.iloc[:,j] + rule_X.iloc[:,i]
                rule_new.iloc[np.where(temp == 2)[0]] = 1
                hitnum_temp = sum(rule_new)
                badnum_temp = np.dot(np.array(rule_new),np.array(Y))
                if(hitnum_temp == 0):
                    badrate_temp = 0
                else:
                    badrate_temp = badnum_temp/hitnum_temp
                hitnum.iloc[i,j] = hitnum_temp
                badnum.iloc[i,j] = badnum_temp
                badrate.iloc[i,j] = badrate_temp
                if( (badrate_temp>badrate_need) & (hitnum_temp>Rule_min_cnt) ):
                    if_cross.iloc[i,j] = 1
    return if_cross, hitnum, badnum, badrate
#------------------------======================OVER====================-------------------------------# 

#-------------------------===========定义计算两个矩阵之间的cosin相关性============--------------#
#定义两个向量之间的相关性
def cos_cal(vector1,vector2):
    #cos_result = sum(np.multiply(vector1,vector2))/math.sqrt(sum(vector1)) / math.sqrt(sum(vector2))
    cos_result = np.dot(vector1,vector2)/(np.linalg.norm(vector1)*np.linalg.norm(vector2))
    return cos_result
#定义一个向量和一个矩阵之间的相关性
def cos_dataframe1(vector,data):
    cos_result = data.apply(cos_cal,args = (vector,),axis = 0)
    return cos_result
#定义两个矩阵之间的相关性
def cos_dataframe2(data1,data2):
    cos_result = data1.apply(cos_dataframe1,args = (data2,),axis = 0)
    return cos_result
#------------------------======================OVER====================-------------------------------# 
    
#-------------------------===========筛除相关性高的规则============--------------#
'''
#rule_cor:规则集的相关系数矩阵
#rule:规则集
#rule_X:样本映射成规则集的0、1矩阵   
#cor_max:最大的相关系数
'''
def dealwithcor(rule_cor,rule,rule_X, cor_max):
    ##初步筛选相关性
    order = np.argsort(-np.array(rule.loc[:,"badrate"]))
    rule = pd.DataFrame(rule.iloc[order,:])
    rule_X = pd.DataFrame(rule_X.iloc[:,order])
    rule_cor = pd.DataFrame(rule_cor.iloc[order,order])
    for i in range(len(rule_cor)):
        rule_cor.iloc[i,i] = 0
    
    if_choose = pd.DataFrame(np.ones([len(rule),1]))
    
    ##取出相关性小于阈值的规则
    for i in range(len(rule)-1):
        if(if_choose.iloc[i,0] == 0 ):
            continue
        else:
            highcor = np.where(rule_cor.iloc[(i+1):,i]>=cor_max)[0]+i+1
            if_choose.iloc[highcor] = 0
    temp_choose = np.where(if_choose.iloc[:,0] == 1)[0]
    rule_cor_choose = pd.DataFrame(rule.iloc[temp_choose,:])
    rule_X_cor_choose = pd.DataFrame(rule_X.iloc[:,temp_choose])
    return rule_cor_choose,rule_X_cor_choose
#------------------------======================OVER====================-------------------------------# 






        
        

    