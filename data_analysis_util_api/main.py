#!/usr/bin/env python
# -*- coding: utf-8 -*-
# python version: 3.7

import time
import json
from models.model_lgb import model_lgb,dataset_scan
from models.model_rule import model_rule
import argparse


if __name__=="__main__":
    '''

     parser = argparse.ArgumentParser(description='Input LgbModel Argument')
 
     parser.add_argument('--dataset', type = str,default="/Users/zexal/PycharmProjects/flaskProject/data_analysis_util_api/upload/datasets/creditcard_all.csv", help='The FilePath of DataSet of LgbModel')
     parser.add_argument('--missing_rate', type=float, default=0.95, help='Threshold for removing missing values')
     parser.add_argument('--identity_rate', type=float, default=0.95, help='Threshold for excluding a single value')
 
     args = parser.parse_args()
 
     print(args.dataset)
     print(args.missing_rate)
     print(args.identity_rate)
   
     global  ITERATION
     start_time = time.time()
     result = model_lgb(args.dataset, "Class", missing_rate=args.missing_rate, identity_rate=args.identity_rate, train_size=0.7)
     #result = model_lgb("upload/datasets/creditcard.csv","Class",missing_rate=0.95, identity_rate=0.95, train_size=0.7)
     print(result)
     end_time = time.time()
     print('Final 模型运行时间为：', end_time-start_time)

    start_time = time.time()
    rt = model_rule("upload/datasets/creditcard.csv", deleteValue=[], cateValue='Class',
                    lift_down_lmt=2.5, badrate_down_lmt=0.2, Rule_min_cnt=100, lift_need=3, min_bad_add=20)
    end_time = time.time()
    print('模型运行时间为：', end_time-start_time)
'''
    global  ITERATION
    start_time = time.time()
    result = model_lgb("upload/datasets/creditcard.csv","Class",missing_rate=0.95, identity_rate=0.95, train_size=0.7)
    print(result)
    end_time = time.time()
    print('Final 模型运行时间为：', end_time-start_time)