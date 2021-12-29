# _*_ coding=utf-8 _*_
from flask import Flask
from flask import request, jsonify
import json
import os, sys
import pandas as pd
app = Flask(__name__)

'''
@app.route('/post/originLgbmodel', methods=['GET', 'POST'])
def post():
    data = request.get_json()
    print(data)
        {"--dataset":"upload/datasets/creditcard.csv", "--missing_rate":0.95, "--identity_rate":0.95}
    print("Has Received")
    print(data["--dataset"], data["--missing_rate"], data["--identity_rate"])
    # ss = os.system("python3 ./data_analysis_util_api/main.py")
    ss = os.system(
        "python3 ./data_analysis_util_api/main.py" + " --dataset " + data["--dataset"] + " --missing_rate " + str(
            data["--missing_rate"]) + " --identity_rate " + str(data["--identity_rate"]))
    print("Success")
    with open('./data_analysis_util_api/return_data.json', 'r') as f:
        load_dict = json.load(f)

    return jsonify(
        data=json.dumps(load_dict),
        extra={
            'message': 'success'
        }
    )
'''

# coding:utf-8

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        print(basepath)
        upload_path = os.path.join(basepath, 'static/', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        print(upload_path)

        f.save(upload_path)
        df = pd.read_csv(upload_path)
        returnDict = {}
        returnDict["threadList"] = list(df.columns)
        returnDict["dataLists"] = []
        returnDict["filePath"] = upload_path
        for index, row in df.iterrows():
            returnDict["dataLists"].append(dict(row))
            if index >= 5:
                break
        with open("fileReturnJson.json", 'w') as file_obj:
            json.dump(returnDict, file_obj)
        return jsonify(
            data=json.dumps(returnDict),
            extra={
                'status': 1,
                'message': 'success'
            }
        )
        return redirect(url_for('upload'))
    return render_template('upload.html')


sys.path.append(os.path.abspath(os.path.join(sys.path[0], '..')))
import data_analysis_util_api.models.model_rule


@app.route('/post/ruleModel', methods=['GET', 'POST'])
def index():
    data = request.get_json()
    print(data)
    print("Has Received")
    res = data_analysis_util_api.models.model_rule.model_rule(
        dataset="data_analysis_util_api/upload/datasets/creditcard.csv",
        deleteValue=[],
        cateValue="Class",
        lift_down_lmt=data["lift_down_lmt"],
        badrate_down_lmt=data["badrate_down_lmt"],
        Rule_min_cnt=data["Rule_min_cnt"],
        lift_need=data["lift_need"],
        min_bad_add=data["min_bad_add"],
        nan_ratio_threshold=data['nan_ratio_threshold'],
        mode_ratio_threshold=data['mode_ratio_threshold'],
        nan_replace_num=data['nan_replace_num'],
        train_size=data['train_size'],
        max_leaf_num=data['max_leaf_num'],
        min_woe_box_percent=data['min_woe_box_percent'],
        min_woe_box_num_min=data['min_woe_box_num_min'],
        cor_max=data['cor_max'])
    #res = data_analysis_util_api.models.model_rule.model_rule("data_analysis_util_api/upload/datasets/creditcard.csv", deleteValue=[], cateValue='Class',lift_down_lmt=2.5, badrate_down_lmt=0.2, Rule_min_cnt=100, lift_need=3, min_bad_add=20,nan_ratio_threshold=data['nan_ratio_threshold'])
    print(res)
    rule_final_tablePath = res["rule_final_table"]
    rule_result_tablePath = res["rule_result_table"]
    df = pd.read_csv(rule_final_tablePath)
    rule_final_tableDict = {}
    rule_final_tableDict["threadList"] = list(df.columns)
    rule_final_tableDict["dataLists"] = []
    rule_final_tableDict["filePath"] = rule_final_tablePath
    for index, row in df.iterrows():
        rule_final_tableDict["dataLists"].append(dict(row))
        if index >= 5:
            break

    df = pd.read_csv(rule_result_tablePath)
    rule_result_tablePathDict = {}
    rule_result_tablePathDict["threadList"] = list(df.columns)
    rule_result_tablePathDict["dataLists"] = []
    rule_result_tablePathDict["filePath"] = rule_result_tablePath
    for index, row in df.iterrows():
        rule_result_tablePathDict["dataLists"].append(dict(row))
        if index >= 5:
            break

    data = {"rule_final_table": rule_final_tableDict,
            "rule_result_table": rule_result_tablePathDict}
    return jsonify(
        data=json.dumps(data),
        extra={
            'status': 1,
            'message': 'success'
        }
    )
    return '<h1>Hello World</h1>'

#sys.path.append(os.path.abspath(os.path.join(sys.path[0], '..')))
import data_analysis_util_api.models.model_lgb

@app.route('/post/lgbModel', methods=['GET', 'POST'])
def postLgb():
    data = request.get_json()
    print(data)
    print("Has Received")
    res = data_analysis_util_api.models.model_lgb.model_lgb(
        dataset="data_analysis_util_api/upload/datasets/creditcard.csv",
        missing_rate=data['missing_rate'],
        flagy=data['flagy'],
        identity_rate=data['identity_rate'],
        train_size=data['train_size']
        )
    #result = data_analysis_util_api.models.model_lgb.model_lgb("data_analysis_util_api/upload/datasets/creditcard.csv","Class", missing_rate=0.95, identity_rate=0.95,train_size=0.7)
    print(res)

    return jsonify(
        data=json.dumps(res),
        extra={
            'status': 1,
            'message': 'success'
        }
    )

    return '<h1>Hello World</h1>'

if __name__ == '__main__':
    app.run(debug=True)
