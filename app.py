# _*_ coding=utf-8 _*_
from flask import Flask
from flask import request, jsonify
import json
import os, sys
app = Flask(__name__)
@app.route('/post/lgbmodel', methods=['GET', 'POST'])
def post():
    data = request.get_json()
    print(data)
    '''
    Template Json 
        {"--dataset":"upload/datasets/creditcard.csv", "--missing_rate":0.95, "--identity_rate":0.95}
    '''
    print("Has Received")
    print(data["--dataset"], data["--missing_rate"], data["--identity_rate"])
    #ss = os.system("python3 ./data_analysis_util_api/main.py")
    ss = os.system("python3 ./data_analysis_util_api/main.py" + " --dataset " + data["--dataset"] + " --missing_rate " + str(data["--missing_rate"]) + " --identity_rate " + str(data["--identity_rate"]))
    print("Success")
    with open('./data_analysis_util_api/return_data.json', 'r') as f:
        load_dict = json.load(f)

    return jsonify(
        data=json.dumps(load_dict),
        extra={
            'message': 'success'
        }
    )
# coding:utf-8

from flask import Flask,render_template,request,redirect,url_for
from werkzeug.utils import secure_filename
import os

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        print(basepath)
        upload_path = os.path.join(basepath,'static/',secure_filename(f.filename))  #注意：没有的文件夹一定要先创建，不然会提示没有该路径
        print(upload_path)
        f.save(upload_path)
        return redirect(url_for('upload'))
    return render_template('upload.html')
sys.path.append(os.path.abspath(os.path.join(sys.path[0], '..')))
import  data_analysis_util_api.models.model_rule
@app.route('/post/ruleModel', methods=['GET', 'POST'])
def index():
    data = request.get_json()
    print(data)
    print("Has Received")
    res = data_analysis_util_api.models.model_rule.model_rule("data_analysis_util_api/upload/datasets/creditcard.csv", deleteValue=[], cateValue='Class',lift_down_lmt=data["lift_down_lmt"], badrate_down_lmt=data["badrate_down_lmt"], Rule_min_cnt=data["Rule_min_cnt"],lift_need=data["lift_need"], min_bad_add=data["min_bad_add"])
    #res = data_analysis_util_api.models.model_rule.model_rule("data_analysis_util_api/upload/datasets/creditcard.csv", deleteValue=[], cateValue='Class',lift_down_lmt=2.5, badrate_down_lmt=0.2, Rule_min_cnt=100, lift_need=3, min_bad_add=20)
    print(res)

    return jsonify(
        data=json.dumps(res),
        extra={
            'message': 'success'
        }
    )
    return '<h1>Hello World</h1>'
if __name__ == '__main__':
    app.run(debug=True)
