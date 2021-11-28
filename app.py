# _*_ coding=utf-8 _*_
from flask import Flask
from flask import request, jsonify
import json
import os, sys
app = Flask(__name__)
@app.route('/post', methods=['GET', 'POST'])
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
