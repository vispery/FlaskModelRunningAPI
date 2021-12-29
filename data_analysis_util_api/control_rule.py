from flask import Flask, redirect, url_for, request, render_template
import models.model_rule

app = Flask(__name__)  # WSGI应用程序

# route装饰器，告诉flask什么样的URL才能触发我们的函数，返回我们想要显示在用户浏览器中的信息


@app.route('/upload')
def upload_file():
    return render_template('upload.html')


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


@app.route('/', methods=['POST'])
def rule_run():
    #  从前端获取数据
    data = request.get_json(silent=True)
    # 调用函数
    res = models.model_rule.model_rule('upload/datasets/creditcard.csv', 'Class', 'V11',
                                       2.5, 0.2, 100, 3, 20)
    #  根据前端的需求，返回相应名称的参数
    return res

# @app.route('/upload/file', methods=['POST', 'GET'])
# def uploader():
    # if request.method == 'POST':
    #     data = request.get_json(silent=True)
    #     res = models.model_rule.model_rule('../upload/creditcard.csv', 'Time', 'V11',
    #                                        2.5, 0.2, 100, 3, 20, 0.999, 0.999, -999, 0.7, 6, 0.01, 50, 0.9)
        # file_path = data['files']  # 应该是个路径
        # modeltype = data['modeltype']

        # res = {
        #     'status': 0,
        #     'data': {
        #         'header': ["Col1", "Col2", "Col3"],
        #         'rows': [
        #             {1, 2, 3},
        #             {1, 2, 3}
        #         ],
        #         'file': 'file_name',
        #     },
        #     'msg': '提示信息'
        # }
        # f = request.files['file']
        # print(request.files)
        # f.save(os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(f.filename)))
        # return res
    # else:
    #     return render_template('upload.html')



if __name__ == '__main__':
    app.run(debug=True)
