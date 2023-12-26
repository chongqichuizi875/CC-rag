from fastbm25 import fastbm25
import json
import pickle
import re
import os
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)

def save_bm25():
    with open ("document_loaders/111.json", 'r') as f:
        data = json.load(f)
    # data[i][0] 长度
    # data[i][1] dict
    # data[i][2] 文本
    corpus = []
    for content in data:
        match = re.search(r'\[\[', content[2])
        context = content[2][:match.start()] if match else content[2]
        images = ' '.join(content[1]['images'])
        tables = ' '.join(content[1]['tables'])
        corpus.append(content[1]['titles']+images+tables+context)
    # corpus = [content[1]['titles']+': '+content[2] for content in data]
    model = fastbm25(corpus)
    with open("document_loaders/bm25.pkl", 'wb') as f:
        pickle.dump(model, f)

def load_bm25():
    with open("document_loaders/bm25.pkl", 'rb') as f:
        model = pickle.load(f)
    return model

def search(model, queries, k):
    results = [model.top_k_sentence(query, k=k) for query in queries]
    return results

if __name__ == "__main__":
    queries = {
    "X5000 牵引车（柴油）质量参数":7,
    "X5000 牵引车（柴油）尺寸参数":7,
    "WP13 系列发动机基本参数":8,
    "ISZ 系列发动机基本参数":10,
    "ISM 发动机基本参数":9,
    "X5000配了几款发动机":11,
    "WP10H 系列发动机技术参数":11,
    "WP11S 系列发动机技术参数":12,
    "WP13G 系列发动机技术参数":13,
    "目前广为应用的共轨系统有哪些公司":14,
    "ECU 电脑控制的高压共轨燃油喷射系统有哪些优点":14,
    "高压供轨燃油喷射系统原理图":15,
    "CPN2.2型高压油泵内部结构":18,
    "燃油计量阀的包含哪些零件":18,
    "喷油器结构":22,
    "ECU“跛形行走”功能是指":23,
    "高压共轨燃料喷射系统各元器件的存放要求":28,
    "诊断灯常亮表示什么意思":31,
    "诊断灯不亮但有故障码":31,
    "跛行回家是什么意思":31,
    "故障码的是怎么读取":31,
    "闪码由 3 位数组成":31,
    "电控系统故障与故障闪码对照表":31,
    "经常使用哪些方法来判断故障":31,
    "潍柴动力国V 系列柴油机采用什么共轨系统":34,
    "燃油计量阀的结构部件有哪些":19,
    "激活巡航的条件是":35,
    "发动机 PTO 功能":35,
    "SCR 是全称是什么意思":36,
    "多功率省油功能是指？":35,
    "SCR 系统的工作机理是":36,
    "添蓝是什么意思":41,
    "SCR 系统由哪些单元组成":38,
    "SCR 系统的组成":38,
    "尿素箱一般采用什么材料":40
    # "变速箱换档困难":84
    }
    save_bm25()
    model = load_bm25()
    results = search(model, list(queries.keys()), k=1)
    # results[i][0][0] 文本
    # results[i][0][1] index
    # results[i][0][2] 分数
    with open ("document_loaders/111.json", 'r') as f:
        data = json.load(f)
    answers = [data[result[0][1]] for result in results]
    correct = 0
    total = 0
    for index, label in enumerate(list(queries.values())):
        total += 1
        if answers[index][1]['content_pos'][0]['page_no'] <= label <= answers[index][1]['content_pos'][-1]['page_no']:
            correct += 1
        else:
            print(f"label: {label}, 回答区间: [{answers[index][1]['content_pos'][0]['page_no']},{answers[index][1]['content_pos'][-1]['page_no']}]")
        
    print(f"accuracy: {float(correct)/total}")