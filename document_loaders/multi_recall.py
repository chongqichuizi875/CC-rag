import sys
# print(sys.path)
sys.path.append('.')
from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
from bm25 import load_bm25, search
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from copy import deepcopy
from joblib import dump, load
import torch
import torch.nn as nn
from configs.kb_config import VECTOR_SEARCH_TOP_K
import os
import shutil
from server.knowledge_base.utils import KnowledgeFile
test_kb_name = "test"
kbService = FaissKBService(test_kb_name)
test_file_name = "陕汽-重卡X5000维修手册（第一部分）.pdf"
testKnowledgeFile = KnowledgeFile(test_file_name, test_kb_name)
def test_search_db(query):
    result = kbService.search_docs(query)
    return result
def test_clear_emb():
    emb_path = f"knowledge_base/{test_kb_name}/vector_store"
    if os.path.exists(emb_path):
        shutil.rmtree(emb_path)
def test_add_doc():
    assert kbService.add_doc(testKnowledgeFile)


queries = {
    "X5000 牵引车（柴油）质量参数":72,
    "X5000 牵引车（柴油）尺寸参数":73,
    "ISZ 系列发动机基本参数":76
}

def test_search_vec(queries):
    ret = []
    top1_acc = 0.0
    topk_acc = 0.0
    total = 0
    for query, global_index in queries.items():
        total += 1
        answers = test_search_db(query)
        one_answer_list = []
        for answer in answers:
            one_answer_list.append((answer[0].metadata['global_index'], answer[1]))
        if one_answer_list and one_answer_list[0][0] == global_index:
            top1_acc += 1
        if one_answer_list:
            for one_answer in one_answer_list:
                if one_answer[0] == global_index:
                    topk_acc += 1
                    break
        one_answer_dict = {
            "label": global_index,
            "answers": one_answer_list
        }
        ret.append(one_answer_dict)

    return ret, top1_acc/total, topk_acc/total


        
def test_search_bm25(queries):
    ret = []
    top1_acc = 0.0
    topk_acc = 0.0
    total = 0
    model = load_bm25()
    results = search(model, list(queries.keys()), k=VECTOR_SEARCH_TOP_K)
    labels = list(queries.values())
    with open ("document_loaders/111.json", 'r') as f:
        data = json.load(f)
    for i, result in enumerate(results):
        total += 1
        one_answer_list = []
        for answer in result:
            one_answer_list.append((data[answer[1]][1]['global_index'], answer[2]))
        if one_answer_list[0][0] == labels[i]:
            top1_acc += 1
        for one_answer in one_answer_list:
            if one_answer[0] == labels[i]:
                topk_acc += 1
                break
        one_answer_dict = {
            "label": labels[i],
            "answers": one_answer_list
        }
        ret.append(one_answer_dict)

    return ret, top1_acc/total, topk_acc/total

def merge_list(list1, list2, label):
    merged_dict = {}
    for key, value in list1:
        merged_dict[key] = [value]
    for key, value in list2:
        if key in merged_dict.keys():
            merged_dict[key].append(value)
        else:
            merged_dict[key] = [18]
            merged_dict[key].append(value)
    for key in merged_dict.keys():
        if len(merged_dict[key]) == 1:
            merged_dict[key].append(0.4)
        merged_dict[key].append(int(key==label))
    return deepcopy(merged_dict)

class ScoreNet(nn.Module):
    def __init__(self):
        super(ScoreNet, self).__init__()
        self.fc1 = nn.Linear(2, 10)  
        self.fc2 = nn.Linear(10, 1) 

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  
        x = self.fc2(x)
        return x


def training(queries, eval=True, save_model=True):
    answers_vec, top1_acc_vec, topk_acc_vec = test_search_vec(queries)
    answers_bm25, top1_acc_bm25, topk_acc_bm25 = test_search_bm25(queries)
    print(f"vec top1 accuracy: {top1_acc_vec}; topK accuracy: {topk_acc_vec}")
    print(f"bm25 top1 accuracy: {top1_acc_bm25}; topK accuracy: {topk_acc_bm25}")


    X = np.empty((0,2))
    Y = np.array([])
    for i, label in enumerate(queries.values()):
        answer_bm25 = answers_bm25[i]['answers']
        answer_vec = answers_vec[i]['answers']
        merged_dict = merge_list(answer_bm25, answer_vec, label)
        raw_data = np.array([t for t in merged_dict.values() if len(t) == 3])
        if raw_data.size > 0:
            X = np.vstack((X, raw_data[:,:2]))
            Y = np.append(Y, raw_data[:,-1])
    # print(np.hstack((X, Y.reshape((Y.size,1)))))
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)

    if eval:
        print(f"总共sample数量: {Y.size}, training set: {y_train.size}, test set: {y_test.size}")
        y_pred = model.predict(X_test)
        print("测试集acc:", accuracy_score(y_test, y_pred))
        print("\n分类报告:\n", classification_report(y_test, y_pred))
    if save_model:
        dump(model, 'document_loaders/model.joblib')
    
def load_multi_recall_model():
    try:
        model = load('document_loaders/model.joblib')
        return model
    except:
        print("no model available")
        return None
    
def test_search_multi_recall(model, queries):
    for query, label in queries.items():
        bm25_model = load_bm25()
        bm25_results = search(bm25_model, [query], k=VECTOR_SEARCH_TOP_K)[0]
        vec_results = test_search_db(query)
        print(bm25_results)
        break

if __name__ == "__main__":
    with open("document_loaders/queries.json", 'r') as f:
        queries = json.load(f)

    # training(queries)
    # model = load_multi_recall_model()
    # test_search_multi_recall(model, queries)
    test_clear_emb()
    test_add_doc()
    answers_vec, top1_acc_vec, topk_acc_vec = test_search_vec(queries)
    print(f"vec top1 accuracy: {top1_acc_vec}; topK accuracy: {topk_acc_vec}")



            