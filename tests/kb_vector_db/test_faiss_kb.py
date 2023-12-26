import sys
# print(sys.path)
sys.path.append('.')
from server.knowledge_base.kb_service.faiss_kb_service import FaissKBService
from server.knowledge_base.kb_service.milvus_kb_service import MilvusKBService
from server.knowledge_base.kb_service.pg_kb_service import PGKBService
from server.knowledge_base.kb_service.es_kb_service import ESKBService
from server.knowledge_base.kb_service.zilliz_kb_service import ZillizKBService
from server.knowledge_base.migrate import create_tables
from server.knowledge_base.utils import KnowledgeFile
import os
import shutil
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
kbService = FaissKBService("test")
test_kb_name = "test"
# test_file_name = "陕汽-新M3000S维修手册 第二部分.pdf"
# test_file_name = "陕汽L3000系列载货车维修手册（第二部分）.docx"
test_file_name = "陕汽-重卡X5000维修手册（第一部分）.pdf"
testKnowledgeFile = KnowledgeFile(test_file_name, test_kb_name)
search_content = "驱动车桥的速比数据表"


def test_init():
    create_tables()


def test_create_db(kbService):
    assert kbService.create_kb()


def test_add_doc(kbService, testKnowledgeFile):
    assert kbService.add_doc(testKnowledgeFile)


def test_search_db(kbService):
    result = kbService.search_docs(search_content)
    # result = kbService.do_search(search_content, 3, 0.5)
    # assert len(result) > 0
    return result
def test_delete_doc():
    assert kbService.delete_doc(testKnowledgeFile)


def test_update_doc():
    assert kbService.update_doc()


def test_delete_db(kbService):
    assert kbService.drop_kb()

def test_clear_emb():
    emb_path = f"knowledge_base/{test_kb_name}/vector_store"
    if os.path.exists(emb_path):
        shutil.rmtree(emb_path)

import json
with open("tests/kb_vector_db/query.json", "r") as f:
    queries = json.load(f)
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
}
# queries = {
#     "WP系列柴油机故障闪码指示灯诊断故障方法":31
#     # "潍柴国V潍柴国V进油进油和回油管回油管的安装":29
# }

d = list()


# faiss的swigfaiss的indexflat.search是向量的L2距离和余弦距离
test_kb_name = "test"
kbService = FaissKBService(test_kb_name)
# test_delete_db(kbService)
# test_create_db(kbService)
test_clear_emb()
test_add_doc(kbService, testKnowledgeFile)

retrival_list = test_search_db(kbService)
accurate = 0
for query, index in queries.items(): 
    answer_list = []
    search_content = query
    answers = test_search_db(kbService)
    print(f"分数排序: {[answer[1] for answer in answers]}")
    # break
    for i in range(len(answers)):
        if answers:
            answer_list.append([i, answers[i][0].metadata, answers[i][0].page_content[:100]])
    if answers:
        length = len(answers[0][0].page_content)
        if answers[0][0].metadata['content_pos'][0]['page_no'] <= index <= answers[0][0].metadata['content_pos'][-1]['page_no']:
            accurate += 1
        else:
            print(f"label: {index}, 回答区间: [{answers[0][0].metadata['content_pos'][0]['page_no']},{answers[0][0].metadata['content_pos'][-1]['page_no']}]")
            # print("-----------------------------------------------------------------------------------")
            # print(f"问题页面：{index}, 回答区间[{answers[0][0].metadata['content_pos'][0]['page_no']}, {answers[0][0].metadata['content_pos'][-1]['page_no']}], \
            #       问题：{query}")
            # print(f"得分：{answers[0][1]}")
            # print(f"key word：{answers[0][0].metadata}")
            # print(f"答案节选：{repr(answers[0][0].page_content)[:50]}")
        d.append({"page":index, 
              "query": query,
              "answer":answer_list,
              "len":length})
accuracy = accurate / len(queries)    
print(f"accuracy: {accuracy}")          
with open("tests/kb_vector_db/answer.json", "w", encoding='utf-8') as f:
    json.dump(d, f, ensure_ascii=False, indent=4)
# retrival_list[i][1]是相似度分数, retrival_list[i][0].page_content是文本, retrival_list[i][0].metadata['source']是source

print("=============================================================================================")

# test_kb_name = "test1"
# kbService = ESKBService(test_kb_name)
# test_create_db(kbService)
# test_add_doc(kbService, testKnowledgeFile)
# print(test_search_db(kbService))
# print("=============================================================================================")
# from server.db.base import Base, engine
# Base.metadata.create_all(bind=engine)
# test_kb_name = "test2"
# kbService = ZillizKBService(test_kb_name)
# test_create_db(kbService)
# test_add_doc(kbService, testKnowledgeFile)
# print(test_search_db(kbService))
# print("=============================================================================================")

# test_kb_name = "test3"
# kbService = PGKBService(test_kb_name)
# test_create_db(kbService)
# test_add_doc(kbService, testKnowledgeFile)
# print(test_search_db(kbService)) # 针对PostgreSQL
# print("=============================================================================================")

# test_kb_name = "test4"
# kbService = MilvusKBService(test_kb_name)
# test_create_db(kbService)
# test_add_doc(kbService, testKnowledgeFile)
# print(test_search_db(kbService)) # 大规模数据库，扩展性强
# print("=============================================================================================")
