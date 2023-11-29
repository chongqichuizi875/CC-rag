# from .tests.api.test_kb_api import *
# from webui_pages.utils import ApiRequest
# api_base_url = api_address()
# api: ApiRequest = ApiRequest(api_base_url)


# kb = "kb_for_api_test"
# test_files = {
#     "FAQ.MD": str(root_path / "docs" / "FAQ.MD"),
#     "README.MD": str(root_path / "README.MD"),
#     "test.txt": get_file_path("samples", "test.txt"),
# }

# test_create_kb()

# import json
# ref_documents=[]
# # print(f"docs: {docs}")
# for inum in range(2):
#     filename = "我是中文.pdf"
#     ref_documents.append({"filename":filename,"score":0.9,"content_pos_1":{"page_no":1,"x":0,"y":0.5},"content_pos_2":{"page_no":1,"x":1,"y":0.8}})
# s = json.dumps({"ref_docs": ref_documents}, ensure_ascii=False)
# print(s)
# print(type(s))


import pandas as pd
import pdfplumber
with pdfplumber.open("knowledge_base/lb_test/content/陕汽-重卡X5000维修手册（第一部分）.pdf") as pdf:
    
    page = pdf.pages[10]   # 第一页的信息
    #1、读文件
    lines = page.extract_text_lines()
    tables = page.find_tables()
    content = []
    for page in pages:
        page_content = []
        for index,line in enumerate(lines):
            for table in table:
                if table_top_line(table,line):
                    content.append({{table.extract(),"table","pos"}})
            if line_not_in_table(line):
                content.append({line,"txt","pos"})
            
            pass
        content.append[page_content]
    #2、合并软回车
    for page_content in content:
        for line in page_content:
            pass
    #3、按“行”读取，判断层级，分章节；提取关键字
    
    #4、遍历所有内容，对部分章节中内容超过1K的，分成len()/1k
    
    #5、插入向量库中
        
    
        
            
            

    # print("______________________________________________________")
    # page.extract_text()
    # page.extract_tables()
    tables = page.find_tables()
    for table in tables:
        # 得到的table是嵌套list类型，转化成DataFrame更加方便查看和分析
        # df = pd.DataFrame(t[1:], columns=t[0])
        # print(df)
        # print("______________________________________________________")
        # print(table.bbox,)
        # print(table.extract())
        print(repr(table))
