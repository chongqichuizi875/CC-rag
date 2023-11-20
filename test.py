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

import json
ref_documents=[]
# print(f"docs: {docs}")
for inum in range(2):
    filename = "我是中文.pdf"
    ref_documents.append({"filename":filename,"score":0.9,"content_pos_1":{"page_no":1,"x":0,"y":0.5},"content_pos_2":{"page_no":1,"x":1,"y":0.8}})
s = json.dumps({"ref_docs": ref_documents}, ensure_ascii=False)
print(s)
print(type(s))

    