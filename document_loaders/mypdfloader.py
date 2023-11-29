import sys
sys.path.append("./")
from typing import Any, List, Union
from langchain.document_loaders.unstructured import UnstructuredFileLoader
import tqdm
import re
import json
from langchain.docstore.document import Document
from text_splitter.chinese_chapter_recursive_splitter import ChineseChapterRecursiveSplitter
from configs.kb_config import CHUNK_SIZE, OVERLAP_SIZE
from copy import deepcopy
import os

key_word_dict = {
    "维修":"维修",
    "故障":"故障",
    "参数":"参数",
    "属性":"属性",
    "操作":"操作",
    "使用":"使用",
    "步骤":"步骤"
}

def in_table_index(location, tables):
    if not tables:
        return -1
    x0, y0, x1, y1 = location
    boundings = [table.bbox for table in tables]
    for index, bounding in enumerate(boundings):
        b_x0, b_y0, b_x1, b_y1 = bounding
        if b_x0 < x0 and b_y0 < y0 and b_x1 > x1 and b_y1 > y1:
            return index
    return -1


def remove_special_chars(text:str) -> str:
    special_chars = r"[-.]{2,}|[■…]"
    text = re.sub(special_chars, "", text)
    return text

def potential_title_pos(text):
    if len(text) == 0:
        return -1
    index_of_newline = text.find('\n')
    return index_of_newline

def create_documents(chapter, title_stack, title_prefix, metadata: dict) -> List[Document]:
    prefix = ""
    for i, sub_title in enumerate(title_stack):
        if sub_title:
            prefix += title_prefix*i+sub_title
    prefix = re.sub(r'\s+', '', prefix)
    metadata['titles'] = prefix
    metadata['image_and_table'] = []
    metadata['keyword'] = []
    
    chapter = post_split(text=chapter)
    pattern = r"\b[图|表]\s+\d+-\d+\s+[^，。；！？：“”‘’（）《》&#8203;``【oaicite:0】``&#8203;、\n]*\n" # 提取图和表
    matches = re.findall(pattern, chapter)
    if matches:
        for match in matches:
            metadata['image_and_table'].append(match.strip())
    for key, value in key_word_dict.items():
        if re.findall(key, chapter):
            metadata['keyword'].append(value)
    
    if len(chapter) < OVERLAP_SIZE:
        return []
    # if len(chapter) > CHUNK_SIZE:
    #     chunks = [chapter[:CHUNK_SIZE]]
    #     chunks.extend([chapter[i-OVERLAP_SIZE:i+CHUNK_SIZE] for i in range(CHUNK_SIZE, len(chapter), CHUNK_SIZE)])
    #     docs = [Document(page_content=chunk, metadata=deepcopy(metadata)) for chunk in chunks]
    #     return docs
    docs = [Document(page_content=chapter, metadata=deepcopy(metadata))]
    metadata['image_and_table'] = []
    return docs

def post_split(text):
    """
    把标题中的空格和换行去掉
    把文中多余的空格/换行替换成单一空格/换行    
    """
    # text = text.replace('\n', title_split, 1)
    # title_pos = text.find(title_split)
    # if title_pos != -1:
    #     text = re.sub(r'\s+', '', text[:title_pos]) + text[title_pos:]
    text = remove_special_chars(text)
    text = re.sub(r'\<image:.*?\>', '', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r' ?\n ?', '\n', text)
    text = re.sub(r'\n{2,}', '\n', text)
    text = remove_newlines_between_chinese(text)
    return text

def remove_newlines_between_chinese(text):
    # 正则表达式，匹配两个汉字之间的换行符，但是如果其中一个汉字是'图'或者'表'则不管
    # pattern = re.compile(r'((?![图表])[\u4e00-\u9fa5])(\s+)((?![图表])[\u4e00-\u9fa5])')
    pattern = re.compile(r'(?<=[^\W\d图表])\s+(?=[^\W\d图表])')
    return pattern.sub(r'', text).strip()
    return pattern.sub(r'\1\2', text).strip()

class MrjOCRPDFLoader(UnstructuredFileLoader):
    def __init__(self, file_path: str or List[str], mode: str = "paged", **unstructured_kwargs: Any):
        super().__init__(file_path, mode, **unstructured_kwargs)
        self.text_splitter = ChineseChapterRecursiveSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE)
    def _get_elements(self) -> List:
        # 流式匹配，手动维护标题栈
        def pdf2text(filepath):
            import pdfplumber
            pages = pdfplumber.open(filepath).pages
            b_unit = tqdm.tqdm(total=len(pages), desc="MrjOCRPDFLoader context page index: 0")
            resp = []
            chapter = ""
            prev_chapter = chapter
            metadata = dict()
            metadata['content_pos'] = []
            loc_dict = {}
            loc_dict['page_no'] = 0
            # 初始化开始和结束的x坐标，为了求一段中开始(结束)值的最值
            loc_dict['left_top'] = {
                'x':1,
                'y':1
            }
            loc_dict['right_bottom'] = {
                'x':0,
                'y':0
            }
            title_stack = []
            title_level_list = self.text_splitter.get_seperators()
            txt, x0, y0, x1, y1 = '', 0.1, 0.1, 0.9, 0.9
            for page_num in range(len(pages)):
                b_unit.set_description("MrjOCRPDFLoader context page index: {}".format(page_num))
                b_unit.refresh()
                page = pages[page_num]
                page_width = page.width
                page_height = page.height
                tables = page.find_tables() # 如果没有tables就是[]
                added_tables = []
                for line_num, line in enumerate(page.extract_text_lines()):
                    matched = False
                    prev_txt, prev_x0, prev_y0, prev_x1, prev_y1 = txt, x0, y0, x1, y1
                    prev_page_num = page_num if line_num else page_num - 1
                    txt, ori_x0, ori_y0, ori_x1, ori_y1 = line["text"],line["x0"],line["top"],line["x1"],line["bottom"]
                    if txt.isdigit():
                        txt, x0, y0, x1, y1 = prev_txt, prev_x0, prev_y0, prev_x1, prev_y1
                        continue
                    table_index = in_table_index((ori_x0, ori_y0, ori_x1, ori_y1), tables)
                    x0 = ori_x0/page_width
                    x1 = ori_x1/page_width
                    y0 = ori_y0/page_height
                    y1 = ori_y1/page_height
                    if table_index != -1:
                        if table_index in added_tables:
                            continue

                        added_tables.append(table_index)
                        chapter += ' ' + str(tables[table_index].extract())
                        loc_dict['page_no'], loc_dict['right_bottom']['x'], loc_dict['right_bottom']['y'] = \
                            page_num+1,tables[table_index].bbox[2]/page_width,tables[table_index].bbox[3]/page_height
                        loc_dict['left_top']['x'], loc_dict['left_top']['y'] = \
                                tables[table_index].bbox[0]/page_width,tables[table_index].bbox[1]/page_height
                        if chapter[:3] == ' [[':
                            chapter = prev_chapter + chapter
                            resp.pop()
                        metadata['content_pos'].append(deepcopy(loc_dict))
                        docs = create_documents(chapter=chapter, 
                                            title_stack=title_stack, 
                                            title_prefix=self.text_splitter.title_prefix,
                                            metadata=metadata)
                        resp.extend(docs)
                        prev_chapter = chapter
                        chapter = ""
                        metadata['content_pos'] = []
                        loc_dict['page_no'], loc_dict['left_top']['x'], loc_dict['left_top']['y'], loc_dict['right_bottom']['x'], loc_dict['right_bottom']['y'] = \
                        page_num+1,tables[table_index].bbox[0]/page_width,tables[table_index].bbox[1]/page_height,tables[table_index].bbox[2]/page_width,tables[table_index].bbox[3]/page_height
                        continue
                    loc_dict['left_top']['x'], loc_dict['left_top']['y'] = \
                        min(x0, loc_dict['left_top']['x']), min(y0, loc_dict['left_top']['y'])
                    for title_level, title in enumerate(title_level_list):
                        match = re.search(pattern=title, string=txt)
                        if match:
                            # 手动维护一个标题栈
                            loc_dict['page_no'], loc_dict['right_bottom']['x'], loc_dict['right_bottom']['y'] = \
                                prev_page_num+1,max(prev_x1, loc_dict['right_bottom']['x']),prev_y1
                            metadata['content_pos'].append(deepcopy(loc_dict))
                            docs = create_documents(chapter=chapter, 
                                                title_stack=title_stack, 
                                                title_prefix=self.text_splitter.title_prefix,
                                                metadata=metadata)
                            resp.extend(docs)
                            prev_chapter = chapter
                            chapter = ""
                            metadata['content_pos'] = []
                            loc_dict['page_no'], loc_dict['left_top']['x'], loc_dict['left_top']['y'], loc_dict['right_bottom']['x'], loc_dict['right_bottom']['y'] = \
                                page_num+1,x0,y0,x1,y1 
                            # 检索到更高级/平级标题，把当前titles拼接加上chapter内容append到list
                            while len(title_stack) > title_level: # 把多余title pop掉
                                title_stack.pop()
                            while len(title_stack) < title_level: # 补齐title_stack到低一级
                                title_stack.append('')
                            cur_line = txt[match.end():]
                            title_pos = potential_title_pos(cur_line)
                            title_stack.append(cur_line)
                            # chapter += "\n" + cur_line[title_pos:]
                            matched = True
                            break
                    if not matched:
                        if len(chapter) < CHUNK_SIZE:
                            # 如果chunk size不满，继续增大end y
                            # 因为只有这一种可能会在page结尾append坐标信息，page结尾append的坐标信息必须是
                            # 当前页面左上和右下的坐标，来标记这一整页都被包含
                            loc_dict['page_no'], loc_dict['right_bottom']['x'], loc_dict['right_bottom']['y'] = \
                                page_num+1,max(x1, loc_dict['right_bottom']['x']), max(y1, loc_dict['right_bottom']['y']) 
                            chapter += post_split(text=txt)
                        else:
                            loc_dict['page_no'], loc_dict['right_bottom']['x'], loc_dict['right_bottom']['y'] = \
                                prev_page_num+1,max(prev_x1, loc_dict['right_bottom']['x']), prev_y1 # chunk size到了要截断，则end y必须是当前位置的y
                            metadata['content_pos'].append(deepcopy(loc_dict))
                            docs = create_documents(chapter=chapter, 
                                                    title_stack=title_stack, 
                                                    title_prefix=self.text_splitter.title_prefix,
                                                    metadata=metadata)
                            resp.extend(docs)
                            prev_chapter = chapter
                            chapter = post_split(text=txt)
                            metadata['content_pos'] = []
                            loc_dict['page_no'], loc_dict['left_top']['x'], loc_dict['left_top']['y'], loc_dict['right_bottom']['x'], loc_dict['right_bottom']['y'] = \
                                page_num+1,x0,y0,x1,y1

                metadata['content_pos'].append(deepcopy(loc_dict))
                b_unit.update(1)
            
            return resp
        if self.file_path[-5:] == ".docx":
            from server.knowledge_base.word2pdf import doc2pdf
            doc2pdf(self.file_path)
            self.file_path = self.file_path[:-5] + ".pdf"
            
        text = pdf2text(self.file_path)
        for chapter in text:
            chapter.metadata['source'] = self.file_path
        new_next = []
        for sub_text in text:
            new_next.append([len(sub_text.page_content), sub_text.metadata, sub_text.page_content])    
        with open("document_loaders/111.json", 'w') as f:
            json.dump(new_next, f, ensure_ascii=False, indent=4)
        
        return text

    def load(self) -> List[Document]:
        docs = self._get_elements()
        return docs



    def pre_get_elements(self) -> List:
        def pdf2text(filepath):
            import fitz
            doc = fitz.open(filepath)
            b_unit = tqdm.tqdm(total=doc.page_count, desc="MrjOCRPDFLoader context page index: 0")
            resp = []
            for i in range(len(doc)):
                b_unit.set_description("MrjOCRPDFLoader context page index: {}".format(i))
                b_unit.refresh()
                page = doc.load_page(i)
                page_text = page.get_text()
                resp.append(remove_special_chars(text=page_text)) 
                b_unit.update(1)
            return resp
        text = pdf2text(self.file_path)
        for chapter in text:
            chapter.metadata['source'] = self.file_path
        return text

class RapidOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        def pdf2text(filepath):
            import fitz # pyMuPDF里面的fitz包，不要与pip install fitz混淆
            from rapidocr_onnxruntime import RapidOCR
            import numpy as np
            ocr = RapidOCR()
            doc = fitz.open(filepath)
            resp = ""

            b_unit = tqdm.tqdm(total=doc.page_count, desc="RapidOCRPDFLoader context page index: 0")
            for i, page in enumerate(doc):

                # 更新描述
                b_unit.set_description("RapidOCRPDFLoader context page index: {}".format(i))
                # 立即显示进度条更新结果
                b_unit.refresh()
                # TODO: 依据文本与图片顺序调整处理方式
                text = page.get_text("")
                resp += text + "\n"

                img_list = page.get_images()
                for img in img_list:
                    pix = fitz.Pixmap(doc, img[0])
                    img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, -1)
                    result, _ = ocr(img_array)
                    if result:
                        ocr_result = [line[1] for line in result]
                        resp += "\n".join(ocr_result)

                # 更新进度
                b_unit.update(1)
            return resp

        text = pdf2text(self.file_path)
        from unstructured.partition.text import partition_text
        return partition_text(text=text, **self.unstructured_kwargs)


if __name__ == "__main__":
    loader = MrjOCRPDFLoader(file_path="/home/cc007/cc/Langchain-Chatchat/knowledge_base/test/content/陕汽-重卡X5000维修手册（第二部分）.pdf")
    docs = loader.load()
    print(docs)
