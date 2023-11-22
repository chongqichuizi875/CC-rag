import sys
sys.path.append("/home/cc007/cc/chat_doc")
from typing import Any, List, Union
from langchain.document_loaders.unstructured import UnstructuredFileLoader
import tqdm
import re
import json
from langchain.docstore.document import Document
from text_splitter.chinese_chapter_recursive_splitter import ChineseChapterRecursiveSplitter
from configs.kb_config import CHUNK_SIZE, OVERLAP_SIZE
from copy import deepcopy

def remove_special_chars(text:str) -> str:
    special_chars = '…■.-'
    text = re.sub(f"[{re.escape(special_chars)}]", "", text)
    return text

def create_documents(chapter, title_stack, title_prefix, metadata: dict) -> List[Document]:
    prefix = ""
    for i, sub_title in enumerate(title_stack):
        if sub_title:
            prefix += title_prefix*i+sub_title
    prefix = re.sub(r'\s+', '', prefix)
    metadata['titles'] = prefix
    chapter = post_split(text=chapter)
    if len(chapter) < OVERLAP_SIZE:
        return []
    # if len(chapter) > CHUNK_SIZE:
    #     chunks = [chapter[:CHUNK_SIZE]]
    #     chunks.extend([chapter[i-OVERLAP_SIZE:i+CHUNK_SIZE] for i in range(CHUNK_SIZE, len(chapter), CHUNK_SIZE)])
    #     docs = [Document(page_content=chunk, metadata=deepcopy(metadata)) for chunk in chunks]
    #     return docs
    docs = [Document(page_content=chapter, metadata=deepcopy(metadata))]
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
    return text

class MrjOCRPDFLoader(UnstructuredFileLoader):
    def __init__(self, file_path: str or List[str], mode: str = "paged", **unstructured_kwargs: Any):
        super().__init__(file_path, mode, **unstructured_kwargs)
        self.text_splitter = ChineseChapterRecursiveSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=OVERLAP_SIZE)
    def _get_elements(self) -> List:
        # 流式匹配，手动维护标题栈
        def pdf2text(filepath):
            import fitz
            doc = fitz.open(filepath)
            b_unit = tqdm.tqdm(total=doc.page_count, desc="MrjOCRPDFLoader context page index: 0")
            resp = []
            chapter = ""
            metadata = dict()
            metadata['b_page'], metadata['b_x'], metadata['b_y'] = 0, 0, 0
            title_stack = []
            title_level_list = self.text_splitter.get_seperators()
            for page_num in range(len(doc)):
                b_unit.set_description("MrjOCRPDFLoader context page index: {}".format(page_num))
                b_unit.refresh()
                page = doc.load_page(page_num)
                page_width = page.rect.width
                page_height = page.rect.height
                for block in page.get_text("blocks"):
                    matched = False
                    x0, y0, x1, y1, txt = block[:5]
                    x0 /= page_width
                    x1 /= page_width
                    y0 /= page_height
                    y1 /= page_height
                    for title_level, title in enumerate(title_level_list):
                        match = re.search(pattern=title, string=txt)
                        if match:
                            # 手动维护一个标题栈
                            if len(title_stack) > title_level: 
                                # 检索到更高级/平级标题，把当前titles拼接加上chapter内容append到list
                                metadata['e_page'], metadata['e_x'], metadata['e_y'] = page_num+1,x1,y1
                                docs = create_documents(chapter=chapter, 
                                                        title_stack=title_stack, 
                                                        title_prefix=self.text_splitter.title_prefix,
                                                        metadata=metadata)
                                
                                resp.extend(docs)
                                # chapter = txt[match.end():]
                                chapter = ""
                                metadata['b_page'], metadata['b_x'], metadata['b_y'] = page_num+1,x0,y0
                                while len(title_stack) > title_level: # 把多余title pop掉
                                    title_stack.pop()
                            else: # 检索到低级标题，查看chapter是否有内容，有的话加上当前title，append到list
                                if chapter.strip():
                                    metadata['e_page'], metadata['e_x'], metadata['e_y'] = page_num+1,x1,y1
                                    docs = create_documents(chapter=chapter, 
                                                        title_stack=title_stack, 
                                                        title_prefix=self.text_splitter.title_prefix,
                                                        metadata=metadata)
                                    # chapter_doc.metadata = metadata
                                    resp.extend(docs)
                                    chapter = ""
                                    metadata['b_page'], metadata['b_x'], metadata['b_y'] = page_num+1,x0,y0
                                # chapter += txt[match.end():]
                                while len(title_stack) < title_level: # 补齐title_stack到低一级
                                    title_stack.append('')
                            title_stack.append(txt[match.end():])
                            matched = True
                            break
                    if not matched:
                        if len(chapter) < CHUNK_SIZE:
                            chapter += post_split(text=txt)
                        else:
                            metadata['e_page'], metadata['e_x'], metadata['e_y'] = page_num+1,x1,y1
                            docs = create_documents(chapter=chapter, 
                                                    title_stack=title_stack, 
                                                    title_prefix=self.text_splitter.title_prefix,
                                                    metadata=metadata)
                            resp.extend(docs)
                            chapter = post_split(text=txt)
                            metadata['b_page'], metadata['b_x'], metadata['b_y'] = page_num+1,x0,y0

                    
                b_unit.update(1)
            return resp
        text = pdf2text(self.file_path)
        new_next = []
        for sub_text in text:
            new_next.append([sub_text.metadata,sub_text.page_content])    
        # with open("/home/cc007/cc/chat_doc/document_loaders/111.json", 'w') as f:
        #     json.dump(new_next, f, ensure_ascii=False, indent=4)
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
    loader = MrjOCRPDFLoader(file_path="/home/cc007/cc/Langchain-Chatchat/knowledge_base/test/content/陕汽-新M3000S维修手册 第二部分.pdf")
    docs = loader.load()
    # print(docs)
