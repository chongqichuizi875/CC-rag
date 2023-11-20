from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader
import tqdm
import re

def remove_special_chars(page_num, text:str) -> str:
    special_chars = '…■.-'
    text = re.sub(f"[{re.escape(special_chars)}]", "", text)
    page_info = f"[{page_num}]"
    text += " " + page_info
    return text

class MrjOCRPDFLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
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
                resp.append(remove_special_chars(page_num=i+1, text=page_text)) 
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
    print(docs)
