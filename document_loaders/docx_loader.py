import docx
from typing import List
from langchain.document_loaders.unstructured import UnstructuredFileLoader
import os
import docx
from docx.document import Document
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import _Cell, Table
from docx.text.paragraph import Paragraph

from docx.image import Bmp,Png
# ,CT_Picture
from docx.image.image import Image


def iter_block_items(parent):
    """
    Yield each paragraph and table child within *parent*, in document order.
    Each returned value is an instance of either Table or Paragraph. *parent*
    would most commonly be a reference to a main Document object, but
    also works for a _Cell object, which itself can contain paragraphs and tables.
    """
    if isinstance(parent, Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("something's not right")

    for child in parent_elm.iterchildren():
        print(">>>>type",type(child))
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)
        # elif isinstance(child,CT_Picture):
        #     yield Image(child,parent)
        else:
            print(">>>>type",type(child))
            pass

def read_table(table):
    return [[cell.text for cell in row.cells] for row in table.rows]


def read_word(word_path):
    doc = docx.Document(word_path)
    for block_index,block in enumerate(iter_block_items(doc)):
        if isinstance(block, Paragraph):
            print("text", [block.text],block)
        elif isinstance(block, Table):
            print("table", read_table(block),block.style)
        elif isinstance(block,Image):
            print("picture")
        if block_index ==50:
            break



class CCDocxLoader(UnstructuredFileLoader):
    def _get_elements(self) -> List:
        read_word(self.file_path)
        return ""
            

if __name__ == "__main__":
    loader = UnstructuredFileLoader(file_path="tests/test_word.docx")
    eles = loader._get_elements()
    for ele_index,els in enumerate(eles):
        print(ele_index,els,type(els))
        if ele_index >9:
            break