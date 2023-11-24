import subprocess
import os
# from configs import logger


def doc2pdf_linux(doc):
    """
    convert a doc/docx document to pdf format (linux only, requires libreoffice)
    :param doc: path to document
    """
    pdf_dir = doc[:str(doc).rfind("/")]
    pdf_dir = pdf_dir[:str(pdf_dir).rfind("/")]+"/"+"content"
    print("pdf_dir:",pdf_dir)
    cmd = 'libreoffice --convert-to pdf'.split() + [doc] +["--outdir",pdf_dir]
    print(type(cmd),cmd)
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    p.wait(timeout=100)
    stdout, stderr = p.communicate()
    if stderr:
        raise subprocess.SubprocessError(stderr)

def doc2pdf(doc):
    """
    convert a doc/docx document to pdf format
    :param doc: path to document
    """
    doc = os.path.abspath(doc) # bugfix - searching files in windows/system32
    
    try:
        from comtypes import client
    except ImportError as e:
        client = None
        # logger.exception(e)
    
    if client is None:
        return doc2pdf_linux(doc)
    name, ext = os.path.splitext(doc)
    try:
        word = client.CreateObject('Word.Application')
        worddoc = word.Documents.Open(doc)
        worddoc.SaveAs(name + '.pdf', FileFormat=17)
    except Exception:
        raise
    finally:
        worddoc.Close()
        word.Quit()



if __name__ == "__main__":
    doc2pdf("knowledge_base/lb_test/content/陕汽L3000系列载货车维修手册（第一部分）.docx")