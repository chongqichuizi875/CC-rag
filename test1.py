import re
pattern = r'^\d+\.\d+\.\d+[^\.]\s*'
text1 = "第1章 传动机构"
text2 = "3.1悬架"
text3 = "3.1.1 悬架结构与原理"
text4 = "3.1.1.1 前悬架"

text = [text1, text2, text3, text4]
for t in text:
    print(re.search(pattern, t))