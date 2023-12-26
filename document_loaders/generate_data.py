import json
import re
import random
with open("document_loaders/111.json", 'r') as f:
    data = json.load(f)

with open("document_loaders/queries.json", 'r') as f:
    queries = json.load(f)

reverse_dict = {}

for content in data:
    match = re.search(r'\[\[', content[2])
    context = content[2][:match.start()] if match else content[2]
    images = ' '.join(content[1]['images'])
    tables = ' '.join(content[1]['tables'])
    reverse_dict[content[1]['global_index']] = content[1]['titles']+images+tables+context
n = len(data)
finetune_data = []
for query, global_index in queries.items():
    one_data = {}
    one_data['query'] = query
    one_data['pos'] = [reverse_dict[global_index]]
    one_data['neg'] = []
    for i in range(20):
        select_neg = data[random.randint(0,n-1)]
        if select_neg[1]['global_index'] != global_index:
            one_data['neg'].append(reverse_dict[select_neg[1]['global_index']])
    finetune_data.append(one_data)
with open("document_loaders/finetune_emb_data.jsonl", 'w', encoding='utf-8') as f:
    for d in finetune_data:
        json_str = json.dumps(d, ensure_ascii=False)
        f.write(json_str+'\n')