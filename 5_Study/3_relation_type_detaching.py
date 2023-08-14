'''
for SimKGC, you need to use json file.
for Relphormer, you need to use txt file. 
​
'''
​
​
​
import json
​
def read_txt(path):
    file = open(path, 'r')
    lines = file.readlines()
    file.close()
    return lines
​
def read_json(path):
    json_file = open(path, 'r', encoding="utf-8")
    file = json.load(json_file)
    return file
​
​
​
wn_train_lines = read_txt("model/SimKGC/data/WN18RR/train.txt") #change path
wn_test_lines = read_txt("model/SimKGC/data/WN18RR/test.txt") 
wn_valid_lines = read_txt("model/SimKGC/data/WN18RR/valid.txt")
​
wn_txt_examples = wn_train_lines + wn_test_lines +  wn_valid_lines
​
​
​
fb_train_lines = read_txt("model/SimKGC/data/FB15k237/train.txt")
fb_test_lines = read_txt("model/SimKGC/data/FB15k237/test.txt")
fb_valid_lines = read_txt("model/SimKGC/data/FB15k237/valid.txt")
​
fb_txt_examples = fb_train_lines + fb_test_lines + fb_valid_lines
​
​
​
wn_train = read_json('model/SimKGC/data/WN18RR/train.txt.json')
wn_test = read_json('model/SimKGC/data/WN18RR/test.txt.json')
wn_valid = read_json('model/SimKGC/data/WN18RR/valid.txt.json')
​
wn_json_examples = wn_train + wn_valid + wn_test
​
​
​
fb_train = read_json('model/SimKGC/data/FB15k237/train.txt.json')
fb_test = read_json('model/SimKGC/data/FB15k237/test.txt.json')
fb_valid = read_json('model/SimKGC/data/FB15k237/valid.txt.json')
​
fb_json_examples = fb_train + fb_valid + fb_test
​
​
​
def read_txt_examples(examples):
​
    relh2tail = {}  
    relt2head = {}
​
    #check how many tail per (relation, head)
    for ex in examples:
        ex = ex.strip().split('\t')
        head, relation, tail = ex[0], ex[1], ex[2]
​
        if relation in relh2tail:
            if head in relh2tail[relation]:
                relh2tail[relation][head].append(tail)
            else:
                relh2tail[relation][head] = [tail]
        else:
            relh2tail[relation] = {head: [tail]}
​
        #check how many head per (relation, tail)
        if relation in relt2head:
            if tail in relt2head[relation]:
                relt2head[relation][tail].append(tail)
            else:
                relt2head[relation][tail] = [head]
        else:
            relt2head[relation] = {tail: [head]}
​
    return relh2tail, relt2head
​
​
​
def read_json_examples(examples):
    relh2tail = {}
    relt2head = {}
    for ex in examples:
        head, relation, tail = ex['head'], ex['relation'], ex['tail']
​
        if relation in relh2tail:
            if head in relh2tail[relation]:
                relh2tail[relation][head].append(tail)
            else :
                relh2tail[relation][head] = [tail]
        else:
            relh2tail[relation] = {head: [tail]}
​
        if relation in relt2head:
            if tail in relt2head[relation]:
                relt2head[relation][tail].append(tail)
            else:
                relt2head[relation][tail] = [head]
        else:
            relt2head[relation] = {tail: [head]} 
            
    return relh2tail, relt2head
​
​
def average_length(relh2tail, relt2head) :
    count_per_relation_head = {}
    count_per_relation_tail = {}
​
    for relation, head_dict in relh2tail.items():
        for head, tail_list in head_dict.items():
            count_per_relation_head[(relation, head)] = len(tail_list)
​
    for relation, tail_dict in relt2head.items():
        for tail, head_list in tail_dict.items():
            count_per_relation_tail[(relation, tail)] = len(head_list)
​
    average_tail_length  = {}  
    average_head_length = {}
​
    for relation, head_dict in relh2tail.items():
        tail_lengths = [len(tail_list) for tail_list in head_dict.values()]
        average_tail_length[relation]  = round(sum(tail_lengths) / len(tail_lengths),5)
​
    for relation, tail_dict in relt2head.items():
        head_lengths = [len(head_list) for head_list in tail_dict.values()]
        average_head_length[relation] = round(sum(head_lengths) / len(head_lengths),5)
​
    return average_tail_length, average_head_length
​
​
​
def make_label(average_tail_length, average_head_length):
    t_label = {}
    h_label = {}
​
    for k, v in average_tail_length.items():
        if v > 1.5:
            t_label[k] = 'N'
        else:
            t_label[k] = '1'
​
    for k, v in average_head_length.items():
        if v > 1.5:
            h_label[k] = 'N'
        else:
            h_label[k] = '1'
​
    relation_label = {}
​
    for relation in set(h_label.keys()).union(t_label.keys()):
        h_val = h_label.get(relation, '')
        t_val = t_label.get(relation, '')
        relation_label[relation] = f'{h_val}-{t_val}'
​
    return relation_label
​
​
​
​
wn_txt_relh2tail, wn_txt_relt2head = read_txt_examples(wn_txt_examples)
fb_txt_relh2tail, fb_txt_relt2head = read_txt_examples(fb_txt_examples)
wn_json_relh2tail, wn_json_relt2head = read_json_examples(wn_json_examples)
fb_json_relh2tail, fb_json_relt2head = read_json_examples(fb_json_examples)
​
wn_txt_average_tail_length, wn_txt_average_head_length =  average_length(wn_txt_relh2tail,wn_txt_relt2head)
fb_txt_average_tail_length, fb_txt_average_head_length = average_length(fb_txt_relh2tail, fb_txt_relt2head)
wn_json_average_tail_length, wn_json_average_head_length = average_length(wn_json_relh2tail, wn_json_relt2head)
fb_json_average_tail_length, fb_json_average_head_length = average_length(fb_json_relh2tail, fb_json_relt2head)
​
wn_txt_relation_label = make_label(wn_txt_average_tail_length, wn_txt_average_head_length) 
fb_txt_relation_label = make_label(fb_txt_average_tail_length, fb_txt_average_head_length)
wn_json_relation_label = make_label(wn_json_average_tail_length, wn_json_average_head_length)
fb_json_relation_label = make_label(fb_json_average_tail_length, fb_json_average_head_length)
​
​
relation_label_path1 = "model/Relphormer/dataset/wn18rr/relation_label.json"
relation_label_path2 = "model/Relphormer/dataset/fb15k-237/relation_label.json"
relation_label_path3 = "model/SimKGC/data/WN18RR/relation_label.json"
relation_label_path4 = "model/SimKGC/data/FB15k237/relation_label.json"
​
​
with open(relation_label_path1,'w') as json_file:
    json.dump(wn_txt_relation_label, json_file, indent=4)
​
with open(relation_label_path2, 'w') as json_file:
    json.dump(fb_txt_relation_label, json_file, indent=4)
​
with open(relation_label_path3, 'w') as json_file:
    json.dump(wn_json_relation_label, json_file, indent=4)
​
with open(relation_label_path4, 'w') as json_file:
    json.dump(fb_json_relation_label, json_file, indent=4)
​
