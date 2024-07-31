import pickle
import numpy as np
import pandas as pd

fpath = "C:/Users/USER/Desktop/fb15k-237.pkl"
wpath = "C:/Users/USER/Desktop/wn18rr.pkl"

fdata, wdata = [], []
with open(fpath, 'rb') as file:
    fdata = pickle.load(file)

with open(wpath, 'rb') as file:
    wdata = pickle.load(file)

fb15k = []
wn18rr = []
for i in range(len(fdata)):
    fb15k.append((i, fdata[i]))

for j in range(len(wdata)):
    wn18rr.append((j, wdata[j]))

rank_fb15k, rank_wn18rr = np.array(fb15k), np.array(wn18rr)

#%% example에서 가지고오기
import ast
import json

example_wn18rr = []
with open('C:/Users/USER/Desktop/wn18rr.txt', 'r') as file:
    for line in file:
        example_wn18rr.append(json.loads(line.strip()))

example_fb15k = []
with open('C:/Users/USER/Desktop/fb15k-237.txt', 'r') as file:
    for line in file:
        example_fb15k.append(json.loads(line.strip()))
        
key_to_extract = ['text_a', 'text_b', 'text_c','en_id','rel', 'real_label' ]
Z_fb, Z_wn = [], []
for index, d in enumerate(example_wn18rr):
    key1 = d.get('text_a', None)
    key2 = d.get('text_b', None)
    key3 = d.get('text_c', None)
    key4 = d.get('en_id', None)
    key5 = d.get('rel', None)
    key6 = d.get('real_label', None)
    Z_wn_tuple = (index, key1, key2, key3,key4, key5, key6)
    Z_wn.append(Z_wn_tuple)

for index, d in enumerate(example_fb15k):
    key1 = d.get('text_a', None)
    key2 = d.get('text_b', None)
    key3 = d.get('text_c', None)
    key4 = d.get('en_id', None)
    key5 = d.get('rel', None)
    key6 = d.get('real_label', None)
    Z_fb_tuple = (index, key1, key2, key3, key4, key5, key6)
    Z_fb.append(Z_fb_tuple)    
    
# Z_wn, Z_fb = np.array(Z_wn), np.array(Z_fb)
Z_wn = np.c_[Z_wn, wdata]
Z_fb = np.c_[Z_fb, fdata]

## 특정 열로 정렬(rank, 7번째 column)
sorted_by_column = 6
Z_wn_indices = np.argsort(Z_wn[:, sorted_by_column])
Z_fb_indices = np.argsort(Z_fb[:, sorted_by_column])

Z_wn = Z_wn[Z_wn_indices]
Z_fb = Z_fb[Z_fb_indices]

entity_wn = []
entity_fb = []
for i in range(len(Z_wn)):
    if Z_wn[i][1] == "[PAD]":
        entity_wn.append((Z_wn[i][0], Z_wn[i][5], Z_wn[i][6], Z_wn[i][4])) # (index, real_label, rank, (h,r))
entity_wn = np.array(entity_wn)        
tmp_wn = entity_wn[:,3] # (h,r)

for i in range(len(Z_fb)):
  if Z_fb[i][1] == "[PAD]":
        entity_fb.append((Z_fb[i][0], Z_fb[i][5], Z_fb[i][6], Z_fb[i][4])) # (index, real_label, rank, (h,r))
entity_fb = np.array(entity_fb)        
tmp_fb = entity_fb[:,3] # (h,r)

#### # (h,r)을 분리하기 h, r로
en_wn = [] 
for i in range(len(entity_wn)):
    tmp = np.array(tmp_wn[i])
    en_wn.append(tmp)
en_wn = np.array(en_wn)

en_fb = []
for i in range(len(entity_fb)):
    tmp = np.array(tmp_fb[i])
    en_fb.append(tmp)
en_fb = np.array(en_fb)
# %% 
for i in range(len(en_fb)):
    en_fb[i][1] = en_fb[i][1] - 14952
for i in range(len(en_wn)):
    en_wn[i][1] = en_wn[i][1] - 40943 
    
