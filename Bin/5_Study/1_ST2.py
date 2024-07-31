import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Step 1.Data load
"""
umls_1_hop = pd.read_csv("C:/Users/PC/Desktop/changed_data/umls/umls_1_hop.txt", sep = '\t', names = [0,1,2])
umls_2_hop = pd.read_csv("C:/Users/PC/Desktop/changed_data/umls/umls_2_hop.txt", sep = '\t', names = [0,1,2])
## numpy array
umls_1_hop = np.array(umls_1_hop)
umls_2_hop = np.array(umls_2_hop)
.

test_umls = pd.read_csv("C:/Users/PC/Desktop/dataset/umls/test.tsv", sep = '\s+', names = [0,1,2])
test2id = pd.read_csv("C:/Users/PC/Desktop/dataset/umls//get_neighbor/test2id.txt", sep =  "\s+", names=[0,1,2])
## numpy array
test_umls = np.array(test_umls)
test2id = np.array(test2id)
"""

"""
## wn18rr
wn_1_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/wn18rr/test2id/test2id_1.txt", sep = '\s+', names = [0,1,2])
wn_2_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/wn18rr/test2id/test2id_2.txt", sep = '\s+', names = [0,1,2])
wn_3_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/wn18rr/test2id/test2id_3.txt", sep = '\s+', names = [0,1,2])
wn_4_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/wn18rr/test2id/test2id_4.txt", sep = '\s+', names = [0,1,2])
wn_5_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/wn18rr/test2id/test2id_5.txt", sep = '\s+', names = [0,1,2])
wn_6_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/wn18rr/test2id/test2id_6.txt", sep = '\s+', names = [0,1,2])
wn_1_hop = np.array(wn_1_hop)
wn_2_hop = np.array(wn_2_hop)
wn_3_hop = np.array(wn_3_hop)
wn_4_hop = np.array(wn_4_hop)
wn_5_hop = np.array(wn_5_hop)
wn_6_hop = np.array(wn_6_hop)


test_wn = pd.read_csv("C:/Users/PC/Desktop/dataset/wn18rr/test.tsv", sep = '\s+', names = [0,1,2])
test2id = pd.read_csv("C:/Users/PC/Desktop/dataset/wn18rr//get_neighbor/test2id.txt", sep =  "\s+", names=[0,1,2])
test_wn = np.array(test_wn)
test2id = np.array(test2id)
"""
# """
## fb15k-237
fb_1_hop = pd.read_csv("C:/Users/PC/Desktop/fb_hop1.txt", sep = '\s+', names = [0,1,2])
fb_2_hop = pd.read_csv("C:/Users/PC/Desktop/fb_hop2.txt", sep = '\s+', names = [0,1,2])
fb_3_hop = pd.read_csv("C:/Users/PC/Desktop/fb_hop3.txt", sep = '\s+', names = [0,1,2])
fb_4_hop = pd.read_csv("C:/Users/PC/Desktop/fb_hop4.txt", sep = '\s+', names = [0,1,2])
fb_5_hop = pd.read_csv("C:/Users/PC/Desktop/fb_hop5.txt", sep = '\s+', names = [0,1,2])
fb_6_hop = pd.read_csv("C:/Users/PC/Desktop/fb_hop6.txt", sep = '\s+', names = [0,1,2])
fb_1_hop = np.array(fb_1_hop)
fb_2_hop = np.array(fb_2_hop)
fb_3_hop = np.array(fb_3_hop)
fb_4_hop = np.array(fb_4_hop)
fb_5_hop = np.array(fb_5_hop)
fb_6_hop = np.array(fb_6_hop)

test_fb = pd.read_csv("C:/Users/PC/Desktop/Banchmark/fb15k-237/test.tsv", sep = '\s+', names = [0,1,2])
test2id = pd.read_csv("C:/Users/PC/Desktop/Banchmark/fb15k-237/get_neighbor/test2id.txt", sep =  "\s+", names=[0,1,2])
test_fb = np.array(test_fb)
test2id = np.array(test2id)
# """
# Step 2.인덱스 분리
## UMLS
"""
## id 먼저 분리
index1 = []
for i in range(len(test2id)):
    for j in range(len(umls_1_hop)):
        if test2id[i][0] == umls_1_hop[j][0] and test2id[i][1] == umls_1_hop[j][1] and test2id[i][2] == umls_1_hop[j][2]:
            index1.append(i)


index = [i for i in range(661)]
index2 = []
for element in index1:
    if element in index1:
        index.remove(element)

index2 = index
"""     

"""
## WN18RR
index = [i for i in range(len(test2id))]
index1 = []
index2 = []
index3 = []
index4 = []
index5 = []
index6 = []

for i in range(len(test2id)):
    for j in range(len(wn_1_hop)):
        if test2id[i][0] == wn_1_hop[j][0] and test2id[i][1] == wn_1_hop[j][1] and test2id[i][2] == wn_1_hop[j][2]:
            index1.append(i)
    for j in range(len(wn_2_hop)):
        if test2id[i][0] == wn_2_hop[j][0] and test2id[i][1] == wn_2_hop[j][1] and test2id[i][2] == wn_2_hop[j][2]:
            index2.append(i)
    for j in range(len(wn_3_hop)):
        if test2id[i][0] == wn_3_hop[j][0] and test2id[i][1] == wn_3_hop[j][1] and test2id[i][2] == wn_3_hop[j][2]:
            index3.append(i)
    for j in range(len(wn_4_hop)):
        if test2id[i][0] == wn_4_hop[j][0] and test2id[i][1] == wn_4_hop[j][1] and test2id[i][2] == wn_4_hop[j][2]:
            index4.append(i)
    for j in range(len(wn_5_hop)):
        if test2id[i][0] == wn_5_hop[j][0] and test2id[i][1] == wn_5_hop[j][1] and test2id[i][2] == wn_5_hop[j][2]:
            index5.append(i)
    for j in range(len(wn_6_hop)):
        if test2id[i][0] == wn_6_hop[j][0] and test2id[i][1] == wn_6_hop[j][1] and test2id[i][2] == wn_6_hop[j][2]:
            index6.append(i)        
"""
# """
## FB15k-237
index = [i for i in range(len(test2id))]
index1 = []
index2 = []
index3 = []
index4 = []
index5 = []
index6 = []

for i in range(len(test2id)):
    for j in range(len(fb_1_hop)):
        if test2id[i][0] == fb_1_hop[j][0] and test2id[i][1] == fb_1_hop[j][1] and test2id[i][2] == fb_1_hop[j][2]:
            index1.append(i)
    for j in range(len(fb_2_hop)):
        if test2id[i][0] == fb_2_hop[j][0] and test2id[i][1] == fb_2_hop[j][1] and test2id[i][2] == fb_2_hop[j][2]:
            index2.append(i)
    for j in range(len(fb_3_hop)):
        if test2id[i][0] == fb_3_hop[j][0] and test2id[i][1] == fb_3_hop[j][1] and test2id[i][2] == fb_3_hop[j][2]:
            index3.append(i)
    for j in range(len(fb_4_hop)):
        if test2id[i][0] == fb_4_hop[j][0] and test2id[i][1] == fb_4_hop[j][1] and test2id[i][2] == fb_4_hop[j][2]:
            index4.append(i)
    for j in range(len(fb_5_hop)):
        if test2id[i][0] == fb_5_hop[j][0] and test2id[i][1] == fb_5_hop[j][1] and test2id[i][2] == fb_5_hop[j][2]:
            index5.append(i)
    for j in range(len(fb_6_hop)):
        if test2id[i][0] == fb_6_hop[j][0] and test2id[i][1] == fb_6_hop[j][1] and test2id[i][2] == fb_6_hop[j][2]:
            index6.append(i)        
# """
            
# Step 3.test셋 원본 분리

"""
## UMLS
test1 = []
test2 = []

## 1-hop umls
for i in index1:
    test1.append(test_umls[i])
test1 = np.array(test1)

## 2- hop umls
for i in index2:
    test2.append(test_umls[i])
test2 = np.array(test2)
"""

"""
## WN18RR      
test1 = []
test2 = [] 
test3 = []
test4 = [] 
test5 = []
test6 = []            
            
for i in index1:
    test1.append(test_wn[i])
test1 = np.array(test1)

for i in index2:
    test2.append(test_wn[i])
test2 = np.array(test2)
  
for i in index3:
    test3.append(test_wn[i])
test3 = np.array(test3)
  
for i in index4:
    test4.append(test_wn[i])
test4 = np.array(test4)  

for i in index5:
    test5.append(test_wn[i])
test5 = np.array(test5)
  
for i in index6:
    test6.append(test_wn[i])
test6 = np.array(test6)  
"""
# """
## FB15k-237    
test1 = []
test2 = [] 
test3 = []
test4 = [] 
test5 = []
test6 = []            
            
for i in index1:
    test1.append(test_fb[i])
test1 = np.array(test1)

for i in index2:
    test2.append(test_fb[i])
test2 = np.array(test2)
  
for i in index3:
    test3.append(test_fb[i])
test3 = np.array(test3)
  
for i in index4:
    test4.append(test_fb[i])
test4 = np.array(test4)  

for i in index5:
    test5.append(test_fb[i])
test5 = np.array(test5)
  
for i in index6:
    test6.append(test_fb[i])
test6 = np.array(test6)  
# """

## Step 4) tsv파일 생성
"""
## UMLS
df1 = pd.DataFrame(test1)
df2 = pd.DataFrame(test2)
path1 = 'C:/Users/PC/Desktop/test1.tsv'  # 파일 경로와 파일명을 원하는 위치로 변경해주세요.
path2 = 'C:/Users/PC/Desktop/test2.tsv' 
df1.to_csv(path1, sep='\t', index=False)
df2.to_csv(path2, sep='\t', index=False)
"""

"""
## WN18RR
df1 = pd.DataFrame(test1)
df2 = pd.DataFrame(test2)
df3 = pd.DataFrame(test3)
df4 = pd.DataFrame(test4)
df5 = pd.DataFrame(test5)
df6 = pd.DataFrame(test6)
path1 = 'C:/Users/PC/Desktop/Testing/wn18rr/test1.tsv'
path2 = 'C:/Users/PC/Desktop/Testing/wn18rr/test2.tsv'
path3 = 'C:/Users/PC/Desktop/Testing/wn18rr/test3.tsv'
path4 = 'C:/Users/PC/Desktop/Testing/wn18rr/test4.tsv'
path5 = 'C:/Users/PC/Desktop/Testing/wn18rr/test5.tsv'
path6 = 'C:/Users/PC/Desktop/Testing/wn18rr/test6.tsv' 
df1.to_csv(path1, sep='\t', index=False)
df2.to_csv(path2, sep='\t', index=False)
df3.to_csv(path3, sep='\t', index=False)
df4.to_csv(path4, sep='\t', index=False)
df5.to_csv(path5, sep='\t', index=False)
df6.to_csv(path6, sep='\t', index=False)
"""

# """
## FB15k-237
df1 = pd.DataFrame(test1)
df2 = pd.DataFrame(test2)
df3 = pd.DataFrame(test3)
df4 = pd.DataFrame(test4)
df5 = pd.DataFrame(test5)
df6 = pd.DataFrame(test6)
path1 = 'C:/Users/PC/Desktop/Testing/fb15k-237/test1.tsv'
path2 = 'C:/Users/PC/Desktop/Testing/fb15k-237/test2.tsv'
path3 = 'C:/Users/PC/Desktop/Testing/fb15k-237/test3.tsv'
path4 = 'C:/Users/PC/Desktop/Testing/fb15k-237/test4.tsv'
path5 = 'C:/Users/PC/Desktop/Testing/fb15k-237/test5.tsv'
path6 = 'C:/Users/PC/Desktop/Testing/fb15k-237/test6.tsv' 
df1.to_csv(path1, sep='\t', index=False)
df2.to_csv(path2, sep='\t', index=False)
df3.to_csv(path3, sep='\t', index=False)
df4.to_csv(path4, sep='\t', index=False)
df5.to_csv(path5, sep='\t', index=False)
df6.to_csv(path6, sep='\t', index=False)
# """