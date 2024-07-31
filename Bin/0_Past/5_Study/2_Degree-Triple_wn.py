import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import plotly.graph_objects as go
# Step 1) Data Load
# WN18RR
# """

train_wn = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/wn18rr/get_neighbor/train2id.txt", sep = "\s+", names = [0,1,2], dtype=str)
train_wn = np.array(train_wn)
wn_h = train_wn[:,0]
wn_r = train_wn[:,1]
wn_t = train_wn[:,2]

test_wn = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/wn18rr/get_neighbor/test2id.txt", sep = "\s+", names = [0,1,2], dtype=str)
test_wn = np.array(test_wn)
wn_h_test = test_wn[:,0]
wn_r_test = test_wn[:,1]
wn_t_test = test_wn[:,2]


## entity 유무 비교

entities_test = np.r_[wn_h_test, wn_t_test]
entities_test = set(sorted(entities_test))
entities_train = np.r_[wn_h, wn_t]
entities_train = set(entities_train)
non_list = entities_test - entities_train


## Step 2) Graph Construction

## wn18rr
G_wn = nx.Graph()

for head, relation, tail in zip(wn_h, wn_r, wn_t):
    G_wn.add_edge(head, tail, relation=relation)

G_wn.add_nodes_from(non_list)
print("Total relation number of WN18RR:",len(G_wn.edges))
print("Varience of relation in WN18RR:", len(wn_r) - len(G_wn.edges))
print("Total number of Graph Entites:", len(G_wn.nodes))
print("----------------------------------------------------")


#%%
## Step 3) Degree 별로 나누기
degree_test = []
for i in range(len(wn_h_test)):
    tmp = G_wn.degree(wn_h_test[i])
    degree_test.append(tmp)
# print(set(degree_test))

area0 = []
area1 = []
area2 = []
area3 = []
area4 = []
area5 = []

for i in range(len(degree_test)):
    if degree_test[i] <=5:
        area1.append((i, degree_test[i]))
    if 5 < degree_test[i] <=10:
        area2.append((i, degree_test[i]))
    if 10 < degree_test[i] <=15:
        area3.append((i, degree_test[i]))
    if 15 < degree_test[i] <=20:
        area4.append((i, degree_test[i]))        
    if 20 < degree_test[i] :
        area5.append((i, degree_test[i]))                
    if degree_test[i] ==0:
        area0.append((i, degree_test[i]))
        
area1, area2, area3 = np.array(area1), np.array(area2), np.array(area3)
area4, area5 = np.array(area4), np.array(area5) # (wn_h_test index, degree)
area0 = np.array(area0)
#%% Detaching 1 ~ 5 degre
## Step 4) Detaching 1 ~ 5 degree 
area1_1 = []
area1_2 = []
area1_3 = []
area1_4 = []
area1_5 = []

for i in range(len(area1)):
    if area1[i][1] == 1:
        area1_1.append((area1[i][0], area1[i][1]))
    if area1[i][1] == 2:
        area1_2.append((area1[i][0], area1[i][1])) 
    if area1[i][1] == 3:
        area1_3.append((area1[i][0], area1[i][1]))        
    if area1[i][1] == 4:
        area1_4.append((area1[i][0], area1[i][1])) 
    if area1[i][1] == 5:
        area1_5.append((area1[i][0], area1[i][1]))                
area1_1, area1_2, area1_3 = np.array(area1_1), np.array(area1_2), np.array(area1_3)
area1_4, area1_5 = np.array(area1_4), np.array(area1_5) # (wn_h_test index, degree)

degree1_id = []
degree2_id = []
degree3_id = []
degree4_id = []
degree5_id = []

for i in range(len(area1_1)):
    degree1_id.append(test_wn[area1_1[i][0]])
degree1_id = np.array(degree1_id)

for i in range(len(area1_2)):
    degree2_id.append(test_wn[area1_2[i][0]])
degree2_id = np.array(degree2_id)

for i in range(len(area1_3)):
    degree3_id.append(test_wn[area1_3[i][0]])
degree3_id = np.array(degree3_id)

for i in range(len(area1_4)):
    degree4_id.append(test_wn[area1_4[i][0]])
degree4_id = np.array(degree4_id)

for i in range(len(area1_5)):
    degree5_id.append(test_wn[area1_5[i][0]])
degree5_id = np.array(degree5_id)

test_wn_full = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/wn18rr/test.tsv", sep = "\s+", names = [0,1,2], dtype=str)
test_wn_full = np.array(test_wn_full)

degree1 = []
degree2 = []
degree3 = []
degree4 = []
degree5 = []

for i in range(len(area1_1)):
    degree1.append(test_wn_full[area1_1[i][0]])
degree1 = np.array(degree1)

for i in range(len(area1_2)):
    degree2.append(test_wn_full[area1_2[i][0]])
degree2 = np.array(degree2)

for i in range(len(area1_3)):
    degree3.append(test_wn_full[area1_3[i][0]])
degree3 = np.array(degree3)

for i in range(len(area1_4)):
    degree4.append(test_wn_full[area1_4[i][0]])
degree4 = np.array(degree4)

for i in range(len(area1_5)):
    degree5.append(test_wn_full[area1_5[i][0]])
degree5 = np.array(degree5)


# degree - entity 수
area1_1_h_id = area1_1[:,0]
area1_2_h_id = area1_2[:,0]
area1_3_h_id = area1_3[:,0]
area1_4_h_id = area1_4[:,0]
area1_5_h_id = area1_5[:,0]
area1_1_h, area1_2_h, area1_3_h, area1_4_h, area1_5_h = [], [], [], [], []
tmp = test_wn_full[:,0]
for i in range(len(area1_1_h_id)): 
    area1_1_h.append(tmp[area1_1_h_id[i]])
area1_1_h = set(area1_1_h)

for i in range(len(area1_2_h_id)): 
    area1_2_h.append(tmp[area1_2_h_id[i]])
area1_2_h = set(area1_2_h)

for i in range(len(area1_3_h_id)): 
    area1_3_h.append(tmp[area1_3_h_id[i]])
area1_3_h = set(area1_3_h)

for i in range(len(area1_4_h_id)): 
    area1_4_h.append(tmp[area1_4_h_id[i]])
area1_4_h = set(area1_4_h)

for i in range(len(area1_5_h_id)): 
    area1_5_h.append(tmp[area1_5_h_id[i]])
area1_5_h = set(area1_5_h)
print(len(area1_1), len(area1_2), len(area1_3), len(area1_4), len(area1_5))
print(len(area1_1_h), len(area1_2_h), len(area1_3_h), len(area1_4_h) ,len(area1_5_h))
#%% Detaching 6 ~ 10 degre
## Step 4) Detaching 6 ~ 10 degree 
area2_1 = []
area2_2 = []
area2_3 = []
area2_4 = []
area2_5 = []

for i in range(len(area2)):
    if area2[i][1] == 6:
        area2_1.append((area2[i][0], area2[i][1]))
    if area2[i][1] == 7:
        area2_2.append((area2[i][0], area2[i][1])) 
    if area2[i][1] == 8:
        area2_3.append((area2[i][0], area2[i][1]))        
    if area2[i][1] == 9:
        area2_4.append((area2[i][0], area2[i][1])) 
    if area2[i][1] == 10:
        area2_5.append((area2[i][0], area2[i][1]))                
area2_1, area2_2, area2_3 = np.array(area2_1), np.array(area2_2), np.array(area2_3)
area2_4, area2_5 = np.array(area2_4), np.array(area2_5) # (wn_h_test index, degree)

degree6_id = []
degree7_id = []
degree8_id = []
degree9_id = []
degree10_id = []

for i in range(len(area2_1)):
    degree6_id.append(test_wn[area2_1[i][0]])
degree6_id = np.array(degree6_id)

for i in range(len(area2_2)):
    degree7_id.append(test_wn[area2_2[i][0]])
degree7_id = np.array(degree7_id)

for i in range(len(area2_3)):
    degree8_id.append(test_wn[area2_3[i][0]])
degree8_id = np.array(degree8_id)

for i in range(len(area2_4)):
    degree9_id.append(test_wn[area2_4[i][0]])
degree9_id = np.array(degree9_id)

for i in range(len(area2_5)):
    degree10_id.append(test_wn[area2_5[i][0]])
degree10_id = np.array(degree10_id)

test_wn_full = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/wn18rr/test.tsv", sep = "\s+", names = [0,1,2], dtype=str)
test_wn_full = np.array(test_wn_full)

degree6 = []
degree7 = []
degree8 = []
degree9 = []
degree10 = []

for i in range(len(area2_1)):
    degree6.append(test_wn_full[area2_1[i][0]])
degree6 = np.array(degree6)

for i in range(len(area2_2)):
    degree7.append(test_wn_full[area2_2[i][0]])
degree7 = np.array(degree7)

for i in range(len(area2_3)):
    degree8.append(test_wn_full[area2_3[i][0]])
degree8 = np.array(degree8)

for i in range(len(area2_4)):
    degree9.append(test_wn_full[area2_4[i][0]])
degree9 = np.array(degree9)

for i in range(len(area2_5)):
    degree10.append(test_wn_full[area2_5[i][0]])
degree10 = np.array(degree10)

# degree - entity 수
area2_1_h_id = area2_1[:,0]
area2_2_h_id = area2_2[:,0]
area2_3_h_id = area2_3[:,0]
area2_4_h_id = area2_4[:,0]
area2_5_h_id = area2_5[:,0]
area2_1_h, area2_2_h, area2_3_h, area2_4_h, area2_5_h = [], [], [], [], []
tmp = test_wn_full[:,0]
for i in range(len(area2_1_h_id)): 
    area2_1_h.append(tmp[area2_1_h_id[i]])
area2_1_h = set(area2_1_h)

for i in range(len(area2_2_h_id)): 
    area2_2_h.append(tmp[area2_2_h_id[i]])
area2_2_h = set(area2_2_h)

for i in range(len(area2_3_h_id)): 
    area2_3_h.append(tmp[area2_3_h_id[i]])
area2_3_h = set(area2_3_h)

for i in range(len(area2_4_h_id)): 
    area2_4_h.append(tmp[area2_4_h_id[i]])
area2_4_h = set(area2_4_h)

for i in range(len(area2_5_h_id)): 
    area2_5_h.append(tmp[area2_5_h_id[i]])
area2_5_h = set(area2_5_h)
print("Degree - Triple:",len(area2_1_h_id), len(area2_2_h_id), len(area2_3_h_id), len(area2_4_h_id) ,len(area2_5_h_id))
print("Degree - Entity:",len(area2_1_h), len(area2_2_h), len(area2_3_h), len(area2_4_h) ,len(area2_5_h))

#%% Detaching 11 ~ 15 degree
## Step 4) Detaching 11 ~ 15 degree
area3_1 = []
area3_2 = []
area3_3 = []
area3_4 = []
area3_5 = []

for i in range(len(area3)):
    if area3[i][1] == 11:
        area3_1.append((area3[i][0], area3[i][1]))
    if area3[i][1] == 12:
        area3_2.append((area3[i][0], area3[i][1])) 
    if area3[i][1] == 13:
        area3_3.append((area3[i][0], area3[i][1]))        
    if area3[i][1] == 14:
        area3_4.append((area3[i][0], area3[i][1])) 
    if area3[i][1] == 15:
        area3_5.append((area3[i][0], area3[i][1]))                
area3_1, area3_2, area3_3 = np.array(area3_1), np.array(area3_2), np.array(area3_3)
area3_4, area3_5 = np.array(area3_4), np.array(area3_5) # (wn_h_test index, degree)

degree11_id = []
degree12_id = []
degree13_id = []
degree14_id = []
degree15_id = []

for i in range(len(area3_1)):
    degree11_id.append(test_wn[area3_1[i][0]])
degree11_id = np.array(degree11_id)

for i in range(len(area3_2)):
    degree12_id.append(test_wn[area3_2[i][0]])
degree12_id = np.array(degree12_id)

for i in range(len(area3_3)):
    degree13_id.append(test_wn[area3_3[i][0]])
degree13_id = np.array(degree13_id)

for i in range(len(area3_4)):
    degree14_id.append(test_wn[area3_4[i][0]])
degree14_id = np.array(degree14_id)

for i in range(len(area3_5)):
    degree15_id.append(test_wn[area3_5[i][0]])
degree15_id = np.array(degree15_id)

test_wn_full = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/wn18rr/test.tsv", sep = "\s+", names = [0,1,2], dtype=str)
test_wn_full = np.array(test_wn_full)

degree11 = []
degree12 = []
degree13 = []
degree14 = []
degree15 = []

for i in range(len(area3_1)):
    degree11.append(test_wn_full[area3_1[i][0]])
degree11 = np.array(degree11)

for i in range(len(area3_2)):
    degree12.append(test_wn_full[area3_2[i][0]])
degree12 = np.array(degree12)

for i in range(len(area3_3)):
    degree13.append(test_wn_full[area3_3[i][0]])
degree13 = np.array(degree13)

for i in range(len(area3_4)):
    degree14.append(test_wn_full[area3_4[i][0]])
degree14 = np.array(degree14)

for i in range(len(area3_5)):
    degree15.append(test_wn_full[area3_5[i][0]])
degree15 = np.array(degree15)

# degree - entity 수
area3_1_h_id = area3_1[:,0]
area3_2_h_id = area3_2[:,0]
area3_3_h_id = area3_3[:,0]
area3_4_h_id = area3_4[:,0]
area3_5_h_id = area3_5[:,0]
area3_1_h, area3_2_h, area3_3_h, area3_4_h, area3_5_h = [], [], [], [], []
tmp = test_wn_full[:,0]
for i in range(len(area3_1_h_id)): 
    area3_1_h.append(tmp[area3_1_h_id[i]])
area3_1_h = set(area3_1_h)

for i in range(len(area3_2_h_id)): 
    area3_2_h.append(tmp[area3_2_h_id[i]])
area3_2_h = set(area3_2_h)

for i in range(len(area3_3_h_id)): 
    area3_3_h.append(tmp[area3_3_h_id[i]])
area3_3_h = set(area3_3_h)

for i in range(len(area3_4_h_id)): 
    area3_4_h.append(tmp[area3_4_h_id[i]])
area3_4_h = set(area3_4_h)

for i in range(len(area3_5_h_id)): 
    area3_5_h.append(tmp[area3_5_h_id[i]])
area3_5_h = set(area3_5_h)
print("Degree - Triple:",len(area3_1_h_id), len(area3_2_h_id), len(area3_3_h_id), len(area3_4_h_id) ,len(area3_5_h_id))
print("Degree - Entity:",len(area3_1_h), len(area3_2_h), len(area3_3_h), len(area3_4_h) ,len(area3_5_h))
#%% Detaching 16 ~ 20 degree
## Step 4) Detaching 16 ~  degree
area4_1 = []
area4_2 = []
area4_3 = []
area4_4 = []
area4_5 = []

for i in range(len(area4)):
    if area4[i][1] == 16:
        area4_1.append((area4[i][0], area4[i][1]))
    if area4[i][1] == 17:
        area4_2.append((area4[i][0], area4[i][1])) 
    if area4[i][1] == 18:
        area4_3.append((area4[i][0], area4[i][1]))        
    if area4[i][1] == 19:
        area4_4.append((area4[i][0], area4[i][1])) 
    if area4[i][1] == 20:
        area4_5.append((area4[i][0], area4[i][1]))                
area4_1, area4_2, area4_3 = np.array(area4_1), np.array(area4_2), np.array(area4_3)
area4_4, area4_5 = np.array(area4_4), np.array(area4_5) # (wn_h_test index, degree)

degree16_id = []
degree17_id = []
degree18_id = []
degree19_id = []
degree20_id = []

for i in range(len(area4_1)):
    degree16_id.append(test_wn[area4_1[i][0]])
degree16_id = np.array(degree16_id)

for i in range(len(area4_2)):
    degree17_id.append(test_wn[area4_2[i][0]])
degree17_id = np.array(degree17_id)

for i in range(len(area4_3)):
    degree18_id.append(test_wn[area4_3[i][0]])
degree18_id = np.array(degree18_id)

for i in range(len(area4_4)):
    degree19_id.append(test_wn[area4_4[i][0]])
degree19_id = np.array(degree19_id)

for i in range(len(area4_5)):
    degree20_id.append(test_wn[area4_5[i][0]])
degree20_id = np.array(degree20_id)

test_wn_full = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/wn18rr/test.tsv", sep = "\s+", names = [0,1,2], dtype=str)
test_wn_full = np.array(test_wn_full)

degree16 = []
degree17 = []
degree18 = []
degree19 = []
degree20 = []

for i in range(len(area4_1)):
    degree16.append(test_wn_full[area4_1[i][0]])
degree16 = np.array(degree16)

for i in range(len(area4_2)):
    degree17.append(test_wn_full[area4_2[i][0]])
degree17 = np.array(degree17)

for i in range(len(area4_3)):
    degree18.append(test_wn_full[area4_3[i][0]])
degree18 = np.array(degree18)

for i in range(len(area4_4)):
    degree19.append(test_wn_full[area4_4[i][0]])
degree19 = np.array(degree19)

for i in range(len(area4_5)):
    degree20.append(test_wn_full[area4_5[i][0]])
degree20 = np.array(degree20)

# degree - entity 수
area4_1_h_id = area4_1[:,0]
area4_2_h_id = area4_2[:,0]
area4_3_h_id = area4_3[:,0]
area4_4_h_id = area4_4[:,0]
area4_5_h_id = area4_5[:,0]
area4_1_h, area4_2_h, area4_3_h, area4_4_h, area4_5_h = [], [], [], [], []
tmp = test_wn_full[:,0]
for i in range(len(area4_1_h_id)): 
    area4_1_h.append(tmp[area4_1_h_id[i]])
area4_1_h = set(area4_1_h)

for i in range(len(area4_2_h_id)): 
    area4_2_h.append(tmp[area4_2_h_id[i]])
area4_2_h = set(area4_2_h)

for i in range(len(area4_3_h_id)): 
    area4_3_h.append(tmp[area4_3_h_id[i]])
area4_3_h = set(area4_3_h)

for i in range(len(area4_4_h_id)): 
    area4_4_h.append(tmp[area4_4_h_id[i]])
area4_4_h = set(area4_4_h)

for i in range(len(area4_5_h_id)): 
    area4_5_h.append(tmp[area4_5_h_id[i]])
area4_5_h = set(area4_5_h)
print("Degree - Triple:",len(area4_1_h_id), len(area4_2_h_id), len(area4_3_h_id), len(area4_4_h_id) ,len(area4_5_h_id))
print("Degree - Entity:",len(area4_1_h), len(area4_2_h), len(area4_3_h), len(area4_4_h) ,len(area4_5_h))

#%% Detaching ~ > 20 degree
## Step 4) Detaching ~ > 20 degree
# area5_1 = []
# area5_2 = []
# area5_3 = []
# area5_4 = []
# area5_5 = []
# area5_6 = []
# area5_7 = []
# area5_8 = []
# area5_9 = []
# area5_10 = []
# area5_11 = []
# area5_12 = []
# area5_13 = []
# area5_14 = []
# area5_15 = []
# area5_16 = []
# area5_17 = []
# area5_18 = []
# area5_19 = []
# area5_20 = []
# area5_21 = []
# area5_22 = []
# area5_23 = []
# area5_24 = []
# area5_25 = []
# area5_26 = []
# area5_27 = []
# area5_28 = []
# area5_29 = []
# area5_30 = []
# area5_31 = []
# area5_32 = []
# area5_33 = []
# area5_34 = []
# area5_35 = []
# area5_36 = []
# area5_37 = []
# area5_38 = []
# area5_39 = []
# area5_40 = []
# area5_37 = []

# for i in range(len(area5)):
#     if area5[i][1] == 21:
#         area5_1.append((area5[i][0], area5[i][1]))
#     if area5[i][1] == 22:
#         area5_2.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 23:
#         area5_3.append((area5[i][0], area5[i][1]))        
#     if area5[i][1] == 24:
#         area5_4.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 25:
#         area5_5.append((area5[i][0], area5[i][1]))
#     if area5[i][1] == 26:
#         area5_6.append((area5[i][0], area5[i][1]))
#     if area5[i][1] == 27:
#         area5_7.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 28:
#         area5_8.append((area5[i][0], area5[i][1]))        
#     if area5[i][1] == 31:
#         area5_9.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 32:
#         area5_10.append((area5[i][0], area5[i][1]))  
#     if area5[i][1] == 34:
#         area5_11.append((area5[i][0], area5[i][1]))
#     if area5[i][1] == 36:
#         area5_12.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 37:
#         area5_13.append((area5[i][0], area5[i][1]))        
#     if area5[i][1] == 38:
#         area5_14.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 39:
#         area5_15.append((area5[i][0], area5[i][1]))  
#     if area5[i][1] == 40:
#         area5_16.append((area5[i][0], area5[i][1]))
#     if area5[i][1] == 41:
#         area5_17.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 44:
#         area5_18.append((area5[i][0], area5[i][1]))        
#     if area5[i][1] == 45:
#         area5_19.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 46:
#         area5_20.append((area5[i][0], area5[i][1]))  
#     if area5[i][1] == 51:
#         area5_21.append((area5[i][0], area5[i][1]))
#     if area5[i][1] == 52:
#         area5_22.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 55:
#         area5_23.append((area5[i][0], area5[i][1]))        
#     if area5[i][1] == 56:
#         area5_24.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 61:
#         area5_25.append((area5[i][0], area5[i][1]))  
#     if area5[i][1] == 64:
#         area5_26.append((area5[i][0], area5[i][1]))
#     if area5[i][1] == 66:
#         area5_27.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 67:
#         area5_28.append((area5[i][0], area5[i][1]))        
#     if area5[i][1] == 79:
#         area5_29.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 78:
#         area5_30.append((area5[i][0], area5[i][1]))  
#     if area5[i][1] == 81:
#         area5_31.append((area5[i][0], area5[i][1]))
#     if area5[i][1] == 92:
#         area5_32.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 109:
#         area5_33.append((area5[i][0], area5[i][1]))        
#     if area5[i][1] == 115:
#         area5_34.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 143:
#         area5_35.append((area5[i][0], area5[i][1]))  
#     if area5[i][1] == 164:
#         area5_36.append((area5[i][0], area5[i][1]))
#     if area5[i][1] == 230:
#         area5_37.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 342:
#         area5_38.append((area5[i][0], area5[i][1]))        
#     if area5[i][1] == 466:
#         area5_39.append((area5[i][0], area5[i][1])) 
#     if area5[i][1] == 482:
#         area5_40.append((area5[i][0], area5[i][1]))                  
# area5_1, area5_17 = np.array(area5_1), np.array(area5_17) # (wn_h_test index, degree)
# area5_2, area5_18 = np.array(area5_2), np.array(area5_18) # (wn_h_test index, degree)
# area5_3, area5_19 = np.array(area5_3), np.array(area5_19) # (wn_h_test index, degree)
# area5_4, area5_20 = np.array(area5_4), np.array(area5_20) # (wn_h_test index, degree)
# area5_5, area5_21 = np.array(area5_5), np.array(area5_21) # (wn_h_test index, degree)
# area5_6, area5_22 = np.array(area5_6), np.array(area5_22) # (wn_h_test index, degree)
# area5_7, area5_23 = np.array(area5_7), np.array(area5_23) # (wn_h_test index, degree)
# area5_8, area5_24 = np.array(area5_8), np.array(area5_24) # (wn_h_test index, degree)
# area5_9, area5_25 = np.array(area5_9), np.array(area5_25) # (wn_h_test index, degree)
# area5_10, area5_26 = np.array(area5_10), np.array(area5_26) # (wn_h_test index, degree)
# area5_11, area5_27 = np.array(area5_11), np.array(area5_27) # (wn_h_test index, degree)
# area5_12, area5_28 = np.array(area5_12), np.array(area5_28) # (wn_h_test index, degree)
# area5_13, area5_29 = np.array(area5_13), np.array(area5_29) # (wn_h_test index, degree)
# area5_14, area5_30 = np.array(area5_14), np.array(area5_30) # (wn_h_test index, degree)
# area5_15, area5_31 = np.array(area5_15), np.array(area5_31) # (wn_h_test index, degree)
# area5_16, area5_32 = np.array(area5_16), np.array(area5_32) # (wn_h_test index, degree)
# area5_33, area5_37 = np.array(area5_33), np.array(area5_37) # (wn_h_test index, degree)
# area5_34, area5_38 = np.array(area5_34), np.array(area5_38) # (wn_h_test index, degree)
# area5_35, area5_39 = np.array(area5_35), np.array(area5_39) # (wn_h_test index, degree)
# area5_36, area5_40 = np.array(area5_36), np.array(area5_40) # (wn_h_test index, degree)

# degree16_id = []
# degree17_id = []
# degree18_id = []
# degree19_id = []
# degree20_id = []

# for i in range(len(area4_1)):
#     degree16_id.append(test_wn[area4_1[i][0]])
# degree16_id = np.array(degree16_id)

# for i in range(len(area4_2)):
#     degree17_id.append(test_wn[area4_2[i][0]])
# degree17_id = np.array(degree17_id)

# for i in range(len(area4_3)):
#     degree18_id.append(test_wn[area4_3[i][0]])
# degree18_id = np.array(degree18_id)

# for i in range(len(area4_4)):
#     degree19_id.append(test_wn[area4_4[i][0]])
# degree19_id = np.array(degree19_id)

# for i in range(len(area4_5)):
#     degree20_id.append(test_wn[area4_5[i][0]])
# degree20_id = np.array(degree20_id)

# test_wn_full = pd.read_csv("C:/Users/PC/Desktop/실험/Banchmark/wn18rr/test.tsv", sep = "\s+", names = [0,1,2], dtype=str)
# test_wn_full = np.array(test_wn_full)

# degree16 = []
# degree17 = []
# degree18 = []
# degree19 = []
# degree20 = []

# for i in range(len(area4_1)):
#     degree16.append(test_wn_full[area4_1[i][0]])
# degree16 = np.array(degree16)

# for i in range(len(area4_2)):
#     degree17.append(test_wn_full[area4_2[i][0]])
# degree17 = np.array(degree17)

# for i in range(len(area4_3)):
#     degree18.append(test_wn_full[area4_3[i][0]])
# degree18 = np.array(degree18)

# for i in range(len(area4_4)):
#     degree19.append(test_wn_full[area4_4[i][0]])
# degree19 = np.array(degree19)

# for i in range(len(area4_5)):
#     degree20.append(test_wn_full[area4_5[i][0]])
# degree20 = np.array(degree20)

# # degree - entity 수
# area4_1_h_id = area4_1[:,0]
# area4_2_h_id = area4_2[:,0]
# area4_3_h_id = area4_3[:,0]
# area4_4_h_id = area4_4[:,0]
# area4_5_h_id = area4_5[:,0]
# area4_1_h, area4_2_h, area4_3_h, area4_4_h, area4_5_h = [], [], [], [], []
# tmp = test_wn_full[:,0]
# for i in range(len(area4_1_h_id)): 
#     area4_1_h.append(tmp[area4_1_h_id[i]])
# area4_1_h = set(area4_1_h)

# for i in range(len(area4_2_h_id)): 
#     area4_2_h.append(tmp[area4_2_h_id[i]])
# area4_2_h = set(area4_2_h)

# for i in range(len(area4_3_h_id)): 
#     area4_3_h.append(tmp[area4_3_h_id[i]])
# area4_3_h = set(area4_3_h)

# for i in range(len(area4_4_h_id)): 
#     area4_4_h.append(tmp[area4_4_h_id[i]])
# area4_4_h = set(area4_4_h)

# for i in range(len(area4_5_h_id)): 
#     area4_5_h.append(tmp[area4_5_h_id[i]])
# area4_5_h = set(area4_5_h)
# print("Degree - Triple:",len(area4_1_h_id), len(area4_2_h_id), len(area4_3_h_id), len(area4_4_h_id) ,len(area4_5_h_id))
# print("Degree - Entity:",len(area4_1_h), len(area4_2_h), len(area4_3_h), len(area4_4_h) ,len(area4_5_h))
area5_1 = []


for i in range(len(area5)):
    if area5[i][1] >= 21:
        area5_1.append((area5[i][0], area5[i][1]))
               
area5_1 = np.array(area5_1) # (wn_h_test index, degree)

degree21_id = []


for i in range(len(area5_1)):
    degree21_id.append(test_wn[area5_1[i][0]])
degree21_id = np.array(degree21_id)

test_wn_full = pd.read_csv("C:/Users/PC/Desktop/실험/Banchmark/wn18rr/test.tsv", sep = "\s+", names = [0,1,2], dtype=str)
test_wn_full = np.array(test_wn_full)

degree21 = []

for i in range(len(area5_1)):
    degree21.append(test_wn_full[area5_1[i][0]])
degree21 = np.array(degree21)

# degree - entity 수
area5_1_h_id = area5_1[:,0]
area5_1_h = []
tmp = test_wn_full[:,0]
for i in range(len(area5_1_h_id)): 
    area5_1_h.append(tmp[area5_1_h_id[i]])
area5_1_h = set(area5_1_h)

print("Degree - Triple:",len(area5_1_h_id))
print("Degree - Entity:",len(area5_1_h))