import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import plotly.graph_objects as go
# Step 1) Data Load
# FB15K-237
# """
train_fb = pd.read_csv("C:/Users/PC/Desktop/실험/Banchmark/fb15k-237/get_neighbor/train2id.txt", sep = "\s+", names = [0,1,2])
train_fb = np.array(train_fb)
fb_h = train_fb[:,0]
fb_r = train_fb[:,1]
fb_t = train_fb[:,2]

test_fb = pd.read_csv("C:/Users/PC/Desktop/실험/Banchmark/fb15k-237/get_neighbor/test2id.txt", sep = "\s+", names = [0,1,2])
test_fb = np.array(test_fb)
fb_h_test = test_fb[:,0]
fb_r_test = test_fb[:,1]
fb_t_test = test_fb[:,2]

entities_test = np.r_[fb_h_test, fb_t_test]
entities_test = set(sorted(entities_test))
entities_train = np.r_[fb_h, fb_t]
entities_train = set(entities_train)
non_list = entities_test - entities_train

## fb15k-237
G_fb = nx.Graph()

for head, relation, tail in zip(fb_h, fb_r, fb_t):
    G_fb.add_edge(head, tail, relation=relation)


G_fb.add_nodes_from(non_list)
print("Total relation number of FB15k-237:",len(G_fb.edges))
print("Varience of relation in FB15k-237:", len(fb_r) - len(G_fb.edges))
print("----------------------------------------------------")
# """

#%% Step 2) Degree 별로 나누기
degree_test = []
for i in range(len(fb_h_test)):
    tmp = G_fb.degree(fb_h_test[i])
    degree_test.append(tmp)
print(set(degree_test))


area1, area2, area3, area4, area5, area6  =[] ,[], [], [], [], []

for i in range(len(degree_test)):

    if 0 < degree_test[i] <=100:
        area1.append((i, degree_test[i]))
    if 100 < degree_test[i] <=200:
        area2.append((i, degree_test[i]))
    if 200 < degree_test[i] <=300:
        area3.append((i, degree_test[i]))
    if 300 < degree_test[i] <=400:
        area4.append((i, degree_test[i]))        
    if 400 < degree_test[i] <=500 :
        area5.append((i, degree_test[i]))                
    if 500 < degree_test[i]:
        area6.append((i, degree_test[i]))

area1, area2, area3 = np.array(area1), np.array(area2), np.array(area3)
area4, area5, area6 = np.array(area4), np.array(area5), np.array(area6)

#%% Step 3) 1 ~ 100, Detaching by 10
area1_1, area1_2, area1_3 ,area1_4, area1_5 = [], [], [], [], []
area1_6, area1_7, area1_8 ,area1_9, area1_10 = [], [], [], [], []

for i in range(len(area1)):   
    if 1 <= area1[i][1] <= 10:
        area1_1.append((area1[i][0], area1[i][1]))
    if 11 <= area1[i][1] <= 20:
        area1_2.append((area1[i][0], area1[i][1])) 
    if 21 <= area1[i][1] <= 30:
        area1_3.append((area1[i][0], area1[i][1]))        
    if 31 <= area1[i][1] <= 40:
        area1_4.append((area1[i][0], area1[i][1])) 
    if 41 <= area1[i][1] <= 50:
        area1_5.append((area1[i][0], area1[i][1]))  
    if 51 <= area1[i][1] <= 60:
        area1_6.append((area1[i][0], area1[i][1]))    
    if 61 <= area1[i][1] <= 70:
        area1_7.append((area1[i][0], area1[i][1]))
    if 71 <= area1[i][1] <= 80:
        area1_8.append((area1[i][0], area1[i][1])) 
    if 81 <= area1[i][1] <= 90:
        area1_9.append((area1[i][0], area1[i][1]))        
    if 91 <= area1[i][1] <= 100:
        area1_10.append((area1[i][0], area1[i][1])) 
area1_1, area1_2, area1_3, area1_4, area1_5 = np.array(area1_1), np.array(area1_2), np.array(area1_3), np.array(area1_4), np.array(area1_5)  
area1_6, area1_7, area1_8, area1_9, area1_10 = np.array(area1_6), np.array(area1_7), np.array(area1_8), np.array(area1_9), np.array(area1_10)  

## id 분리
degree1_id, degree2_id, degree3_id, degree4_id ,degree5_id = [], [], [], [], [] 
degree6_id, degree7_id, degree8_id, degree9_id ,degree10_id = [], [], [], [], [] 
for i in range(len(area1_1)):
    degree1_id.append(test_fb[area1_1[i][0]])
degree1_id = np.array(degree1_id)
for i in range(len(area1_2)):
    degree2_id.append(test_fb[area1_2[i][0]])
degree2_id = np.array(degree2_id)
for i in range(len(area1_3)):
    degree3_id.append(test_fb[area1_3[i][0]])
degree3_id = np.array(degree3_id)
for i in range(len(area1_4)):
    degree4_id.append(test_fb[area1_4[i][0]])
degree4_id = np.array(degree4_id)
for i in range(len(area1_5)):
    degree5_id.append(test_fb[area1_5[i][0]])
degree5_id = np.array(degree5_id)
for i in range(len(area1_6)):
    degree6_id.append(test_fb[area1_6[i][0]])
degree6_id = np.array(degree6_id)
for i in range(len(area1_7)):
    degree7_id.append(test_fb[area1_7[i][0]])
degree7_id = np.array(degree7_id)
for i in range(len(area1_8)):
    degree8_id.append(test_fb[area1_8[i][0]])
degree8_id = np.array(degree8_id)
for i in range(len(area1_9)):
    degree9_id.append(test_fb[area1_9[i][0]])
degree9_id = np.array(degree9_id)
for i in range(len(area1_10)):
    degree10_id.append(test_fb[area1_10[i][0]])
degree10_id = np.array(degree10_id)

## test set 분리
test_fb_full = pd.read_csv("C:/Users/PC/Desktop/실험/Banchmark/fb15k-237/test.tsv", sep = "\s+", names = [0,1,2], dtype=str)
test_fb_full = np.array(test_fb_full)
degree1, degree2, degree3, degree4, degree5 = [], [], [], [], []
degree6, degree7, degree8, degree9, degree10 = [], [], [], [], []
for i in range(len(area1_1)):
    degree1.append(test_fb_full[area1_1[i][0]])
degree1 = np.array(degree1)
for i in range(len(area1_2)):
    degree2.append(test_fb_full[area1_2[i][0]])
degree2 = np.array(degree2)
for i in range(len(area1_3)):
    degree3.append(test_fb_full[area1_3[i][0]])
degree3 = np.array(degree3)
for i in range(len(area1_4)):
    degree4.append(test_fb_full[area1_4[i][0]])
degree4 = np.array(degree4)
for i in range(len(area1_5)):
    degree5.append(test_fb_full[area1_5[i][0]])
degree5 = np.array(degree5)
for i in range(len(area1_6)):
    degree6.append(test_fb_full[area1_6[i][0]])
degree6 = np.array(degree6)
for i in range(len(area1_7)):
    degree7.append(test_fb_full[area1_7[i][0]])
degree7 = np.array(degree7)
for i in range(len(area1_8)):
    degree8.append(test_fb_full[area1_8[i][0]])
degree8 = np.array(degree8)
for i in range(len(area1_9)):
    degree9.append(test_fb_full[area1_9[i][0]])
degree9 = np.array(degree9)
for i in range(len(area1_10)):
    degree10.append(test_fb_full[area1_10[i][0]])
degree10 = np.array(degree10)

# degree - entity 수
area1_1_h_id = area1_1[:,0]
area1_2_h_id = area1_2[:,0]
area1_3_h_id = area1_3[:,0]
area1_4_h_id = area1_4[:,0]
area1_5_h_id = area1_5[:,0]
area1_6_h_id = area1_6[:,0]
area1_7_h_id = area1_7[:,0]
area1_8_h_id = area1_8[:,0]
area1_9_h_id = area1_9[:,0]
area1_10_h_id = area1_10[:,0]
area1_1_h, area1_2_h, area1_3_h, area1_4_h, area1_5_h = [], [], [], [], []
area1_6_h, area1_7_h, area1_8_h, area1_9_h, area1_10_h = [], [], [], [], []

tmp = test_fb_full[:,0]
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
for i in range(len(area1_6_h_id)): 
    area1_6_h.append(tmp[area1_6_h_id[i]])
area1_6_h = set(area1_6_h)
for i in range(len(area1_7_h_id)): 
    area1_7_h.append(tmp[area1_7_h_id[i]])
area1_7_h = set(area1_7_h)
for i in range(len(area1_8_h_id)): 
    area1_8_h.append(tmp[area1_8_h_id[i]])
area1_8_h = set(area1_8_h)
for i in range(len(area1_9_h_id)): 
    area1_9_h.append(tmp[area1_9_h_id[i]])
area1_9_h = set(area1_9_h)
for i in range(len(area1_10_h_id)): 
    area1_10_h.append(tmp[area1_10_h_id[i]])
area1_10_h = set(area1_10_h)

weight1_1 = [len(area1_1_h_id), len(area1_2_h_id), len(area1_3_h_id), len(area1_4_h_id), len(area1_5_h_id),
             len(area1_6_h_id), len(area1_7_h_id), len(area1_8_h_id), len(area1_9_h_id), len(area1_10_h_id)]
weight1_2 = [len(area1_1_h), len(area1_2_h), len(area1_3_h), len(area1_4_h), len(area1_5_h),
             len(area1_6_h), len(area1_7_h), len(area1_8_h), len(area1_9_h), len(area1_10_h)]
print("Degree - Triple:", weight1_1)
print("Degree - Entity:",weight1_2)