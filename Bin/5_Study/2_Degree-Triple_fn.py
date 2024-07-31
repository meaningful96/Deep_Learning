import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import plotly.graph_objects as go
# Step 1) Data Load
# FB15K-237
# """
train_fb = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/fb15k-237/get_neighbor/train2id.txt", sep = "\s+", names = [0,1,2])
train_fb = np.array(train_fb)
fb_h = train_fb[:,0]
fb_r = train_fb[:,1]
fb_t = train_fb[:,2]

test_fb = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/fb15k-237/get_neighbor/test2id.txt", sep = "\s+", names = [0,1,2])
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

#%% 분포확인
## Step 3) Degree 별로 나누기
degree_test = []
for i in range(len(fb_h_test)):
    tmp = G_fb.degree(fb_h_test[i])
    degree_test.append(tmp)
print(set(degree_test))

area0 = []
area1, area5, area9, area13, area17  = [], [], [], [], []
area2, area6, area10, area14, area18 = [], [], [], [], []
area3, area7, area11, area15, area19 = [], [], [], [], []
area4, area8, area12, area16, area20 = [], [], [], [], []
area21 = []

for i in range(len(degree_test)):
    if degree_test[i] ==0:
        area0.append((i, degree_test[i]))
    if 0 < degree_test[i] <=100:
        area1.append((i, degree_test[i]))
    if 100 < degree_test[i] <=200:
        area2.append((i, degree_test[i]))
    if 200 < degree_test[i] <=300:
        area3.append((i, degree_test[i]))
    if 300 < degree_test[i] <=400:
        area4.append((i, degree_test[i]))                     
    if 400 < degree_test[i] <=500:
        area5.append((i, degree_test[i]))
    if 500 < degree_test[i] <=600:
        area6.append((i, degree_test[i]))
    if 600 < degree_test[i] <=700:
        area7.append((i, degree_test[i]))   
    if 700 < degree_test[i] <=800:
        area8.append((i, degree_test[i]))
    if 800 < degree_test[i] <=90:
        area9.append((i, degree_test[i]))
    if 900 < degree_test[i] <=1000:
        area10.append((i, degree_test[i]))           
    if 1000 < degree_test[i] <=1100:
        area11.append((i, degree_test[i]))
    if 1100 < degree_test[i] <=1200:
        area12.append((i, degree_test[i]))
    if 1200 < degree_test[i] <=1300:
        area13.append((i, degree_test[i]))   
    if 1300 < degree_test[i] <=1400:
        area14.append((i, degree_test[i]))
    if 1400 < degree_test[i] <=1500:
        area15.append((i, degree_test[i]))
    if 1500 < degree_test[i] <=1600:
        area16.append((i, degree_test[i]))           
    if 1600 < degree_test[i] <=1700:
        area17.append((i, degree_test[i]))
    if 1700 < degree_test[i] <=1800:
        area18.append((i, degree_test[i]))
    if 1800 < degree_test[i] <=1900:
        area19.append((i, degree_test[i]))  
    if 1900 < degree_test[i] <=2000:
        area20.append((i, degree_test[i]))  
    if 2000 < degree_test[i] :
        area21.append((i, degree_test[i]))  

# # sorted_degree = sorted(degree_test)
# count = []
# temp = 0
# for i in range(51):
#     count.append((i, degree_test.count(i)))
#     temp = temp + degree_test.count(i)
# count = np.array(count)
# print("0~50 sum:", temp)
# for i in range(51):
#     print(count[i][1])
# import matplotlib.pyplot as plt

# plt.close("all")

# f1 = plt.figure(figsize = (10,5))
# ax1 = plt.axes()
# ax1.plot(count[:,0], count[:,1], 'r.')
# plt.xticks([i*5 for i in range(20)])
# plt.xlabel("Degree 1~100")
# plt.ylabel("Triple")
# area0 = np.array(area0)
# area1, area5, area9, area13, area17  = np.array(area1), np.array(area5), np.array(area9), np.array(area13), np.array(area17)
# area2, area6, area10, area14, area18 = np.array(area2), np.array(area6), np.array(area10), np.array(area14), np.array(area18) 
# area3, area7, area11, area15, area19 = np.array(area3), np.array(area7), np.array(area11), np.array(area15), np.array(area19)
# area4, area8, area12, area16, area20 = np.array(area4), np.array(area8), np.array(area12), np.array(area16), np.array(area20)
# area21 = np.array(area21)

# ## histogram 그리기
# weight = [len(area0), len(area1), len(area2), len(area3), len(area4),
#           len(area5), len(area6), len(area7), len(area8), len(area9),
#           len(area10), len(area11), len(area12), len(area13), len(area14),
#           len(area15), len(area16), len(area17), len(area18), len(area19),
#           len(area20), len(area21)
#           ]


# x = [i for i in range(0,22)]
# f2 = plt.figure(figsize = (10,5))
# ax2 = plt.axes()
# ax2.bar(x, weight)
# plt.xlabel("Degree (unit: 100)")
# plt.ylabel("Triple")
#%% Step 3) Degree 별로 나누기
degree_test = []
for i in range(len(fb_h_test)):
    tmp = G_fb.degree(fb_h_test[i])
    degree_test.append(tmp)
print(set(degree_test))

area0 = []
area1, area5, area9, area13, area17  = [], [], [], [], []
area2, area6, area10, area14, area18 = [], [], [], [], []
area3, area7, area11, area15, area19 = [], [], [], [], []
area4, area8, area12, area16, area20 = [], [], [], [], []
area21 = []
for i in range(len(degree_test)):
    if degree_test[i] ==0:
        area0.append((i, degree_test[i]))
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
    if 500 < degree_test[i] <=600:
        area6.append((i, degree_test[i]))
    if 600 < degree_test[i] <=700:
        area7.append((i, degree_test[i]))
    if 700 < degree_test[i] <=800:
        area8.append((i, degree_test[i]))
    if 800 < degree_test[i] <=900:
        area9.append((i, degree_test[i]))        
    if 900 < degree_test[i] <=1000 :
        area10.append((i, degree_test[i]))        
    if 1000 < degree_test[i] <=1100 :
        area11.append((i, degree_test[i]))            
    if 1100 < degree_test[i] <=1200 :
        area12.append((i, degree_test[i]))
    if 1200 < degree_test[i] <=1300 :
        area13.append((i, degree_test[i]))            
    if 1300 < degree_test[i] <=1400 :
        area14.append((i, degree_test[i]))  
    if 1400 < degree_test[i] <=1500 :
        area15.append((i, degree_test[i]))            
    if 1500 < degree_test[i] <=1600 :
        area16.append((i, degree_test[i]))           
    if 1600 < degree_test[i] <=1700 :
        area17.append((i, degree_test[i]))            
    if 1700 < degree_test[i] <=1800 :
        area18.append((i, degree_test[i]))       
    if 1800 < degree_test[i] <=1900 :
        area19.append((i, degree_test[i]))            
    if 1900 < degree_test[i] <=2000 :
        area20.append((i, degree_test[i]))          
    if 2000 < degree_test[i] :
        area21.append((i, degree_test[i]))          
area0 = np.array(area0)
area1, area5, area9, area13, area17  = np.array(area1), np.array(area5), np.array(area9), np.array(area13), np.array(area17)
area2, area6, area10, area14, area18 = np.array(area2), np.array(area6), np.array(area10), np.array(area14), np.array(area18) 
area3, area7, area11, area15, area19 = np.array(area3), np.array(area7), np.array(area11), np.array(area15), np.array(area19)
area4, area8, area12, area16, area20 = np.array(area4), np.array(area8), np.array(area12), np.array(area16), np.array(area20)
area21 = np.array(area21)        

#%% Degree 0 ~ 20
area0_1 = []
area1_1, area1_2, area1_3 ,area1_4, area1_5 = [], [], [], [], []
area1_6, area1_7, area1_8 ,area1_9, area1_10 = [], [], [], [], []
area1_11, area1_12, area1_13 ,area1_14, area1_15 = [], [], [], [], []
area1_16, area1_17, area1_18 ,area1_19, area1_20 = [], [], [], [], []

for i in range(len(area0)):
    if area0[i][1] == 0:
        area0_1.append((area0[i][0], area0[i][1])) 

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
    if area1[i][1] == 6:
        area1_6.append((area1[i][0], area1[i][1]))    
    if area1[i][1] == 7:
        area1_7.append((area1[i][0], area1[i][1]))
    if area1[i][1] == 8:
        area1_8.append((area1[i][0], area1[i][1])) 
    if area1[i][1] == 9:
        area1_9.append((area1[i][0], area1[i][1]))        
    if area1[i][1] == 10:
        area1_10.append((area1[i][0], area1[i][1])) 
    if area1[i][1] == 11:
        area1_11.append((area1[i][0], area1[i][1]))  
    if area1[i][1] == 12:
        area1_12.append((area1[i][0], area1[i][1]))    
    if area1[i][1] == 13:
        area1_13.append((area1[i][0], area1[i][1]))
    if area1[i][1] == 14:
        area1_14.append((area1[i][0], area1[i][1])) 
    if area1[i][1] == 15:
        area1_15.append((area1[i][0], area1[i][1]))        
    if area1[i][1] == 16:
        area1_16.append((area1[i][0], area1[i][1])) 
    if area1[i][1] == 17:
        area1_17.append((area1[i][0], area1[i][1]))  
    if area1[i][1] == 18:
        area1_18.append((area1[i][0], area1[i][1]))    
    if area1[i][1] == 19:
        area1_19.append((area1[i][0], area1[i][1]))
    if area1[i][1] == 20:
        area1_20.append((area1[i][0], area1[i][1])) 
              
area0_1 = np.array(area0_1)
area1_1, area1_2, area1_3, area1_4, area1_5 = np.array(area1_1), np.array(area1_2), np.array(area1_3), np.array(area1_4), np.array(area1_5)  
area1_6, area1_7, area1_8, area1_9, area1_10 = np.array(area1_6), np.array(area1_7), np.array(area1_8), np.array(area1_9), np.array(area1_10)  
area1_11, area1_12, area1_13, area1_14, area1_15 = np.array(area1_11), np.array(area1_12), np.array(area1_13), np.array(area1_14), np.array(area1_15)  
area1_16, area1_17, area1_18, area1_19, area1_20 = np.array(area1_16), np.array(area1_17), np.array(area1_18), np.array(area1_19), np.array(area1_20)  

degree0_id = []
degree1_id, degree2_id, degree3_id, degree4_id ,degree5_id = [], [], [], [], [] 
degree6_id, degree7_id, degree8_id, degree9_id ,degree10_id = [], [], [], [], [] 
degree11_id, degree12_id, degree13_id, degree14_id ,degree15_id = [], [], [], [], [] 
degree16_id, degree17_id, degree18_id, degree19_id ,degree20_id = [], [], [], [], [] 

for i in range(len(area0_1)):
    degree0_id.append(test_fb[area0_1[i][0]])
degree0_id = np.array(degree0_id)
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
for i in range(len(area1_11)):
    degree11_id.append(test_fb[area1_11[i][0]])
degree11_id = np.array(degree11_id)
for i in range(len(area1_12)):
    degree12_id.append(test_fb[area1_12[i][0]])
degree12_id = np.array(degree12_id)
for i in range(len(area1_13)):
    degree13_id.append(test_fb[area1_13[i][0]])
degree13_id = np.array(degree13_id)
for i in range(len(area1_14)):
    degree14_id.append(test_fb[area1_14[i][0]])
degree14_id = np.array(degree14_id)
for i in range(len(area1_15)):
    degree15_id.append(test_fb[area1_15[i][0]])
degree15_id = np.array(degree15_id)
for i in range(len(area1_16)):
    degree16_id.append(test_fb[area1_16[i][0]])
degree16_id = np.array(degree16_id)
for i in range(len(area1_17)):
    degree17_id.append(test_fb[area1_17[i][0]])
degree17_id = np.array(degree17_id)
for i in range(len(area1_18)):
    degree18_id.append(test_fb[area1_18[i][0]])
degree18_id = np.array(degree18_id)
for i in range(len(area1_19)):
    degree19_id.append(test_fb[area1_19[i][0]])
degree19_id = np.array(degree19_id)
for i in range(len(area1_20)):
    degree20_id.append(test_fb[area1_20[i][0]])
degree20_id = np.array(degree20_id)

test_fb_full = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/fb15k-237/test.tsv", sep = "\s+", names = [0,1,2], dtype=str)
test_fb_full = np.array(test_fb_full)

degree0 = []
degree1, degree2, degree3, degree4, degree5 = [], [], [], [], []
degree6, degree7, degree8, degree9, degree10 = [], [], [], [], []
degree11, degree12, degree13, degree14, degree15 = [], [], [], [], []
degree16, degree17, degree18, degree19, degree20 = [], [], [], [], []

for i in range(len(area0_1)):
    degree0.append(test_fb_full[area0_1[i][0]])
degree0 = np.array(degree0)
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
for i in range(len(area1_11)):
    degree11.append(test_fb_full[area1_11[i][0]])
degree11 = np.array(degree11)
for i in range(len(area1_12)):
    degree12.append(test_fb_full[area1_12[i][0]])
degree12 = np.array(degree12)
for i in range(len(area1_13)):
    degree13.append(test_fb_full[area1_13[i][0]])
degree13 = np.array(degree13)
for i in range(len(area1_14)):
    degree14.append(test_fb_full[area1_14[i][0]])
degree14 = np.array(degree14)
for i in range(len(area1_15)):
    degree15.append(test_fb_full[area1_15[i][0]])
degree15 = np.array(degree15)
for i in range(len(area1_16)):
    degree16.append(test_fb_full[area1_16[i][0]])
degree16 = np.array(degree16)
for i in range(len(area1_17)):
    degree17.append(test_fb_full[area1_17[i][0]])
degree17 = np.array(degree17)
for i in range(len(area1_18)):
    degree18.append(test_fb_full[area1_18[i][0]])
degree18 = np.array(degree18)
for i in range(len(area1_19)):
    degree19.append(test_fb_full[area1_19[i][0]])
degree19 = np.array(degree19)
for i in range(len(area1_20)):
    degree20.append(test_fb_full[area1_20[i][0]])
degree20 = np.array(degree20)

# degree - entity 수
area0_1_h_id = area0_1[:,0]
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
area1_11_h_id = area1_11[:,0]
area1_12_h_id = area1_12[:,0]
area1_13_h_id = area1_13[:,0]
area1_14_h_id = area1_14[:,0]
area1_15_h_id = area1_15[:,0]
area1_16_h_id = area1_16[:,0]
area1_17_h_id = area1_17[:,0]
area1_18_h_id = area1_18[:,0]
area1_19_h_id = area1_19[:,0]
area1_20_h_id = area1_20[:,0]

area0_1_h = []
area1_1_h, area1_2_h, area1_3_h, area1_4_h, area1_5_h = [], [], [], [], []
area1_6_h, area1_7_h, area1_8_h, area1_9_h, area1_10_h = [], [], [], [], []
area1_11_h, area1_12_h, area1_13_h, area1_14_h, area1_15_h = [], [], [], [], []
area1_16_h, area1_17_h, area1_18_h, area1_19_h, area1_20_h = [], [], [], [], []
tmp = test_fb_full[:,0]
for i in range(len(area0_1_h_id)): 
    area0_1_h.append(tmp[area0_1_h_id[i]])
area0_1_h = set(area0_1_h)
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
for i in range(len(area1_11_h_id)): 
    area1_11_h.append(tmp[area1_11_h_id[i]])
area1_11_h = set(area1_11_h)
for i in range(len(area1_12_h_id)): 
    area1_12_h.append(tmp[area1_12_h_id[i]])
area1_12_h = set(area1_12_h)
for i in range(len(area1_13_h_id)): 
    area1_13_h.append(tmp[area1_13_h_id[i]])
area1_13_h = set(area1_13_h)
for i in range(len(area1_14_h_id)): 
    area1_14_h.append(tmp[area1_14_h_id[i]])
area1_14_h = set(area1_14_h)
for i in range(len(area1_15_h_id)): 
    area1_15_h.append(tmp[area1_15_h_id[i]])
area1_15_h = set(area1_15_h)
for i in range(len(area1_16_h_id)): 
    area1_16_h.append(tmp[area1_16_h_id[i]])
area1_16_h = set(area1_16_h)
for i in range(len(area1_17_h_id)): 
    area1_17_h.append(tmp[area1_17_h_id[i]])
area1_17_h = set(area1_17_h)
for i in range(len(area1_18_h_id)): 
    area1_18_h.append(tmp[area1_18_h_id[i]])
area1_18_h = set(area1_18_h)
for i in range(len(area1_19_h_id)): 
    area1_19_h.append(tmp[area1_19_h_id[i]])
area1_19_h = set(area1_19_h)
for i in range(len(area1_20_h_id)): 
    area1_20_h.append(tmp[area1_20_h_id[i]])
area1_20_h = set(area1_20_h)

weight1_1 = [len(area1_1_h_id), len(area1_2_h_id), len(area1_3_h_id), len(area1_4_h_id), len(area1_5_h_id),
             len(area1_6_h_id), len(area1_7_h_id), len(area1_8_h_id), len(area1_9_h_id), len(area1_10_h_id),
             len(area1_11_h_id), len(area1_12_h_id), len(area1_13_h_id), len(area1_14_h_id), len(area1_15_h_id),
             len(area1_16_h_id), len(area1_17_h_id), len(area1_18_h_id), len(area1_19_h_id), len(area1_20_h_id)]
weight1_2 = [len(area1_1_h), len(area1_2_h), len(area1_3_h), len(area1_4_h), len(area1_5_h),
             len(area1_6_h), len(area1_7_h), len(area1_8_h), len(area1_9_h), len(area1_10_h),
             len(area1_11_h), len(area1_12_h), len(area1_13_h), len(area1_14_h), len(area1_15_h),
             len(area1_16_h), len(area1_17_h), len(area1_18_h), len(area1_19_h), len(area1_20_h)]

print("Degree - Triple:", weight1_1)
print("Degree - Entity:",weight1_2)
