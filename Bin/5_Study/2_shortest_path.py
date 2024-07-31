import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import plotly.graph_objects as go
# Step 1) Data Load

#-----------------------------------------------------------------------------------#
# UMLS
"""
train_umls = pd.read_csv("C:/Users/PC/Desktop/dataset/umls/get_neighbor/train2id.txt", sep = "\s+", names = [0,1,2])
train_umls = np.array(train_umls)
umls_h = train_umls[:,0]
umls_r = train_umls[:,1]
umls_t = train_umls[:,2]

test_umls = pd.read_csv("C:/Users/PC/Desktop/dataset/umls/get_neighbor/test2id.txt", sep = "\s+", names = [0,1,2])
test_umls = np.array(test_umls)
umls_h_test = test_umls[:,0]
umls_r_test = test_umls[:,1]
umls_t_test = test_umls[:,2]
#"""

# WN18RR

"""

train_wn = pd.read_csv("C:/Users/PC/Desktop/dataset/wn18rr/get_neighbor/train2id.txt", sep = "\s+", names = [0,1,2])
train_wn = np.array(train_wn)
wn_h = train_wn[:,0]
wn_r = train_wn[:,1]
wn_t = train_wn[:,2]

test_wn = pd.read_csv("C:/Users/PC/Desktop/dataset/wn18rr/get_neighbor/test2id.txt", sep = "\s+", names = [0,1,2])
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
"""

# WN18RR

# """
train_fb = pd.read_csv("C:/Users/PC/Desktop/dataset/fb15k-237/get_neighbor/train2id.txt", sep = "\s+", names = [0,1,2])
train_fb = np.array(train_fb)
fb_h = train_fb[:,0]
fb_r = train_fb[:,1]
fb_t = train_fb[:,2]

test_fb = pd.read_csv("C:/Users/PC/Desktop/dataset/fb15k-237/get_neighbor/test2id.txt", sep = "\s+", names = [0,1,2])
test_fb = np.array(test_fb)
fb_h_test = test_fb[:,0]
fb_r_test = test_fb[:,1]
fb_t_test = test_fb[:,2]

entities_test = np.r_[fb_h_test, fb_t_test]
entities_test = set(sorted(entities_test))
entities_train = np.r_[fb_h, fb_t]
entities_train = set(entities_train)
non_list = entities_test - entities_train





#%%
"""
## Step 2) Graph Construction
#-----------------------------------------------------------------------------------#

## umls
G_umls = nx.Graph()

for head, relation, tail in zip(umls_h, umls_r, umls_t):
    G_umls.add_edge(head, tail, relation=relation)

print("Total relation number of UMLS:",len(G_umls.edges))
print("Varience of relation in UMLS:", len(umls_r) - len(G_umls.edges))
print("-"*57)
"""
#-----------------------------------------------------------------------------------#


"""
## wn18rr
G_wn = nx.Graph()

for head, relation, tail in zip(wn_h, wn_r, wn_t):
    G_wn.add_edge(head, tail, relation=relation)

G_wn.add_nodes_from(non_list)
print("Total relation number of WN18RR:",len(G_wn.edges))
print("Varience of relation in WN18RR:", len(wn_r) - len(G_wn.edges))
print("Total number of Graph Entites:", len(G_wn.nodes))
print("----------------------------------------------------")
"""

#-----------------------------------------------------------------------------------#
# """
## fb15k-237
G_fb = nx.Graph()

for head, relation, tail in zip(fb_h, fb_r, fb_t):
    G_fb.add_edge(head, tail, relation=relation)


G_fb.add_nodes_from(non_list)
print("Total relation number of FB15k-237:",len(G_fb.edges))
print("Varience of relation in FB15k-237:", len(fb_r) - len(G_fb.edges))
print("----------------------------------------------------")
# """
#-----------------------------------------------------------------------------------#

#%%

## Step 3) BFS
def BFS(Graph, Start_node, End_node):
    visited = set() # 방문한 노드를 저장할 빈 set을 만듬
    queue = deque([Start_node]) # 새로운 quere를 만들고 Start_node를 Enqueue해 추가함.
    distance = {Start_node : 0} # Start node에서 visited 사이에 저장된 노드들의 거리
                                # print(len(distance)) 하면 거리 출력
    shortest_path = []
    # print("Initial Values!!!!")
    # print(visited)
    # print(queue)
    # print(len(distance))
    # print("-"*60)           
    # print("Algorithm Start!!!!")
    
    while queue: # queue에 노드가 있으면 계속 진행 
        node = queue.popleft() # queue에 있는 노드 삭제
        
        if node == End_node: # 만약 삭제된 노드가 End_node이면 종료 
            return distance[node]
        
        if node not in visited: # 만약 노드를 아직 방문하지 않았다면
            visited.add(node) # visited에 마킹
            
            for neighbor in Graph.neighbors(node): # Go through all neighbors of this node
                if neighbor not in visited and neighbor not in queue: # 만약 방문하지 않은 이웃이 있다면
                    queue.append(neighbor) # queue에 이웃 추가
                    distance[neighbor] = distance[node] + 1 # 시작노드에서 이웃 노드로의 거리 계산
                    # print(distance)
                    # print("Path Length is:", max(distance.values()))
                shortest_path.append(max(distance.values()))
        shortest_path = list(set(shortest_path))
    if not shortest_path:
        shortest_path.append(0)
    
        # print("="*29, "Result", "="*29)
        # print("Shortest Path Length is: " ,shortest_path[len(shortest_path) - 1])
        
    
    # If the end node is not reachable from the start node, return None
    return max(shortest_path)   
#%%

import pickle

## Step 4) Experiment       
"""

## UMLS
Same_head_tail = []
for i in range(len(umls_h_test)):
    for j in range(len(umls_h)):
        if umls_h_test[i] == umls_h[j] and umls_t_test[i] == umls_t[j]:
            Same_head_tail.append(i)
        
Same_head_tail = set(Same_head_tail)
print("Triple에서 head와 tail이 같은 수: ", len(Same_head_tail))
print("-"*57)

Shortest_Path = []
index = []
for i in range(len(umls_h_test)):
    result = BFS(G_umls, umls_h_test[i], umls_t_test[i])
    Shortest_Path.append(result)
    index.append((i,result))

index = np.array(index)
    
for i in range(max(Shortest_Path) + 1):
    print(i,"는 총: ", Shortest_Path.count(i))                         
print("-"*26, "END", "-"*26)    

um_hop1 = []
um_hop2 = []

for i in range(len(umls_h_test)):
    if index[i][1] == 1:
        um_hop1.append(test_umls[i])
    if index[i][1] == 2:
        um_hop2.append(test_umls[i])
print(len(um_hop1) + len(um_hop2))        

um_hop1 = np.array(um_hop1)
um_hop2 = np.array(um_hop2)

"""

"""

## wn18rr
Same_head_tail = []
for i in range(len(wn_h_test)):
    for j in range(len(wn_h)):
        if wn_h_test[i] == wn_h[j] and wn_t_test[i] == wn_t[j]:
            Same_head_tail.append(i)
        
Same_head_tail = set(Same_head_tail)
print("Triple에서 head와 tail이 같은 수: ", len(Same_head_tail))
print("-"*57)

Shortest_Path = []
index = []
for i in range(len(wn_h_test)):
    result = BFS(G_wn, wn_h_test[i], wn_t_test[i])
    Shortest_Path.append(result)
    index.append((i,result))

index = np.array(index)
    
for i in range(max(Shortest_Path) + 1):
    print("Shortest Path가",i,"hop인 총 개수: ", Shortest_Path.count(i))                         
print("-"*26, "END", "-"*26)    

wn_hop1 = []
wn_hop2 = []
wn_hop3 = []
wn_hop4 = []
wn_hop5 = []
wn_hop6 = []

for i in range(len(wn_h_test)):
    if index[i][1] == 1:
        wn_hop1.append(test_wn[i])
    if index[i][1] == 2:
        wn_hop2.append(test_wn[i])
    if index[i][1] == 3:
        wn_hop3.append(test_wn[i])        
    if index[i][1] == 4:
        wn_hop4.append(test_wn[i])
    if index[i][1] == 5:
        wn_hop5.append(test_wn[i])
    if index[i][1] == 6:
        wn_hop6.append(test_wn[i])
print(len(wn_hop1) + len(wn_hop2) + len(wn_hop3) + len(wn_hop4) + len(wn_hop5) + len(wn_hop6))      

wn_hop1 = np.array(wn_hop1)
wn_hop2 = np.array(wn_hop2)
wn_hop3 = np.array(wn_hop3)
wn_hop4 = np.array(wn_hop4)
wn_hop5 = np.array(wn_hop5)
wn_hop6 = np.array(wn_hop6)

"""

# """

## FB15k-237
Same_head_tail = []
for i in range(len(fb_h_test)):
    for j in range(len(fb_h)):
        if fb_h_test[i] == fb_h[j] and fb_t_test[i] == fb_t[j]:
            Same_head_tail.append(i)
        
Same_head_tail = set(Same_head_tail)
print("Triple에서 head와 tail이 같은 수: ", len(Same_head_tail))
print("-"*57)

Shortest_Path = []
index = []
for i in range(len(fb_h_test)):
    result = BFS(G_fb, fb_h_test[i], fb_t_test[i])
    Shortest_Path.append(result)
    index.append((i,result))

index = np.array(index)
    
for i in range(max(Shortest_Path) + 1):
    print(i,"는 총: ", Shortest_Path.count(i))                         
print("-"*26, "END", "-"*26)    

fb_hop1 = []
fb_hop2 = []
fb_hop3 = []
fb_hop4 = []
fb_hop5 = []
fb_hop6 = []

for i in range(len(fb_h_test)):
    if index[i][1] == 1:
        fb_hop1.append(test_fb[i])
    if index[i][1] == 2:
        fb_hop2.append(test_fb[i])
    if index[i][1] == 3:
        fb_hop3.append(test_fb[i])        
    if index[i][1] == 4:
        fb_hop4.append(test_fb[i])
    if index[i][1] == 5:
        fb_hop5.append(test_fb[i])
    if index[i][1] == 6:
        fb_hop6.append(test_fb[i])
print(len(fb_hop1) + len(fb_hop2) + len(fb_hop3) + len(fb_hop4) + len(fb_hop5) + len(fb_hop6))      
             


fb_hop1 = np.array(fb_hop1)
fb_hop2 = np.array(fb_hop2)
fb_hop3 = np.array(fb_hop3)
fb_hop4 = np.array(fb_hop4)
fb_hop5 = np.array(fb_hop5)
fb_hop6 = np.array(fb_hop6)
# """

####################################################################################################
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
fb_1_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/fb15k-237/test2id/test2id_1.txt", sep = '\s+', names = [0,1,2])
fb_2_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/fb15k-237/test2id/test2id_2.txt", sep = '\s+', names = [0,1,2])
fb_3_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/fb15k-237/test2id/test2id_3.txt", sep = '\s+', names = [0,1,2])
fb_4_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/fb15k-237/test2id/test2id_4.txt", sep = '\s+', names = [0,1,2])
fb_5_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/fb15k-237/test2id/test2id_5.txt", sep = '\s+', names = [0,1,2])
fb_6_hop = pd.read_csv("C:/Users/PC/Desktop/Testing/fb15k-237/test2id/test2id_6.txt", sep = '\s+', names = [0,1,2])
fb_1_hop = np.array(fb_1_hop)
fb_2_hop = np.array(fb_2_hop)
fb_3_hop = np.array(fb_3_hop)
fb_4_hop = np.array(fb_4_hop)
fb_5_hop = np.array(fb_5_hop)
fb_6_hop = np.array(fb_6_hop)

test_fb = pd.read_csv("C:/Users/PC/Desktop/dataset/fb15k-237/test.tsv", sep = '\s+', names = [0,1,2])
test2id = pd.read_csv("C:/Users/PC/Desktop/dataset/fb15k-237//get_neighbor/test2id.txt", sep =  "\s+", names=[0,1,2])
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
