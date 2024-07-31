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
"""

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
# """

# FB15k-237

"""
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

"""

#%%

## Step 2) Graph Construction
#-----------------------------------------------------------------------------------#
"""
## umls
G_umls = nx.Graph()

for head, relation, tail in zip(umls_h, umls_r, umls_t):
    G_umls.add_edge(head, tail, relation=relation)

print("Total relation number of UMLS:",len(G_umls.edges))
print("Varience of relation in UMLS:", len(umls_r) - len(G_umls.edges))
print("-"*57)
# """
#-----------------------------------------------------------------------------------#


# """
## wn18rr
G_wn = nx.Graph()

for head, relation, tail in zip(wn_h, wn_r, wn_t):
    G_wn.add_edge(head, tail, relation=relation)

G_wn.add_nodes_from(non_list)
print("Total relation number of WN18RR:",len(G_wn.edges))
print("Varience of relation in WN18RR:", len(wn_r) - len(G_wn.edges))
print("Total number of Graph Entites:", len(G_wn.nodes))
print("----------------------------------------------------")
# """

#-----------------------------------------------------------------------------------#
"""
## fb15k-237
G_fb = nx.Graph()

for head, relation, tail in zip(fb_h, fb_r, fb_t):
    G_fb.add_edge(head, tail, relation=relation)


G_fb.add_nodes_from(non_list)
print("Total relation number of FB15k-237:",len(G_fb.edges))
print("Varience of relation in FB15k-237:", len(fb_r) - len(G_fb.edges))
print("----------------------------------------------------")
"""
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
       
        if 7 in distance.values():  # Check if 7 is in the distance values
            break
        
    if not shortest_path:
        shortest_path.append(0)
    
        # print("="*29, "Result", "="*29)
        # print("Shortest Path Length is: " ,shortest_path[len(shortest_path) - 1])
        
    
    # If the end node is not reachable from the start node, return None
    return max(shortest_path)   
#%
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

# """

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

# """

"""

## FB15k-237
Same_head_tail = []
for i in range(len(fb_h_test)):
    for j in range(len(fb_h)):
        if fb_h_test[i] == fb_h[j] and fb_t_test[i] == fb_t[j]:
            Same_head_tail.append(i)
        
Same_head_tail = set(Same_head_tail)
print("Triple에서 head와 tail이 같은 수: ", len(Same_head_tail))
print("-"*57)

print("Algorithm Start!!!")
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
"""