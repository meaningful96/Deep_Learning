import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

# Step 1) DataLoad
train_wn = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/wn18rr/train.tsv", sep = "\s+")
test_wn = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/wn18rr/test.tsv", sep = "\s+")

train_fb = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/fb15k-237/train.tsv", sep = "\s+")
test_fb = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/fb15k-237/test.tsv", sep = "\s+")

train_wn, test_wn = np.array(train_wn), np.array(test_wn)
train_fb, test_fb = np.array(train_fb), np.array(test_fb)

#------------------------------------------------------------------#

# Step 2) Triple Detaching
## wn18rr
head_wn = train_wn[:,0]
relation_wn = train_wn[:,1]
tail_wn = train_wn[:,2]

## fb15k-237
head_fb = train_fb[:,0]
relation_fb = train_fb[:,1]
tail_fb = train_fb[:,2]

#------------------------------------------------------------------#
# Step 3) Constructing the Graph

## wn18rr
G_wn = nx.Graph() #nx.DiGraph()면 directed Graph가 생성됨.

for head, relation, tail in zip(head_wn, relation_wn, tail_wn):
    G_wn.add_edge(head, tail, relation=relation)

print("Total relation number of WN18RR:",len(G_wn.edges))
print("Varience of relation in WN18RR:", len(relation_wn) - len(G_wn.edges))
print("----------------------------------------------------")
## fb15k-237
G_fb = nx.Graph()

for head, relation, tail in zip(head_fb, relation_fb, tail_fb):
    G_fb.add_edge(head, tail, relation=relation)

print("Total relation number of FB15K-237:",len(G_fb.edges))
print("Varience of relation in FB15K-237:", len(relation_fb) - len(G_fb.edges))
#------------------------------------------------------------------#

# Step 4) BFS: 너비 우선 탐색
def BFS_1(graph, start):
    visited = []
    queue = [start]
    
    while queue:
        node = queue.pop(0)
        if node not in visited:
            visited.append(node)
            neighbours = graph[node]
            for neighbour in neighbours:
                queue.append(neighbour)
    return visited

## 그래프, 시작, 끝
def BFS_2(graph, start, end):
    visited = []
    queue = [start]
    
    while queue:
        node = queue.pop(0)
        
        if node == end:
            print(f"End node {end} found!")
            return True            
        
        if node not in visited:
            visited.append(node)
            neighbours = graph[node]
            for neighbour in neighbours:
                queue.append(neighbour)
                
    print("End node not found in the graph.")
    return False
  

## 이제 문제는 특정 Start Point node에서 시작해서 특정 End point가 나왔을 때 알고리즘을 종료(break)하고
## 지금까지 visited에 쌓인 리스트 길이가 몇인지를 BFS와 DFS동시에 비교해 shortest path를 찾아야 한다.

