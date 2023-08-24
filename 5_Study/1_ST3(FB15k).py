import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# FB15k-237

# """
train_fb = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/fb15k-237/get_neighbor/train2id.txt", sep = "\s+", names = [0,1,2])
train_fb = np.array(train_fb)
fb_h = train_fb[:,0]
fb_r = train_fb[:,1]
fb_t = train_fb[:,2]

test_fb = pd.read_csv("https://raw.githubusercontent.com/zjunlp/Relphormer/main/dataset/fb15k-237/get_neighbor/test2id.txt", sep = "\s+", names = [0,1,2])
test_fb = np.array(test_fb)
fb_h_test = test_fb[:,0]
fb_r_test = test_fb[:,1]
fb_t_test = test_fb[:,2]

entities_test = np.r_[fb_h_test, fb_t_test]
entities_test = set(sorted(entities_test))
entities_train = np.r_[fb_h, fb_t]
entities_train = set(entities_train)
non_list = entities_test - entities_train

# """

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

## Step 3) BFS
# def BFS(Graph, Start_node, End_node):
#     visited = set() # 방문한 노드를 저장할 빈 set을 만듬
#     queue = deque([Start_node]) # 새로운 quere를 만들고 Start_node를 Enqueue해 추가함.
#     distance = {Start_node : 0} # Start node에서 visited 사이에 저장된 노드들의 거리
#                                 # print(len(distance)) 하면 거리 출력
#     shortest_path = []
#     # print("Initial Values!!!!")
#     # print(visited)
#     # print(queue)
#     # print(len(distance))
#     # print("-"*60)
#     # print("Algorithm Start!!!!")

#     while queue: # queue에 노드가 있으면 계속 진행
#         node = queue.popleft() # queue에 있는 노드 삭제

#         if node == End_node: # 만약 삭제된 노드가 End_node이면 종료
#             return distance[node]

#         if node not in visited: # 만약 노드를 아직 방문하지 않았다면
#             visited.add(node) # visited에 마킹

#             for neighbor in Graph.neighbors(node): # Go through all neighbors of this node
#                 if neighbor not in visited and neighbor not in queue: # 만약 방문하지 않은 이웃이 있다면
#                     queue.append(neighbor) # queue에 이웃 추가
#                     distance[neighbor] = distance[node] + 1 # 시작노드에서 이웃 노드로의 거리 계산
#                     # print(distance)
#                     # print("Path Length is:", max(distance.values()))
#                 shortest_path.append(max(distance.values()))
#         shortest_path = list(set(shortest_path))
#     if not shortest_path:
#         shortest_path.append(0)

#         # print("="*29, "Result", "="*29)
#         # print("Shortest Path Length is: " ,shortest_path[len(shortest_path) - 1])


#     # If the end node is not reachable from the start node, return None
#     return max(shortest_path)


## Step 3) 
def find_shortest_path_length(graph, start_node, end_node):
    try:
        shortest_path = nx.shortest_path(graph, source=start_node, target=end_node)
        shortest_path_length = nx.shortest_path_length(graph, source=start_node, target=end_node)
        return shortest_path_length
    except nx.NetworkXNoPath:
        return 0
    
    
import pickle


## FB15k-237
# Same_head_tail = []
# for i in range(len(fb_h_test)):
#     for j in range(len(fb_h)):
#         if fb_h_test[i] == fb_h[j] and fb_t_test[i] == fb_t[j]:
#             Same_head_tail.append(i)

# Same_head_tail = set(Same_head_tail)
# print("Triple에서 head와 tail이 같은 수: ", len(Same_head_tail))
# print("-"*57)

print("Algorithm Start!!!!!!!!")
Shortest_Path = []
index = []
for i in range(len(fb_h_test)):
    result = find_shortest_path_length(G_fb, fb_h_test[i], fb_t_test[i])
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

file_path1 = '/home/youminkk/fb_hop1.txt'
file_path2 = '/home/youminkk/fb_hop2.txt'
file_path3 = '/home/youminkk/fb_hop3.txt'
file_path4 = '/home/youminkk/fb_hop4.txt'
file_path5 = '/home/youminkk/fb_hop5.txt'
file_path6 = '/home/youminkk/fb_hop6.txt'
np.savetxt(file_path2, fb_hop2, fmt='%d', delimiter='\t')
np.savetxt(file_path3, fb_hop3, fmt='%d', delimiter='\t')
np.savetxt(file_path4, fb_hop4, fmt='%d', delimiter='\t')
np.savetxt(file_path5, fb_hop5, fmt='%d', delimiter='\t')
np.savetxt(file_path6, fb_hop6, fmt='%d', delimiter='\t')
np.savetxt(file_path1, fb_hop1, fmt='%d', delimiter='\t')
print(f"The array has been successfully saved to {file_path6}.")
