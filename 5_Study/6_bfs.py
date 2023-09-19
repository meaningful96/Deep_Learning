import json
from typing import List, Dict, Tuple
from collections import deque, defaultdict
import requests
from collections import deque

class LinkGraph:
    def __init__(self, train_path: str):
        print('Start to build link graph!!!')
        # id -> {(relation, id)}
        self.graph = defaultdict(set)
        
        # Fetch the JSON from the URL
        response = requests.get(train_path)
        response.raise_for_status()  # Will raise an exception if the HTTP request returned an error
        examples = response.json()
        
        for ex in examples:
            head_id, tail_id, relation = ex['head_id'], ex['tail_id'], ex['relation']
            self.graph[head_id].add((relation, tail_id))
            """
            Because of the Bi-Encoder with BERT, the embeddings of the triple can be learned
            as the undirected graph automatically.
            """
        print('Done building link graph with {} nodes'.format(len(self.graph)))

    def get_neighbor_ids(self, entity_id: str, max_to_keep=1024) -> List[str]:
        # make sure different calls return the same results
        neighbor_ids = self.graph.get(entity_id, set())
        
        """
        It returns the neighbors as tuple such as (r1, t1). Also all the neighbor tuple are stored in the list
        return: 'h1' neighbor: [(r2,t2), (r3,t3), (r4, t4)]
        """
        return sorted(list(neighbor_ids))[:max_to_keep]    

def excluded_entity(graph, tail_id: str, n_hops: int) -> List[str]:
    visited = set() # 방문한 엔티티를 표시하기 위한 집합
    visited.add(tail_id) # tail포함
    queue = deque([(tail_id, 0)]) # BFS를 위한 큐 (엔티티, 현재 깊이)
    while queue:
        current_entity, depth = queue.popleft()
        if depth == n_hops: # n-hop 까지만 탐색
            break
        neighbors = graph.get_neighbor_ids(current_entity)
        for _, neighbor_id in neighbors:
            if neighbor_id not in visited: # 아직 방문하지 않은 이웃만 큐에 추가
                queue.append((neighbor_id, depth + 1))
                visited.add(neighbor_id)
    return list(visited)


        
        
        
train_path_wn = 'https://raw.githubusercontent.com/meaningful96/Deep_Learning/main/1_DataSet/Banchmark/WN18RR/train.txt.json'
graph = LinkGraph(train_path_wn)
graph.get_neighbor_ids('00260881')

excluded_entity(graph,"00260881", 2)

