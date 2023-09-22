import json
from typing import List, Dict, Tuple
from collections import defaultdict, deque
import random
import time
import datetime

class LinkGraph:
    def __init__(self, train_path: str):
        print('Start to build link graph!!!')
        # id -> {(relation, id)}
        self.graph = defaultdict(set)
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        
        for ex in examples:
            head_id, head, relation, tail_id, tail = ex['head_id'], ex['head'], ex['relation'], ex['tail_id'], ex['tail']
            self.graph[head_id].add((head_id, head, relation, tail_id, tail))
        print('Done building link graph with {} nodes'.format(len(self.graph)))

    def get_neighbor_ids(self, tail_id: str, n_hops: int) -> List[str]:
        if n_hops <= 0:
            return []

        # Fetch immediate neighbors for the given tail_id
        neighbors = [item[3] for item in self.graph.get(tail_id, set())]

        # List to collect neighbors found in subsequent hops
        distant_neighbors = []

        # Use recursion to fetch neighbors of neighbors
        for neighbor in neighbors:
            distant_neighbors.extend(self.get_neighbor_ids(neighbor, n_hops-1))

        # Return unique neighbor IDs by converting to set and then back to list
        return list(set(neighbors + distant_neighbors))

    def bfs(self, start: str) -> Dict[str, int]:
        visited = {}
        queue = deque([(start, 0)])

        while queue:
            node, depth = queue.popleft()

            if node not in visited:
                visited[node] = depth
                for neighbor in self.graph.get(node, []):
                    queue.append((neighbor[3], depth + 1))

        return visited
        
    def create_positive_samples(self, head_id: str, tail_id: str, tail_hops: int, max_samples: int) -> List[Dict[str, any]]:
        link_graph = self.graph
    
        # Exclude tail entity and its neighbors determined by tail_hops.
        exclude_ids = self.get_neighbor_ids(tail_id, tail_hops)
        exclude_ids.append(tail_id)
        exclude_ids.append(head_id)
    
        # Calculate the hop distance for all entities starting from the head entity.
        hop_distances = self.bfs(head_id)
    
        """
        # On average, for WN18RR, hard negative samples with a batch size of 256 fall within a 6-hop boundary. 
        # Similarly, samples from FB15K-237 with a batch size of 1024 encompass nearly all triples within 5 hops.
        """
    
        # Sort entities based on their proximity to the head entity.
        sorted_entities = sorted(hop_distances.keys(), key=lambda k: hop_distances[k])

        results = []
        sample_count = 0
    
        for entity in sorted_entities:
            # Stop if we've collected the desired number of samples.
            if sample_count >= max_samples:
                break

            # Retrieve triples that are associated with the current entity.
            for triple in link_graph.get(entity, []):
                # Add the triple to the sample list if the tail is not in the exclude_ids.
                if triple[0] and triple[3] not in exclude_ids:
                    results.append({
                        "head_id": triple[0],
                        "head": triple[1],
                        "relation": triple[2],
                        "tail_id": triple[3],
                        "tail": triple[4]
                        })
                    sample_count += 1
                
                    # Stop if we've collected the desired number of samples.
                    if sample_count >= max_samples:
                        break

        return results 
        
        
start_time = time.time()    
train_path_fb = "C:/Users/USER/Desktop/json/FB15k237/train.txt.json"
G_fb = LinkGraph(train_path_fb)

with open(train_path_fb, 'r', encoding='utf-8') as f:
    train_data_wn = json.load(f)

# A list to store the results
all_positive_samples = []

# Iterate over the WordNet train set
for example in train_data_wn:
    head_id = example['head_id']
    tail_id = example['tail_id']
    
    # Create positive samples
    positive_samples = G_fb.create_positive_samples(head_id, tail_id, 1, 1023)
    positive_samples.insert(0, example)
    # Add the results to the list
    all_positive_samples.append(positive_samples)

# Save the combined results to a JSON file
output_path = "/home/youminkk/Paper_reconstruction/SimKGC_HardNegative/data/FB15k237/positive_samples_valid.json"
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_positive_samples, f, ensure_ascii=False, indent=4)

end_time = time.time()
sec = end_time - start_time

print("Total Length",len(all_positive_samples))
print("Positive samples of FB15K-237 saved")
print("Taking time:", datetime.timedelta(seconds = sec))
print("Done!!")

