import json
import time
import datetime

class LinkGraph:
    def __init__(self, train_path: str, excluded_hops: int = 0):
        self.graph = {}
        self.examples = json.load(open(train_path, 'r', encoding='utf-8'))
        self.id_to_name = {}

        for ex in self.examples:
            head_id, tail_id, relation = ex['head_id'], ex['tail_id'], ex['relation']
            self.id_to_name[head_id] = ex['head']
            self.id_to_name[tail_id] = ex['tail']
            
            self.graph.setdefault(head_id, {})[tail_id] = relation
            self.graph.setdefault(tail_id, {})[head_id] = relation

    def get_excluded_tail_neighbors(self, tail_id: str, excluded_hops: int) -> set:
        excluded_neighbors = {tail_id}
        curr_queue = {tail_id}
        
        for _ in range(excluded_hops):
            curr_queue = {neighbor for ent in curr_queue for neighbor in self.graph.get(ent, {}).keys()}
            excluded_neighbors.update(curr_queue)

        return excluded_neighbors

    def get_batched_sample(self, example, max_samples=5000):
        head_id, tail_id = example['head_id'], example['tail_id']
        positive_samples = []
        excluded_hop = 1

        while len(positive_samples) < max_samples:
            excluded_neighbors = self.get_excluded_tail_neighbors(tail_id, excluded_hop)
            
            neighbors_of_head = self.graph.get(head_id, {})

            for neighbor, relation in neighbors_of_head.items():
                if neighbor not in excluded_neighbors:
                    positive_samples.append({
                        'head_id': head_id,
                        'relation': relation,
                        'tail_id': neighbor,
                        'head': self.id_to_name[head_id],
                        'tail': self.id_to_name[neighbor]
                    })
                    if len(positive_samples) >= max_samples:
                        return positive_samples

            excluded_hop += 1

        return positive_samples

    def id_to_entity(self, entity_id: str) -> str:
        return self.id_to_name.get(entity_id, entity_id)

    def get_all_batched_samples(self, max_samples=4999) -> dict:
        return {index: self.get_batched_sample(example, max_samples=max_samples) for index, example in enumerate(self.examples, start=1)}

def generate_single_batch(train_path: str, example_index: int, excluded_hops: int, batch_size: int):
    start_time = time.time()

    G = LinkGraph(train_path=train_path, excluded_hops=excluded_hops)
    actual_example = G.examples[example_index]
    batch = G.get_batched_sample(actual_example, max_samples=batch_size-1)

    end_time = time.time()
    elapsed_time = datetime.timedelta(seconds=(end_time - start_time))

    return batch, elapsed_time

def generate_and_save_batches(train_path: str, output_path: str, excluded_hops: int, batch_size: int):
    start_time = time.time()

    G = LinkGraph(train_path=train_path, excluded_hops=excluded_hops)
    batches = G.get_all_batched_samples(max_samples=batch_size-1)
    with open(output_path, "w", encoding='utf-8') as f:
        json.dump(batches, f, ensure_ascii=False)

    end_time = time.time()
    elapsed_time = datetime.timedelta(seconds=(end_time - start_time))

    return elapsed_time

# 경로 설정
train_path_wn = "/home/youminkk/Paper_reconstruction/SimKGC_HardNegative/data/WN18RR/train.txt.json"
train_path_fb = "/home/youminkk/Paper_reconstruction/SimKGC_HardNegative/data/FB15k237/train.txt.json"
output_path_wn = "/home/youminkk/Paper_reconstruction/SimKGC_HardNegative/data/WN18RR/WN18RR_batched_samples.json"
output_path_fb = "/home/youminkk/Paper_reconstruction/SimKGC_HardNegative/data/FB15k237/FB15k237_batched_samples.json"
example_index = 0

batch_wn, elapsed_time_wn = generate_single_batch(train_path_wn, example_index, 1, 1024)
batch_fb, elapsed_time_fb = generate_single_batch(train_path_fb, example_index, 1, 3072)

print(f"WN Batch Generation Time: {elapsed_time_wn}")
print(f"FB Batch Generation Time: {elapsed_time_fb}")

elapsed_time_wn_save = generate_and_save_batches(train_path_wn, output_path_wn, 1, 1024)
elapsed_time_fb_save = generate_and_save_batches(train_path_fb, output_path_fb, 1, 3072)

print(f"WN Batch Saving Time: {elapsed_time_wn_save}")
print(f"FB Batch Saving Time: {elapsed_time_fb_save}")
