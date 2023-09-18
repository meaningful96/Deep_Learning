import json

class CustomLinkGraph:

    def __init__(self, train_path: str):  

        print("Start to build link graph!!")
        self.graph = {}
        examples = json.load(open(train_path, 'r', encoding='utf-8'))
        for ex in examples: 
            head, head_id, relation, tail, tail_id = ex['head'], ex['head_id'], ex['relation'], ex['tail'], ex['tail_id']
            if head_id not in self.graph:
                self.graph[head_id] = []


            info = {
                'head': head,
                'relation': relation,
                'tail': tail,
                'tail_id': tail_id
            }
            self.graph[head_id].append(info)

        print("Done build link graph with {} nodes".format(len(self.graph)))

    def get_neighbor_ids(self, entity_id: str, max_to_keep=10):
        neighbor_ids = self.graph.get(entity_id, [])
        return neighbor_ids[:max_to_keep]

def check_duplicate(train_path:str):
    print("="*30)
    print("Check the duplication of 'hr'")
    duplicate = {}
    examples = json.load(open(train_path, 'r', encoding='utf-8'))
    for ex in examples:
        head_id, relation = ex['head_id'], ex['relation']
        if head_id not in duplicate:
            duplicate[head_id] = set()
        duplicate[head_id].add(relation)
    return duplicate

def count_original_example(train_path:str) -> int:
    print("="*30)
    print("Checking the number of examples")
    examples = json.load(open(train_path, 'r', encoding='utf-8'))
    print(len(examples))
    
def count_unique_examples(train_path:str) -> int:
    print("="*30)
    print("Checking for unique 'head' and 'relation' combinations")
    unique_examples = set()
    examples = json.load(open(train_path, 'r', encoding='utf-8'))
    for ex in examples:
        head_id, relation = ex['head_id'], ex['relation']
        unique_examples.add((head_id, relation))
    print(len(unique_examples))
    return len(unique_examples)

train_path_wn = "C:/Users/USER/Desktop/json/WN18RR/train.txt.json"
train_path_fb = "C:/Users/USER/Desktop/json/FB15k237/train.txt.json"
# WN18RR = CustomLinkGraph(train_path=train_path_wn)
# FB15k237 = CustomLinkGraph(train_path=train_path_fb)

check_wn = check_duplicate(train_path_wn)
truth_wn = count_original_example(train_path_wn)
count_wn = count_unique_examples(train_path_wn)
print("~"*59)
check_fb = check_duplicate(train_path_fb)
truth_fb = count_original_example(train_path_fb)
count_fb = count_unique_examples(train_path_fb)



