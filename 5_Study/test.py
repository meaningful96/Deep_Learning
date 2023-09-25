import json
import torch
import torch.utils.data.dataset 
from typing import List, Dict

"""
path = "/home/youminkk/Paper_reconstruction/SimKGC_HardNegative/data/WN18RR/positive_samples_train.json"

with open(path, 'r', encoding='utf-8') as f:
    examples = json.load(f)           

dict_total = []
for index, sublist in enumerate(examples):
    if len(sublist) != 256:
        dict_total.append((index, sublist))




print(len(dict_total))
"""

path = "/home/youminkk/Paper_reconstruction/SimKGC_HardNegative/data/WN18RR/positive_samples_valid.json"

with open(path, 'r', encoding='utf-8') as f:
    examples = json.load(f)           

dict_total = []
for index, sublist in enumerate(examples):
    if len(sublist) != 256:
        dict_total.append((index, sublist))




print(len(dict_total))
