import numpy as np
import pandas as pd
import torch
import json
import ast

# Replace this with the path to your text file
file_path = 'C:/Users/USER/Desktop/id.txt'

with open(file_path, 'r') as file:
    text = file.read()
    my_dict_e = ast.literal_eval(text)
    
fb_en = list(my_dict_e.keys())
fb_en_id = list(my_dict_e.values())

fb_en, fb_en_id = np.array(fb_en), np.array(fb_en_id)
fb_test_e = np.c_[fb_en,fb_en_id]

#---------------------------------------------------------------------------------#
## relation id 불러오기
with open('C:/Users/USER/Desktop/id_r_fb.txt', 'r') as file:
    text = file.read()
    my_dict_r = ast.literal_eval(text)
    
fb_rel = list(my_dict_r.keys())
fb_rel_id = list(my_dict_r.values())

fb_rel, fb_rel_id = np.array(fb_rel), np.array(fb_rel_id)
fb_test_r = np.c_[fb_rel,fb_rel_id]
