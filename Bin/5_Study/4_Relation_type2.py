'''
for SimKGC, Needed to use the "json" file.
for Relphormer, Needed to use the "txt" file.
'''

import json
import pandas as pd
import numpy as np


## Step 1) Load datasets which are the original data and relations with their types
with open("C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/relation_label.txt", 'r') as file:
    type_wn_txt = json.load(file)
with open("C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/relation_label.txt", 'r') as file:
    type_fb_txt = json.load(file)

type_wn_txt = np.array(list(zip(type_wn_txt.keys(), type_wn_txt.values())))
type_fb_txt = np.array(list(zip(type_fb_txt.keys(), type_fb_txt.values())))

original_wn_test2id = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/wn18rr/get_neighbor/test2id.txt", sep = '\s+', names = ["head","relation","tail"])
original_fb_test2id = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/fb15k-237/get_neighbor/test2id.txt", sep = '\s+', names = ["head","relation","tail"])
original_wn_test = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/wn18rr/test.tsv", sep = '\s+', names = ["head","relation","tail"])
original_fb_test = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/fb15k-237/test.tsv", sep = '\s+', names = ["head","relation","tail"])

original_wn_test, original_wn_test2id = np.array(original_wn_test), np.array(original_wn_test2id)
original_fb_test, original_fb_test2id = np.array(original_fb_test), np.array(original_fb_test2id)


## Step 2) Load the relation2id file

relation2id_wn = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/wn18rr/get_neighbor/relation2id.txt", sep = "\s+", names = ["relation", "index"])
relation2id_fb = pd.read_csv("C:/Users/USER/Desktop/논문/연구/실험/Banchmark/fb15k-237/get_neighbor/relation2id.txt", sep = "\s+", names = ["relation", "index"])
relation2id_wn, relation2id_fb = np.array(relation2id_wn), np.array(relation2id_fb)

## mapping the relation type to the relation index
index_type_wn, index_type_fb = [], []
for i in range(len(type_wn_txt)):
    for j in range(len(relation2id_wn)):
        if relation2id_wn[i][0] == type_wn_txt[j][0]:
            index_type_wn.append((relation2id_wn[i][1], str(type_wn_txt[j][1])))

for i in range(len(type_fb_txt)):
    for j in range(len(relation2id_fb)):
        if relation2id_fb[i][0] == type_fb_txt[j][0]:
            index_type_fb.append((relation2id_fb[i][1], str(type_fb_txt[j][1])))            

index_type_wn, index_type_fb = np.array(index_type_wn), np.array(index_type_fb)

## Group the same relation types
wn_t11, wn_t1N, wn_tN1, wn_tNN = [], [], [], []
fb_t11, fb_t1N, fb_tN1, fb_tNN = [], [], [], []
for i in range(len(index_type_wn)):
    if index_type_wn[i][1] == "1-1":
        wn_t11.append(i)
    if index_type_wn[i][1] == "1-N":
        wn_t1N.append(i)
    if index_type_wn[i][1] == "N-1":
        wn_tN1.append(i)
    if index_type_wn[i][1] == "N-N":
        wn_tNN.append(i)

for i in range(len(index_type_fb)):
    if index_type_fb[i][1] == "1-1":
        fb_t11.append(i)
    if index_type_fb[i][1] == "1-N":
        fb_t1N.append(i)
    if index_type_fb[i][1] == "N-1":
        fb_tN1.append(i)
    if index_type_fb[i][1] == "N-N":
        fb_tNN.append(i)                                
    
#%% Step 3) Reconstructing the triple ID based on the relational types

## WN18RR
wn_1_id, wn_2_id, wn_3_id, wn_4_id = [], [], [], [] # 1-1, 1-N, N-1, N-N
wn_1_original, wn_2_original, wn_3_original, wn_4_original= [], [], [], []
for i in range(len(original_wn_test2id)):     
    if original_wn_test2id[i][1] in wn_t11:
        wn_1_id.append(original_wn_test2id[i])
        wn_1_original.append(original_wn_test[i])
    if original_wn_test2id[i][1] in wn_t1N:
        wn_2_id.append(original_wn_test2id[i])
        wn_2_original.append(original_wn_test[i])            
    if original_wn_test2id[i][1] in wn_tN1:
        wn_3_id.append(original_wn_test2id[i])
        wn_3_original.append(original_wn_test[i])
    if original_wn_test2id[i][1] in wn_tNN:
        wn_4_id.append(original_wn_test2id[i])   
        wn_4_original.append(original_wn_test[i])         
wn_1_id, wn_2_id, wn_3_id, wn_4_id = np.array(wn_1_id), np.array(wn_2_id), np.array(wn_3_id), np.array(wn_4_id)
wn_1_original, wn_2_original, wn_3_original, wn_4_original = np.array(wn_1_original), np.array(wn_2_original), np.array(wn_3_original), np.array(wn_4_original)         

## FB15k-237
fb_1_id, fb_2_id, fb_3_id, fb_4_id = [], [], [], [] # 1-1, 1-N, N-1, N-N
fb_1_original, fb_2_original, fb_3_original, fb_4_original= [], [], [], []
for i in range(len(original_fb_test2id)):     
    if original_fb_test2id[i][1] in fb_t11:
        fb_1_id.append(original_fb_test2id[i])
        fb_1_original.append(original_fb_test[i])
    if original_fb_test2id[i][1] in fb_t1N:
        fb_2_id.append(original_fb_test2id[i])
        fb_2_original.append(original_fb_test[i])            
    if original_fb_test2id[i][1] in fb_tN1:
        fb_3_id.append(original_fb_test2id[i])
        fb_3_original.append(original_fb_test[i])
    if original_fb_test2id[i][1] in fb_tNN:
        fb_4_id.append(original_fb_test2id[i])   
        fb_4_original.append(original_fb_test[i])        
        
fb_1_id, fb_2_id, fb_3_id, fb_4_id = np.array(fb_1_id), np.array(fb_2_id), np.array(fb_3_id), np.array(fb_4_id)
fb_1_original, fb_2_original, fb_3_original, fb_4_original = np.array(fb_1_original), np.array(fb_2_original), np.array(fb_3_original), np.array(fb_4_original)         

#%% Making txt & json files

# text file(txt)
# Saving the array to a text file with tab separation
# WN18RR, 1-1
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/get_neighbor/wn_11_id.txt',wn_1_id, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/wn_11.txt',wn_1_original, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/wn_11.tsv',wn_1_original, delimiter='\t', fmt='%s')
# WN18RR, 1-N
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/get_neighbor/wn_1N_id.txt',wn_2_id, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/wn_1N.txt',wn_2_original, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/wn_1N.tsv',wn_2_original, delimiter='\t', fmt='%s')
# WN18RR, N-1
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/get_neighbor/wn_N1_id.txt',wn_3_id, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/wn_N1.txt',wn_3_original, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/wn_N1.tsv',wn_3_original, delimiter='\t', fmt='%s')
# WN18RR, N-N
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/get_neighbor/wn_NN_id.txt',wn_4_id, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/wn_NN.txt',wn_4_original, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/wn18rr/wn_NN.tsv',wn_4_original, delimiter='\t', fmt='%s')


# FB15K-237, 1-1
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/get_neighbor/fb_11_id.txt',fb_1_id, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/fb_11.txt',fb_1_original, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/fb_11.tsv',fb_1_original, delimiter='\t', fmt='%s')

# FB15K-237,1-N
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/get_neighbor/fb_1N_id.txt',fb_2_id, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/fb_1N.txt',fb_2_original, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/fb_1N.tsv',fb_2_original, delimiter='\t', fmt='%s')

# FB15K-237,N-1
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/get_neighbor/fb_N1_id.txt',fb_3_id, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/fb_N1.txt',fb_3_original, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/fb_N1.tsv',fb_3_original, delimiter='\t', fmt='%s')

# FB15K-237,N-N
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/get_neighbor/fb_NN_id.txt',fb_4_id, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/fb_NN.txt',fb_4_original, delimiter='\t', fmt='%s')
np.savetxt('C:/Users/USER/Desktop/논문/연구/실험/Banchmark_Relation_Type/fb15k-237/fb_NN.tsv',fb_4_original, delimiter='\t', fmt='%s')

#------------------------------------------------------------------------------------------------------------------------------------#
