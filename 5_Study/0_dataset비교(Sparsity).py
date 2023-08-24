"""
Created on meaningful96

DL Project
"""
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

plt.close("all")


def load_dataset(file_path):
    triples = []
    with open(file_path, "r") as file:
        for line in file:
            head, relation, tail = line.strip().split('\t')
            triples.append((head, relation, tail))
    return triples

dataset_path1 = ("C:/Users/USER/Desktop/datasets_knowledge_embedding-master/WN18RR/text/test.txt")
dataset_path2 = ("C:/Users/USER/Desktop/datasets_knowledge_embedding-master/FB15k-237/test.txt")

triples1 = load_dataset(dataset_path1)
triples2 = load_dataset(dataset_path2)


def create_graph(triples):
    graph = nx.MultiDiGraph()
    for head, relation, tail in triples:
        graph.add_edge(head, tail, label=relation)
    return graph

graph_wn18rr = create_graph(triples1)
graph_fb15k_237 = create_graph(triples2)



def degree_distribution(graph):
    degrees = [d for n, d in graph.degree()]
    degree_counts = np.bincount(degrees)
    degree_freq = degree_counts / len(degrees)
    return pd.Series(degree_freq, index=np.arange(len(degree_freq)))

degree_dist_wn = degree_distribution(graph_wn18rr)
degree_dist_fb = degree_distribution(graph_fb15k_237)

import matplotlib.pyplot as plt
import seaborn as sns

def plot_degree_distribution(degree_dist):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=degree_dist.index, y=degree_dist.values)
    plt.xlabel("Degree")
    plt.ylabel("Frequency")
    plt.title("Degree Distribution")
    plt.show()

plot_degree_distribution(degree_dist_wn)
plot_degree_distribution(degree_dist_fb)