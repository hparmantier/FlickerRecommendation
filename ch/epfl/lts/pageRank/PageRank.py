import networkx as nx
import igraph
import pickle
import numpy as np
#helloworld

def run_classic_page_rank(graph, alpha=None):
    return nx.pagerank(graph)

#uses adjacency matrix (non numpy array) as argument
def all_personalized_pagerank_from_adj(adj):
    g = igraph.Graph.Adjacency(adj.tolist(), mode=igraph.ADJ_UNDIRECTED)
    pr = []
    for node in g.vs:
        pr.append(g.personalized_pagerank(directed=False, damping=0.85, reset_vertices=node))
    return pr

def node_personalized_pagerank_from_adj(adj, index):
    g = igraph.Graph.Adjacency(adj.tolist(), mode=igraph.ADJ_UNDIRECTED)
    pr = g.personalized_pagerank(directed=False, damping=0.85, reset_vertices=g.vs[index])
    return list(enumerate(pr))



def all_personalized_pagerank_from_nx(graph):
    adj = from_nx_to_adj(graph)
    return all_personalized_pagerank_from_adj(adj)


def node_personalized_pagerank_from_nx(graph, index):
    # adj = from_nx_to_adj(graph)
    # return node_personalized_pagerank_from_adj(adj, index)
    dict = {}
    for node in graph.nodes(data=False):
        if node == index:
            dict[node] = 0.5
        else:
            dict[node] = 0.1
    return nx.pagerank(graph, alpha=0.85, personalization=dict)

def from_nx_to_adj(graph):
    return nx.adjacency_matrix(graph)