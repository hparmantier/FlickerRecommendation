import ch.epfl.lts.unsupGraph.FeatureGraphsMF as fg
from ch.epfl.lts.unsupGraph.Graph import Graph
import ch.epfl.lts.clustering.Cluster as cl
import ch.epfl.lts.unsupGraph.TagGraph as tg
import TagDistance as td
import os
import numpy as np


def feat_cluster_query(query_index, k, n_img):
    #FEATURE READER/EXTRACTION
    datas = fg.datas_for_2_features_in_mf(n_img)
    closests = {}
    #PARAMS FOR EACH FEATURE
    knc = {'homogeneous_texture': (15, 3), 'edge_histogram': (150, 4)}
    for feat, data in datas.iteritems():
        graph = Graph.fromData(data)
        cluster = cl.Cluster(data)
        graph.buildNNGraph(knc[feat][0])
        adj = graph.nnGet()
        cluster.labels_and_score_from_spectral_clustering(adj, knc[feat][1])
        closests[feat] = cluster.topN_closest_cluster_nodes(k, query_index)
    return closests

def feat_save_closests(outpath, query_name, closests):
    #LIST IMG FILES IN DIRECTORY
    for feat, close in closests.iteritems():
        top10 = map(lambda x: 'im'+str(x[0]+1)+'.jpg', close)
        np.savetxt(outpath+feat+'-ClQ-'+rm_ext(query_name)+'.txt', top10, fmt="%s")

def tag_cluster_query(query_index, k, n_img):
    query_node = 'im'+str(query_index+1)+'.jpg'
    tops = td.get_top_closests_tagged(query_node, n_img)
    g = tg.adjacency_from_tag_graph_for_n_img(n_img)
    labels = cl.label_from_tag_adj_clustering(g, 5)
    label = labels[query_index]
    # imgs_in_label = cl.imgs_in__label(labels, query_index)
    filtered = filter(lambda x: labels[name_to_index(x)] == label, tops)
    if len(filtered) <= k:
         return filtered
    else:
         return filtered[:k]

def name_to_index(name):
    return int(rm_ext(name)[2:])-1

def tag_save_closests(outpath, query_name, closests):
    np.savetxt(outpath+'tag-ClQ-'+rm_ext(query_name)+'.txt', closests, fmt="%s")

def rm_ext(f):
    return os.path.splitext(f)[0]





