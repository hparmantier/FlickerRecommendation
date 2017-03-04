import ch.epfl.lts.unsupGraph.FeatureGraphsMF as fg
from ch.epfl.lts.unsupGraph.Graph import Graph
import os
import numpy as np
import ch.epfl.lts.pageRank.PageRank as pr
import ch.epfl.lts.unsupGraph.TagGraph as tg
from operator import itemgetter

def ppr_query(query_index, n_img):
    #FEATURE READER
    datas = fg.datas_for_2_features_in_mf(n_img)
    #PARAMS FOR EACH FEATURE
    knc = {'homogeneous_texture': (15, 3), 'edge_histogram': (150, 4)}
    ppranks = {}
    #ITERATION ON FEATURES AND PR RUNS
    for key,value in datas.iteritems():
        graph = Graph.fromData(value)
        graph.buildNNGraph(knc[key][0])
        graph.buildNXGraph()
        adj = graph.nnGet()
        ppranks[key] = dict(pr.node_personalized_pagerank_from_adj(adj, query_index))
    #COMPUTATIONS FOR TAG
    tag_mat = tg.adjacency_from_tag_graph_for_n_img(n_img)
    ppranks['tags'] = dict(pr.node_personalized_pagerank_from_adj(tag_mat, query_index))
    return ppranks


def save_ppr(ppranks, query_name, k):
    outpath = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\outputs\\ranks\\'
    #SAVE RANKS
    for feat, ranks in ppranks.iteritems():
        sorted_ranks = sorted_dict(ranks)
        zip = map(lambda x: ('im'+str(x[0]+1), str(x[0]), str(x[1])), sorted_ranks)
        np.savetxt(outpath+feat+'-ppr-'+query_name+'.txt', zip[1:k+1],fmt='%s')




def rm_ext(f):
    return os.path.splitext(f)[0]

def sorted_dict(dict):
    return sorted(dict.items(), key=itemgetter(1), reverse=True)