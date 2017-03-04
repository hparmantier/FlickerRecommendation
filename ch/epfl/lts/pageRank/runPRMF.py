import ch.epfl.lts.unsupGraph.FeatureGraphsMF as fg
from ch.epfl.lts.unsupGraph.Graph import Graph
import ch.epfl.lts.unsupGraph.TagGraph as tg
import os
import numpy as np
import ch.epfl.lts.pageRank.PageRank as pr
from operator import itemgetter

def pr_query(query_index, n_img):
    #FEATURE READER
    datas = fg.datas_for_2_features_in_mf(n_img)
    #PARAMS FOR EACH FEATURE
    knc = {'homogeneous_texture': (15, 3), 'edge_histogram': (150, 4)}
    pranks = {}
    #ITERATION ON FEATURES AND PR RUNS
    for key,value in datas.iteritems():
        graph = Graph.fromData(value)
        graph.buildNNGraph(knc[key][0])
        graph.buildNXGraph()
        nx = graph.nxGet()
        pranks[key] = pr.run_classic_page_rank(nx)
    #TAG CASE
    tag_nx = tg.nx_from_tag_graph_for_n_img(n_img)
    pranks['tags'] = pr.run_classic_page_rank(tag_nx)
    return pranks


def save_pr(pranks, query_name, k):
    #LIST IMG FILES IN DIRECTORY
    outpath = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\outputs\\ranks\\'
    #SAVE RANKS
    for feat, ranks in pranks.iteritems():
        sorted_ranks = sorted_dict(ranks)
        zip = map(lambda x: ('im'+str(x[0]+1)+'.jpg', str(x[0]), str(x[1])), sorted_ranks)
        np.savetxt(outpath+feat+'-pr-'+query_name+'.txt', zip[:k+1],fmt='%s')




def rm_ext(f):
    return os.path.splitext(f)[0]

def sorted_dict(dict):
    return sorted(dict.items(), key=itemgetter(1), reverse=True)