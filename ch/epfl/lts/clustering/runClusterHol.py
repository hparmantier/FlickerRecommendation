import os
import numpy as np
from ch.epfl.lts.unsupGraph.HolidaysReader import HolidaysReader
from ch.epfl.lts.unsupGraph.Graph import Graph
from operator import itemgetter
from ch.epfl.lts.clustering.Cluster import Cluster

def cluster_query(query_index, k):
    #FEATURE READER
    reader = HolidaysReader()
    reader.extractData()
    datas = reader.all()
    closests = {}
    #PARAMS FOR EACH FEATURE
    knc = {'ch': (50, 4), 'caf': (65,6), 'gist': (100,3), 'rand' : (120,3)}
    #ITERATION ON FEATURES AND PR RUNS
    for key,value in datas.iteritems():
        graph = Graph.fromData(value)
        cluster = Cluster(value)
        graph.buildNNGraph(knc[key][0])
        adj = graph.nnGet()
        cluster.labels_and_score_from_spectral_clustering(adj, knc[key][1])
        closests[key] = cluster.topN_closest_cluster_nodes(k, query_index)
    return closests


def save_closest(outpath, query_name, closests):
    #LIST IMG FILES IN DIRECTORY
    imgs_path = 'D:\cours\MA1\Semester Project\datasets\holidays\jpg'
    files = [f for f in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, f))]
    for feat, close in closests.iteritems():
        top10 = map(lambda x: files[x[0]], close)
        np.savetxt(outpath+feat+'-ClQ-'+query_name+'.txt', top10, fmt="%s")

def rm_ext(f):
    return os.path.splitext(f)[0]

def sorted_dict(dict):
    return sorted(dict.items(), key=itemgetter(1), reverse=True)







