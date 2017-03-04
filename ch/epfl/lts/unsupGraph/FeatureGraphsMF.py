
from Graph import Graph
from MFReader import MFReader
import TagGraph as tg

#CREATE GRAPH FILES FROM IMAGES FEATURES
#returns dictionary of adjacency matrix for each layer
def graph_per_mf_layer(n_img):
    graphs = {}
    #DATA EXTRACTION AND GRAPH BUILD FOR 2 FEATURES
    reader = MFReader(n_img)
    reader.extractData()
    f_datas = reader.all()
    for feat, data in f_datas.iteritems():
        g = Graph.fromData(data)
        g.buildNNGraph(50)
        adj = g.nnGet()
        graphs[feat] = adj
    #GRAPH CREATION FOR TAG LAYER
    adj_tag_graph = tg.adjacency_from_tag_graph_for_n_img(n_img)
    graphs['tags'] = adj_tag_graph

    return graphs

def datas_for_2_features_in_mf(n_img):
    #DATA EXTRACTION AND GRAPH BUILD FOR 2 FEATURES
    reader = MFReader(n_img)
    reader.extractData()
    f_datas = reader.all()
    return f_datas


