import random
import os
import pickle
import numpy as np
from collections import Counter
import cv2
import ch.epfl.lts.pageRank.PageRank as pr
from ch.epfl.lts.unsupGraph.HolidaysReader import HolidaysReader
from ch.epfl.lts.unsupGraph.Graph import Graph
from collections import OrderedDict
from operator import itemgetter

def pr_query(query_index):
    #FEATURE READER
    reader = HolidaysReader()
    reader.extractData()
    datas = reader.all()
    #PARAMS FOR EACH FEATURE
    knc = {'ch': (50, 4), 'caf': (65,6), 'gist': (100,3), 'rand' : (120,3)}
    ppranks = {}
    pranks = {}
    #ITERATION ON FEATURES AND PR RUNS
    for key,value in datas.iteritems():
        graph = Graph.fromData(value)
        graph.buildNNGraph(knc[key][0])
        graph.buildNXGraph()
        nx = graph.nxGet()
        pranks[key] = pr.run_classic_page_rank(nx)
    return pranks


def save_pr(pranks, query_name, k):
    #LIST IMG FILES IN DIRECTORY
    imgs_path = 'D:\cours\MA1\Semester Project\datasets\holidays\jpg'
    outpath = 'D:\cours\MA1\Semester Project\datasets\holidays\outputs\\ranks\\'
    files = [f for f in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, f))]
    #SAVE RANKS
    for feat, ranks in pranks.iteritems():
        sorted_ranks = sorted_dict(ranks)
        zip = map(lambda x: (rm_ext(files[x[0]]), str(x[0]), str(x[1])), sorted_ranks)
        np.savetxt(outpath+feat+'-pr-'+query_name+'.txt', zip[:k+1],fmt='%s')

def rm_ext(f):
    return os.path.splitext(f)[0]

def sorted_dict(dict):
    return sorted(dict.items(), key=itemgetter(1), reverse=True)
