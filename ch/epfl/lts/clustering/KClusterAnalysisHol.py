import numpy as np
import scipy.io
from sklearn import cluster, metrics
import matplotlib.pyplot as plt
from ch.epfl.lts.unsupGraph.Graph import Graph
from ch.epfl.lts.clustering.Cluster import Cluster
from  ch.epfl.lts.unsupGraph.HolidaysReader import HolidaysReader


def main():
    reader = HolidaysReader()
    reader.extractData()
    datas = reader.all()
    kparams = range(5,150,10)
    for k,v in datas.iteritems():
        g = Graph.fromData(v)
        c = Cluster(v)
        print k+':'
        scores = []
        for param in kparams:
            g.buildNNGraph(param)
            gk = g.nnGet()
            c.labels_and_score_from_spectral_clustering(gk,5)
            scores.append(c.getScore())
        print kparams
        print scores
        plt.plot(kparams, scores)
        plt.show()


main()