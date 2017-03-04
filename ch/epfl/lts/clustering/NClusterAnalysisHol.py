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
    nc = range(3,15,1)
    kparam = {'ch': 50, 'caf': 65, 'gist': 100, 'rand' : 120}
    for k,v in datas.iteritems():
        g = Graph.fromData(v)
        c = Cluster(v)
        print k+':'
        g.buildNNGraph(kparam[k])
        tr = g.nnGet()
        scores = c.silhouettes_from_different_ncluster(tr,nc)
        print nc
        print scores
        plt.plot(nc, scores)
        plt.show()


main()