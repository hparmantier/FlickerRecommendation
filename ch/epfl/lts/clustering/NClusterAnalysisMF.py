import ch.epfl.lts.unsupGraph.FeatureGraphsMF as fg
import matplotlib.pyplot as plt
from ch.epfl.lts.unsupGraph.Graph import Graph
from ch.epfl.lts.clustering.Cluster import Cluster
from  ch.epfl.lts.unsupGraph.HolidaysReader import HolidaysReader


def main():
    datas = fg.datas_for_2_features_in_mf(2500)
    nc = range(3,15,1)
    kparam = {'edge_histogram': 150, 'homogeneous_texture': 15}
    for feat,data in datas.iteritems():
        g = Graph.fromData(data)
        c = Cluster(data)
        print feat+':'
        g.buildNNGraph(kparam[feat])
        tr = g.nnGet()
        scores = c.silhouettes_from_different_ncluster(tr,nc)
        print nc
        print scores
        plt.plot(nc, scores)
        plt.show()


main()