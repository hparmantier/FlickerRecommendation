import ch.epfl.lts.unsupGraph.FeatureGraphsMF as fg
from ch.epfl.lts.clustering.Cluster import Cluster
from ch.epfl.lts.unsupGraph.Graph import Graph
import matplotlib.pyplot as plt

def main():
    datas = fg.datas_for_2_features_in_mf(2500)
    kparams = range(5,200, 10)
    for feat, data in datas.iteritems():
        print '###################'
        print feat+':'
        print '###################'
        g = Graph.fromData(data)
        c = Cluster(data)
        scores = []
        for param in kparams:
            g.buildNNGraph(param)
            gk = g.nnGet()
            c.labels_and_score_from_spectral_clustering(gk, 5)
            scores.append(c.getScore())
        print kparams
        print scores
        plt.plot(kparams, scores)
        plt.show()

main()
