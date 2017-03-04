from sklearn import cluster, metrics
import itertools as it
from scipy.spatial import distance
from operator import itemgetter
##test
class Cluster(object):
    def __init__(self, data):
        self.data = data

    def labels_and_score_from_spectral_clustering(self, tr_matrix, nc):
        self.labels = cluster.spectral_clustering(tr_matrix, nc,eigen_solver='arpack')
        self.silhouette = metrics.silhouette_score(self.data,self.labels, metric='euclidean')

    def silhouettes_from_different_ncluster(self, tr_matrix, clusters):
        silhouettes = []
        for i in range(len(clusters)):
            self.labels_and_score_from_spectral_clustering(tr_matrix, clusters[i])
            silhouettes.append(self.silhouette)
        return silhouettes

    def getLabels(self):
        return self.labels

    def getScore(self):
        return self.silhouette

    def topN_closest_cluster_nodes(self, n, img_index):
        query_cluster = self.labels[img_index]
        i = 0
        cluster = []
        for label in self.labels:
            if label == query_cluster and i != img_index:
                cluster.append(i)
            i += 1
        dist = dict(map(lambda x: (x, distance.euclidean(self.data[img_index], self.data[x])), cluster))
        tops = sorted(dist.items(), key=itemgetter(1))[:n]
        return tops

#CASE TAG GRAPH
def label_from_tag_adj_clustering(tr_matrix, nc):
    return cluster.spectral_clustering(tr_matrix, nc, eigen_solver='arpack')

def imgs_in__label(labels, query_index):
    cluster = labels[query_index]
    acc = []
    i = 0
    for label in labels:
        if label == cluster and i != query_index:
            acc.append(i)
        i += 1
    return map(lambda i: 'im'+str(i+1)+'jpg', acc)




