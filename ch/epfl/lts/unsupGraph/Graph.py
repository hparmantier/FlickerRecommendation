import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from numpy import loadtxt
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
import pickle

# CLASS CREATING GRAPH FROM NN COMPUTATION ON IMAGE FEATURE

class Graph(object):
    def __init__(self, path, imgs, features, matrix = None):
        if matrix is None:
            self.path = path
            self.imgs, self.features = imgs, features
            self.nngraph = np.zeros(shape=(imgs,imgs))
            self.data = np.zeros(shape=(imgs,features))
            self.nxgraph = nx.Graph()
        else :
            self.path = ''
            self.imgs, self.features = matrix.shape[0], matrix.shape[1]
            self.nngraph = None
            self.nxgraph = nx.Graph()
            self.data = matrix
            self.standard = self.data.std()

    @classmethod
    def fromData(cls, matrix):
        return cls('', 0,0,matrix)

    def loadData(self):
        dataset = np.zeros(shape=(self.imgs, self.features))
        for i in range (0,self.imgs):
            path = self.path+`i`+'.txt'
            dataset[i] = loadtxt(path, delimiter='\n')
        self.data = dataset


    def buildNNGraph(self, k):
        nn = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(self.data)
        kg = nn.kneighbors_graph(self.data).toarray()
        self.nngraph = kg

    def buildNXGraph(self, weighted=False):
        (m,n) = self.nngraph.shape
        for i in range(m):
            self.nxgraph.add_node(i)
        for i in range(m):
            for j in range(n):
                if self.nngraph[i][j] == 1:
                    if weighted:
                        kernel = np.exp(-1*(euclidean(self.data[i], self.data[j])**2)/self.standard**2)
                    else:
                        kernel = 1
                    self.nxgraph.add_edge(i,j,weight=kernel)

    def nnDraw(self):
        G1 = nx.from_numpy_matrix(self.nngraph)
        nx.draw_networkx(G1)
        plt.show()

    def nxDraw(self):
        nx.draw_networkx(self.nxgraph)
        plt.show()

    def nnSave(self, file):
        np.save(file, self.nngraph)

    def nxSave(self, file):
        pickle.dump(self.nxgraph,open(file,'w'))



    def nxGet(self):
        return self.nxgraph

    def nnGet(self):
        return self.nngraph