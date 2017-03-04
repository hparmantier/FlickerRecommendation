import numpy as np
import itertools as it
import networkx as nx
import pickle
import matplotlib.pyplot as plt

from numpy import loadtxt

class TagGraph(object):
    def __init__(self):
        self.path = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\\tags\Data\\'

    def generateDict(self,n_imgs):
        self.imgs = n_imgs
        dict = {}
        g = nx.Graph()
        for i in range(n_imgs):
            g.add_node(i)
            path = self.path+`i`+'.txt'
            img_tags = open(path, 'r').read().split('\n')
            img_tags.pop()
            for tag in img_tags:
                if tag in dict.keys():
                    dict[tag].append(i)
                else :
                    dict[tag] = [i]
        self.dict = dict
        self.nxgraph = g

    def removeUnique(self):
        for key in self.dict.keys():
            if len(self.dict[key]) == 1 :
                self.dict.pop(key)


    def buildNNGraph(self):
        matrix = np.zeros(shape=(self.imgs, self.imgs))
        for key in self.dict.keys():
            combinations = it.combinations(self.dict[key],2)
            for c in combinations:
                matrix[c[0]][c[1]] = 1
                matrix[c[1]][c[0]] = 1
        self.nngraph = matrix

    def buildNXGraph(self):
        for key in self.dict.keys():
            combinations = it.combinations(self.dict[key],2)
            for c in combinations:
                self.nxgraph.add_edge(c[0],c[1], weight=1)

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

    def nnGet(self):
        return self.nngraph

    def nxGet(self):
        return self.nxgraph

def adjacency_from_tag_graph_for_n_img(n_img):
    tag_graph = TagGraph()
    tag_graph.generateDict(n_img)
    tag_graph.removeUnique()
    tag_graph.buildNNGraph()
    return tag_graph.nnGet()

def nx_from_tag_graph_for_n_img(n_img):
    tag_graph = TagGraph()
    tag_graph.generateDict(n_img)
    tag_graph.removeUnique()
    tag_graph.buildNXGraph()
    return tag_graph.nxGet()

