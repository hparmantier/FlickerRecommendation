import numpy as np
from numpy import loadtxt

class MFReader(object):
    def __init__(self, n_img):
        self.n_img = n_img
        self.path = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\\'
        self.paths = map(lambda x: self.path+x, ['features_homogeneoustexture\\', 'features_edgehistogram\\'])
        self.alias = ['homogeneous_texture', 'edge_histogram']
        self.n_feats = [43,150]
        self.datas = {}

    def extractData(self):
        for i in range(2):
            dataset = np.zeros(shape=(self.n_img, self.n_feats[i]))
            for index in range(self.n_img):
                path = self.paths[i]+str(index)+'.txt'
                dataset[index] = loadtxt(path, delimiter='\n')
            self.datas[self.alias[i]] = dataset

    def all(self):
        return self.datas


