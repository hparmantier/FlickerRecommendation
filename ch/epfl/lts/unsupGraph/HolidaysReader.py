import numpy as np
import scipy.io



class HolidaysReader(object):
    def __init__(self):
        self.path = 'D:\cours\MA1\Semester Project\datasets\holidays\\features\holidays\\'
        self.paths = map(lambda x: self.path+x, ['ColorHistogram_1000', 'fc6_caffe', 'gist', 'rand'])
        self.alias = ["color_Hist", "features", "gist_mat", "feat"]
        self.names = ['ch', 'caf', 'gist', 'rand']
        self.datas = {}

    def extractData(self):
        for i in range(4):
            data = np.transpose(scipy.io.loadmat(self.paths[i])[self.alias[i]])
            self.datas[self.names[i]] = data

    def dataCaf(self):
        return self.datas['caf']

    def dataCh(self):
        return self.datas['ch']

    def dataGist(self):
        return self.datas['gist']

    def dataRand(self):
        return self.datas['rand']

    def all(self):
        return self.datas


    def names(self):
        return self.names