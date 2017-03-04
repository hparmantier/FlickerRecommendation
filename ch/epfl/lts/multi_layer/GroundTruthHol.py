import os
from collections import defaultdict
import math

class IndicatorHol(object):
    def __init__(self, node):
        self.gr_truth = {}
        self.relevants = []
        self.q_node = node


    def extract_holidays_ground_truth(self):
        imgs_path = 'D:\cours\MA1\Semester Project\datasets\holidays\jpg'
        files = [f for f in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, f))]
        gt = defaultdict(list)
        i = 0
        for f in files:
            name = os.path.splitext(f)[0]
            group = int(name[1:-2])
            #index = name[4:]
            gt[group].append((f, i))
            i+=1
        self.gr_truth = gt


    def extract_relevant_docs_from_query_node(self):
        for group, nodes in self.gr_truth.iteritems():
            names = map(lambda(x,y): x, nodes)
            if self.q_node in names:
                self.relevants = nodes
                return True
        return False

    def indicate(self, nodeK):
        names = map(lambda(x,y): x, self.relevants)
        if nodeK in names:
            return 1
        else :
            return 0

###################################################END CLASS############################################################

def ndcg_evaluation_from_query_node(node, tops, k):
    indicator = IndicatorHol(node)
    indicator.extract_holidays_ground_truth()
    indicator.extract_relevant_docs_from_query_node()
    l = map(lambda x: indicator.indicate(x), tops)
    z = get_ideal_ndcg(node, tops, k)
    if z == 0:
        z = 1
    sum = 0
    for i in range(k):
        sum += (math.pow(2, l[i]) - 1) / math.log(i + 2, 2)
    return sum/z

def get_ideal_ndcg(node, tops, k):
    indicator = IndicatorHol(node)
    indicator.extract_holidays_ground_truth()
    indicator.extract_relevant_docs_from_query_node()
    l = map(lambda x: indicator.indicate(x), tops)
    sorted = sorted(l, key=int, reverse=True)
    max = 0
    for i in range(k):
        max += (math.pow(2,sorted[i])-1) / math.log(i+2, 2)
    return max





