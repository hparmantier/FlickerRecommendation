import os
import math

class IndicatorMF(object):
    def __init__(self, node):
        self.subcategories = {  'clouds': 'sky',
                                'sea': 'water',
                                'ocean': 'water',
                                'river': 'water',
                                'lake': 'water',
                                'portrait': 'people',
                                'boy': 'people',
                                'man': 'people',
                                'girl': 'people',
                                'woman': 'people',
                                'baby': 'people',
                                'tree': 'plant life',
                                'flower': 'plant life',
                                'dog': 'animals',
                                'bird': 'animals',
                                'architecture': 'man-built structures',
                                'building': 'man-built structures',
                                'house': 'man-built structures',
                                'city': 'man-built structures',
                                'urban': 'man-built structures',
                                'bridge': 'man-built structures',
                                'road': 'man-built structures',
                                'street': 'man-built structures',
                                'car': 'transport'}
        self.categories = ['sky', 'water', 'people', 'night', 'plant life', 'animals', 'man-built structures', 'sunset', 'indoor', 'transport']
        self.dir = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\\tags\Data\\'
        self.query_name = node
        self.query_tag_index = tag_index_from_img_name(self.query_name)
        self.query_cats = self.generate_cat_dependance(self.query_tag_index)

    def generate_cat_dependance(self, tag_index):
        tags = tag_list_from_file(self.dir, tag_index)
        cat = []
        for t in tags:
            if t in self.subcategories.keys():
                cat.append(t)
                tt = self.subcategories[t]
                if tt not in cat:
                    cat.append(tt)
            else:
                if t in self.categories:
                    if t not in cat:
                        cat.append(t)
        return cat

    def indicate(self, other):
        other_index = tag_index_from_img_name(other)
        other_cats = self.generate_cat_dependance(other_index)
        if set(self.query_cats).isdisjoint(other_cats):
            return 0
        else:
            return 1

############################################HELPER METHODS##############################################################

def tag_list_from_file(dir, tag_index):
    tags = open(dir+str(tag_index)+'.txt', 'r').read().split('\n')
    tags.pop()
    return tags

def tag_index_from_img_name(name):
    return int(rm_ext(name)[2:])-1

def rm_ext(f):
    return os.path.splitext(f)[0]

def ndcg_evaluation_from_query_node(node, tops, k):
    indicator = IndicatorMF(node)
    l = map(lambda x: indicator.indicate(x), tops)
    sum = 0
    z = get_ideal_ndcg(k,tops,node)
    if z == 0 :
        z = 1
    if len(l) < k:
        for i in range(len(l)):
            sum += (math.pow(2, l[i]) - 1) / math.log(i + 2, 2)
        for i in range(len(tops), k):
            sum += (math.pow(2,0) - 1) / math.log(i+2, 2)
    else:
        for i in range(k):
            sum += (math.pow(2, l[i]) - 1) / math.log(i + 2, 2)
    return sum/z

def get_ideal_ndcg(k, tops, node):
    indicator = IndicatorMF(node)
    l = map(lambda x: indicator.indicate(x), tops)
    sorted = sorted(l, key=int, reverse=True)
    max = 0
    if len(sorted) < k:
        for i in range(len(sorted)):
            max += (math.pow(2,sorted[i])-1) / math.log(i+2, 2)
        for i in range(len(sorted),k):
            max += (math.pow(2,0)-1) / math.log(i+2, 2)
    else:
        for i in range(k):
            max += (math.pow(2,sorted[i])-1) / math.log(i + 2, 2)
    return max