import os
from operator import itemgetter

class Indicator(object):
    def __init__(self, node, n_img):
        self.categories = ['sky', 'water', 'people', 'night', 'plant_life', 'animals', 'structures', 'sunset', 'indoor', 'transport']
        self.tag_path = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\\tags\Data\\'
        self.img_path = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\mirflickr25k'
        self.n_img = n_img
        self.q_node = node
        self.tag_f = str(int(rm_ext(self.q_node)[2:])-1)
        self.tag_to_imgs = {}
        self.q_ngb = []
        self.q_tags = open(self.tag_path+self.tag_f+'.txt', 'r').read().split('\n')
        self.q_tags.pop()

    def get_q_tags(self):
        return self.q_tags

    def get_tag_file(self):
        return self.tag_f

    def set_tag_map(self):
        dict = {}
        for i in range(self.n_img):
            path = self.tag_path+str(i)+'.txt'
            img = 'im'+str(i+1)+'.jpg'
            tags = open(path, 'r').read().split('\n')
            tags.pop()
            for t in tags:
                if t in dict.keys():
                    dict[t].append(img)
                else:
                    dict[t] = [img]
        self.tag_to_imgs = dict

    def get_tag_distr(self):
        return self.tag_to_imgs

    def set_query_ngbs(self):
        dict = {}
        for tag in self.q_tags:
            if tag in self.tag_to_imgs.keys():
                imgs = self.tag_to_imgs[tag]
                for img in imgs:
                    if img != self.q_node:
                        if img in dict.keys():
                            dict[img] += 1
                        else:
                            dict[img] = 1
        self.q_ngb = sorted_dict(dict)

    def get_query_ngb(self):
        return self.q_ngb



def rm_ext(f):
    return os.path.splitext(f)[0]

def sorted_dict(dict):
    return sorted(dict.items(), key=itemgetter(1), reverse=True)

def get_top_closests_tagged(node, n_img):
    indicator = Indicator(node, n_img)
    indicator.set_tag_map()
    indicator.set_query_ngbs()
    tops = map(lambda x: x[0], indicator.get_query_ngb())
    return tops