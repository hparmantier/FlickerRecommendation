import os
import random
from operator import itemgetter
from collections import defaultdict
import ch.epfl.lts.pageRank.runPPRMF as ppr
import ch.epfl.lts.pageRank.runPRMF as pr
import ch.epfl.lts.clustering.runClusterMF as c
import ch.epfl.lts.multi_layer.GroundTruthMF as gt
import numpy as np
import seaborn as sns
import csv
import pandas as pd
import matplotlib.pyplot as plt


def sorted_dict(dict):
    return sorted(dict.items(), key=itemgetter(1), reverse=True)

def rm_ext(f):
    return os.path.splitext(f)[0]

def main():
    imgs_path = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\mirflickr25k'
    outpath = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\outputs\\ndcg'

    #LIST IMG FILES IN DIRECTORY
    past_imgs = []
    ks = [10, 20, 40, 50, 80, 100]
    n_img = 2500
    #MEAN NDCG USING MULTIPLE IMAGES
    for k in ks:
        print '##########################################'
        print '################# K='+str(k)+' ######################'
        print '##########################################'

        csv_path = outpath+str(k)+'k_ndcg_metrics.csv'

        #CSV WRITER
        with open(csv_path, 'a') as outcsv:
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            #EVALUATION METRICS FOR DIFFERENT COMPUTATIONS
            for i in range(20):
                print 'IMAGE='+str(i+1)
                #QUERY IMAGE
                query_index = random.randint(0,n_img)
                while (query_index in past_imgs):
                    query_index = random.randint(0,n_img)
                past_imgs.append(query_index)
                query_name = 'im'+str(query_index+1)
                #COMPUTATIONS
                print '   querying...'
                feat_closests = c.feat_cluster_query(query_index, k,n_img)
                tag_closests = c.tag_cluster_query(query_index, k, n_img)
                ppranks = ppr.ppr_query(query_index,n_img)
                pranks = pr.pr_query(query_index,n_img)


                #NDCG FOR CLUSTER METHOD
                print '   cluster feat ndcg...'
                for feat, close in feat_closests.iteritems():
                    topK = map(lambda x: 'im'+str(x[0]+1)+'.jpg', close)
                    ndcg = gt.ndcg_evaluation_from_query_node(query_name, topK, k)
                    writer.writerow([query_name, feat, 'cluster', ndcg])
                #NDCG FOR CLUSTER TAG
                print '   cluster tag ndcg'
                ndcg = gt.ndcg_evaluation_from_query_node(query_name, tag_closests, k)
                writer.writerow([query_name, 'tag', 'cluster', ndcg])

                #NDCG FOR PPR METHOD
                print '   ppr ndcg...'
                for feat, ranks in ppranks.iteritems():
                    sorted_ranks = sorted_dict(ranks)
                    zip = map(lambda x: 'im'+str(x[0]+1)+'.jpg', sorted_ranks)
                    topK = zip[1:k+1]
                    ndcg = gt.ndcg_evaluation_from_query_node(query_name, topK, k)
                    writer.writerow([query_name, feat, 'ppr', ndcg])
                #NDCG FOR CLASSIC PR METHOD
                print '   pr ndcg...'
                for feat, ranks in pranks.iteritems():
                    sorted_ranks = sorted_dict(ranks)
                    zip = map(lambda x: 'im'+str(x[0]+1)+'.jpg', sorted_ranks)
                    topK = zip[1:k+1]
                    ndcg = gt.ndcg_evaluation_from_query_node(query_name, topK, k)
                    writer.writerow([query_name, feat, 'pr', ndcg])


def bar_plot_for_k(filepath):
    sns.set_style("whitegrid")
    dmetrics = pd.read_csv(filepath)
    dmetrics.columns = ['img_name', 'feature', 'algorithm', 'ndcg']
    data = sns.load_dataset("titanic")
    sns.barplot(x="feature", y="ndcg", hue="algorithm", data=dmetrics)
    plt.show()


main()


