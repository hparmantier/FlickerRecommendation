import os
import random
from operator import itemgetter
from collections import defaultdict
import ch.epfl.lts.pageRank.runPPRHol as ppr
import ch.epfl.lts.pageRank.runPRHol as pr
import ch.epfl.lts.clustering.runClusterHol as c
import ch.epfl.lts.multi_layer.GroundTruthHol as gt
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
    imgs_path = 'D:\cours\MA1\Semester Project\datasets\holidays\jpg'
    outpath = 'D:\cours\MA1\Semester Project\datasets\holidays\outputs\\ndcg\\'

    #LIST IMG FILES IN DIRECTORY
    files = [f for f in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, f))]
    past_imgs = []
    cluster_metrics_k = defaultdict(list)
    ppr_metrics_k = defaultdict(list)
    ks = [10, 15, 20,25, 30, 40, 50]
    #MEAN NDCG USING MULTIPLE IMAGES
    for k in ks:
        print '##########################################'
        print '################# K='+str(k)+' ######################'
        print '##########################################'

        csv_path = outpath+str(k)+'k_ndcg_metrics.csv'

        #CSV WRITER
        with open(csv_path, 'a') as outcsv:
            writer = csv.writer(outcsv, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
            #writer.writerow(['img_name', 'feature', 'algorithm', 'ndcg'])
            #EVALUATION METRICS FOR DIFFERENT COMPUTATIONS
            for i in range(20):
                print 'IMAGE='+str(i+1)
                #QUERY IMAGE
                query_index = random.randint(0,1490)
                while (query_index in past_imgs):
                    query_index = random.randint(0,1490)
                past_imgs.append(query_index)
                query_name = rm_ext(files[query_index])
                #COMPUTATIONS
                print '   querying...'
                closests = c.cluster_query(query_index, k)
                ppranks = ppr.ppr_query(query_index)
                pranks = pr.pr_query(query_index)


                #NDCG FOR CLUSTER METHOD
                print '   cluster ndcg...'
                for feat, close in closests.iteritems():
                    topK = map(lambda x: files[x[0]], close)
                    ndcg = gt.ndcg_evaluation_from_query_node(files[query_index], topK, k)
                    writer.writerow([files[query_index], feat, 'cluster', ndcg])
                #NDCG FOR PPR METHOD
                print '   ppr ndcg...'
                for feat, ranks in ppranks.iteritems():
                    sorted_ranks = sorted_dict(ranks)
                    zip = map(lambda x: files[x[0]], sorted_ranks)
                    topK = zip[1:k+1]
                    ndcg = gt.ndcg_evaluation_from_query_node(files[query_index], topK, k)
                    writer.writerow([files[query_index], feat, 'ppr', ndcg])
                #NDCG FOR CLASSIC METHOD
                print '   pr ndcg...'
                for feat, ranks in pranks.iteritems():
                    sorted_ranks = sorted_dict(ranks)
                    zip = map(lambda x: files[x[0]], sorted_ranks)
                    topK = zip[1:k+1]
                    ndcg = gt.ndcg_evaluation_from_query_node(files[query_index], topK, k)
                    writer.writerow([files[query_index], feat, 'pr', ndcg])


def bar_plot_for_k(filepath):
    sns.set_style("whitegrid")
    dmetrics = pd.read_csv(filepath)
    dmetrics.columns = ['img_name', 'feature', 'algorithm', 'ndcg']
    data = sns.load_dataset("titanic")
    sns.barplot(x="feature", y="ndcg", hue="algorithm", data=dmetrics)
    plt.show()


main()


