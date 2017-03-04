import ch.epfl.lts.pageRank.runPPRMF as ppr
import ch.epfl.lts.pageRank.runPRMF as pr
import ch.epfl.lts.clustering.runClusterMF as cl
import random

def main():
    imgs_path = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\mirflickr25k\\'
    outpath = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\outputs\\ranks\\'
    n_img = 2500
    k = 10
    #QUERY IMAGE
    query_index = random.randint(0,n_img)
    query_name = 'im'+str(query_index+1)
    #COMPUTATIONS CLUSTER FOR 2 CASES (FEATURES, TAGS)
    print 'cluster query'
    closests = cl.feat_cluster_query(query_index, k, n_img)
    print 'tag cluster query'
    tag_closests = cl.tag_cluster_query(query_index, k, n_img)


    #COMPUTATIONS PAGERANK
    print 'ppr query'
    ppranks = ppr.ppr_query(query_index, n_img)
    print 'pr query'
    pranks = pr.pr_query(query_index, n_img)

    #SAVING
    print 'saving...'
    cl.feat_save_closests(outpath, query_name, closests)
    cl.tag_save_closests(outpath, query_name, tag_closests)
    ppr.save_ppr(ppranks, query_name, k)
    pr.save_pr(pranks, query_name, k)


main()
