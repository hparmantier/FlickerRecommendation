import os
import random

import ch.epfl.lts.pageRank.runPPRHol as ppr
import ch.epfl.lts.pageRank.runPRHol as pr
import ch.epfl.lts.clustering.runClusterHol as c


def main():
    imgs_path = 'D:\cours\MA1\Semester Project\datasets\holidays\jpg'
    outpath = 'D:\cours\MA1\Semester Project\datasets\holidays\outputs\\ranks\\'

    #LIST IMG FILES IN DIRECTORY
    files = [f for f in os.listdir(imgs_path) if os.path.isfile(os.path.join(imgs_path, f))]
    #QUERY IMAGE
    query_index = random.randint(0,1490)
    query_name = rm_ext(files[query_index])

    #COMPUTATIONS
    closests = c.cluster_query(query_index, 10)
    ppranks = ppr.ppr_query(query_index)
    pranks = pr.pr_query(query_index)

    #SAVING
    c.save_closest(outpath, query_name, closests)
    ppr.save_ppr(ppranks, query_name, 10)
    pr.save_pr(pranks, query_name, 10)


def rm_ext(f):
    return os.path.splitext(f)[0]

main()