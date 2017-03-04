from MLGraph import MLGraph
from ProbabilityToStay import ProbabilityToStay
from Distances import Distances
from GroundTruthMF import IndicatorMF
from RandomWalk import RandomWalk
import GroundTruthMF as gt
import matplotlib.pyplot as plt

def main():
    path = 'D:\cours\MA1\Semester Project\datasets\mirflickr_datasets\mirflickr25k_annotations'
    n_img = 2500
    k = 25
    query_index = 4
    steps = [1,5,10,15,20,25,30,35,40]
    ndcg = []
    query_name = 'im'+str(query_index+1)+'.jpg'
    mlg = MLGraph('mirflickr', n_img)
    #DICTIONARY FROM FEATURE TO CORRESPONDING GRAPH
    graphs = mlg.get_graphs()
    probability = []
    for _ in range(n_img):
        probability.append([1. for i in xrange(len(graphs.items()))])
    rw = RandomWalk(graphs.values())

    for s in steps:
        ranks = rw.make_n_random_walk_steps(query_index, probability, s)
        topK = map(lambda x: 'im'+str(x[0]+1)+'.jpg', ranks[1:k+1])
        ndcg.append(gt.ndcg_evaluation_from_query_node(query_name, topK, k))
    print steps
    print ndcg
    plt.plot(steps, ndcg)
    plt.show()




    # ps = ProbabilityToStay()
    # indicator = IndicatorMF(path)
    # labeled_nodes = indicator.img2cat()
    # labeled_filtered = dict(filter(lambda (x1,x2): x1 < n_img,labeled_nodes.items()))
    # d = Distances()
    # probas = []
    # for k,v in graphs.iteritems():
    #     p = ps.get_probability_using_neighbours_in_radius_sigma_inverted_distance(v, labeled_filtered, 0.3)
    #     probas.append((k,p))
    #     print 'Graph:'+k
    #     print p
    #     print len(p)

main()

