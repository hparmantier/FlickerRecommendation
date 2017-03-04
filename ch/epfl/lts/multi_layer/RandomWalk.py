import heapq
from numpy import array
import networkx as nx
import logging

#logging.basicConfig(level=logging.DEBUG)
##RANDOM WALK GRAPH FROM LTS RENATA KHASANOVA CODE
class RandomWalk:
    def __init__(self, graph, decoder=None):
        self._graph = graph

        # self.reverse_graph()

        self.make_transition_probability_weighted_graph()

        if decoder is None:
            #{i:i for i in xrange(len(probabilities))}
            self._decoder_node2ind = {node:ind for node, ind in zip(graph[0].nodes(), xrange(len(graph[0].nodes())))}
            self._decoder_ind2node = {ind:node for node, ind in zip(graph[0].nodes(), xrange(len(graph[0].nodes())))}
        else:
            self._decoder_node2ind = decoder

    def get_probability_for_node(self, probabilities, node):
        return probabilities[node]


    def get_rating_value(self, rank, node):
        return rank[self._decoder_node2ind[node]]

    def get_query(self, query):
        return self._decoder_node2ind[query]

    def set_rating_value(self, rank, node, value):
        rank[self._decoder_node2ind[node]] = value


    def reverse_graph(self):
        map(lambda l: l.reverse(copy=False), self._graph)

    def get_ranking_forNDCG(self, rank):
        return [(self._decoder_ind2node[ind], value) for value, ind in zip(rank, xrange(len(rank)))]


# [0.9180568542901121, 0.9477937879739079, 0.9472593098116362]


    def make_random_walk_steps_till_convergence(self, query, probabilities, convergence_error=None, eta=0.98):
        number_of_nodes = len(self._graph[0].nodes())
        if convergence_error is None:
            convergence_error = 1. / number_of_nodes * 0.01
        new_ranks = [0 for _ in xrange(number_of_nodes)]
        new_ranks[self.get_query(query)] = 1.
        old_ranks = [0 for _ in xrange(number_of_nodes)]
        n=0
        while self.calculate_error(new_ranks, old_ranks) > convergence_error:
            old_ranks, new_ranks = new_ranks, old_ranks
            new_ranks = self.do_one_step(query, probabilities, old_ranks, eta)
            n+=1

        ranks = self.get_ranking_forNDCG(new_ranks)
        return heapq.nlargest(self._graph[0].number_of_nodes(), ranks, lambda t: t[1]), n


    def calculate_error(self, new_ranks, old_ranks):
        return sum((i-j)**2 for i, j in zip(new_ranks, old_ranks)) ** (1./2)

    def make_n_random_walk_steps(self, query, probabilities, n, eta=0.35):
        number_of_nodes = len(self._graph[0].nodes())
        ranks = [1. / number_of_nodes  for i in xrange(number_of_nodes)]

        for _ in xrange(n):
            ranks = self.do_one_step(query, probabilities, ranks, eta)
        ranks = self.get_ranking_forNDCG(ranks)
        return heapq.nlargest(self._graph[0].number_of_nodes(), ranks, lambda t: t[1])

    # private
    def do_one_step(self, query_node, probabilities, old_ranks, eta):
        # list of the rates
        rates = [array(self.do_step_for_one_layer(layer, probabilities, index_layer, old_ranks)) for layer, index_layer in zip(self._graph, xrange(len(self._graph)))]
        # sum up the rates
        rates_walking_part = reduce(lambda x,y: x+y, rates)
        rates = rates_walking_part * (eta)

        # add jumping part:
        rates[self.get_query(query_node)] += (1-eta)
        return rates

    def do_step_for_one_layer(self, layer, probability, index_layer, rank):
        return [sum([layer[self._decoder_ind2node[ind]][j]['weight'] * self.get_probability_for_node(probability, j)[index_layer] * self.get_rating_value(rank, j) for j in layer[self._decoder_ind2node[ind]]]) for ind in xrange(len(layer.nodes()))]

    @staticmethod
    def update_graph(multi_graph, n1, n2, new_weight):
        multi_graph[n1][n2]['weight'] = new_weight
        return multi_graph

    def make_transition_probability_weighted_graph(self):
        """
        This method change the graph. In new graph the weights equal to transition probabilities
        :param graph: Graph that will be changed
        :return:
        """
        degrees_of_all_layers = [layer.out_degree(weight='weight') for layer in self._graph]
        map(lambda (layer, degrees):
            map(lambda (u, v, weight): self.update_graph(layer, u, v, weight['weight'] / float(degrees[u] + 0.0000001)), layer.edges(data=True))
        , zip(self._graph, degrees_of_all_layers))


def test_1():
    graph = nx.DiGraph()
    graph.add_nodes_from(xrange(4))

    w = [{1: 0.5, 3: 0.5}, {3: 1}, {1: 0.5, 3: 0.5}, {0: 0.3, 1: 0.4, 2: 0.3}]
    for n, index in zip(w, xrange(4)):
        for k, v in n.iteritems():
            graph.add_edge(index,k, weight=v)

    probability = [1. for i in xrange(4)]
    r = RandomWalk([graph], [probability])
    r.make_n_random_walk_steps(2,20)
    r.make_random_walk_steps_till_convergence(1)

def test_2():
    graph = nx.DiGraph()
    graph.add_nodes_from(xrange(10))

    f = [{1: {'weight': 0.2881826371807402}, 2: {'weight': 0.7011249819454258}, 3: {'weight': 0.34637130968632907}, 4: {'weight': 0.3373179295465066}, 5: {'weight': 0.7030568062038827}, 6: {'weight': 0.6594891222589647}, 7: {'weight': 2.9324258260856147e-06}, 8: {'weight': 0.3423324176201711}, 9: {'weight': 0.5055209844804006}}, {0: {'weight': 0.2881826371807402}, 2: {'weight': 0.20813552209346683}, 3: {'weight': 0.1276147823907101}, 4: {'weight': 0.8396978110965048}, 5: {'weight': 0.4169765394357211}, 6: {'weight': 0.2097257037624973}, 7: {'weight': 0.0012854431017619284}, 8: {'weight': 0.14204922253105837}, 9: {'weight': 0.627459408360586}}, {0: {'weight': 0.7011249819454258}, 1: {'weight': 0.20813552209346683}, 3: {'weight': 0.29950731559293586}, 4: {'weight': 0.21276454980590903}, 5: {'weight': 0.454212810905543}, 6: {'weight': 0.5522877977826267}, 7: {'weight': 8.536905374239662e-07}, 8: {'weight': 0.43499464362631024}, 9: {'weight': 0.4123855888378498}}, {0: {'weight': 0.34637130968632907}, 1: {'weight': 0.1276147823907101}, 2: {'weight': 0.29950731559293586}, 4: {'weight': 0.2191517039620457}, 5: {'weight': 0.37785736224977384}, 6: {'weight': 0.422838699175566}, 7: {'weight': 1.5333398181677222e-07}, 8: {'weight': 0.6713928437420265}, 9: {'weight': 0.13113707108395709}}, {0: {'weight': 0.3373179295465066}, 1: {'weight': 0.8396978110965048}, 2: {'weight': 0.21276454980590903}, 3: {'weight': 0.2191517039620457}, 5: {'weight': 0.5177722633063939}, 6: {'weight': 0.24083944074024846}, 7: {'weight': 0.00048113689661720963}, 8: {'weight': 0.25365338097875784}, 9: {'weight': 0.5926547364395025}}, {0: {'weight': 0.7030568062038827}, 1: {'weight': 0.4169765394357211}, 2: {'weight': 0.454212810905543}, 3: {'weight': 0.37785736224977384}, 4: {'weight': 0.5177722633063939}, 6: {'weight': 0.7550023506244243}, 7: {'weight': 2.4634718352311803e-05}, 8: {'weight': 0.492554892126609}, 9: {'weight': 0.38876518211543454}}, {0: {'weight': 0.6594891222589647}, 1: {'weight': 0.2097257037624973}, 2: {'weight': 0.5522877977826267}, 3: {'weight': 0.422838699175566}, 4: {'weight': 0.24083944074024846}, 5: {'weight': 0.7550023506244243}, 7: {'weight': 1.8471292550632583e-06}, 8: {'weight': 0.49849003473179043}, 9: {'weight': 0.20550418983997462}}, {0: {'weight': 2.9324258260856147e-06}, 1: {'weight': 0.0012854431017619284}, 2: {'weight': 8.536905374239662e-07}, 3: {'weight': 1.5333398181677222e-07}, 4: {'weight': 0.00048113689661720963}, 5: {'weight': 2.4634718352311803e-05}, 6: {'weight': 1.8471292550632583e-06}, 8: {'weight': 3.596038724859438e-07}, 9: {'weight': 9.381764527550599e-05}}, {0: {'weight': 0.3423324176201711}, 1: {'weight': 0.14204922253105837}, 2: {'weight': 0.43499464362631024}, 3: {'weight': 0.6713928437420265}, 4: {'weight': 0.25365338097875784}, 5: {'weight': 0.492554892126609}, 6: {'weight': 0.49849003473179043}, 7: {'weight': 3.596038724859438e-07}, 9: {'weight': 0.12563826563961714}}, {0: {'weight': 0.5055209844804006}, 1: {'weight': 0.627459408360586}, 2: {'weight': 0.4123855888378498}, 3: {'weight': 0.13113707108395709}, 4: {'weight': 0.5926547364395025}, 5: {'weight': 0.38876518211543454}, 6: {'weight': 0.20550418983997462}, 7: {'weight': 9.381764527550599e-05}, 8: {'weight': 0.12563826563961714}}]

    for n, index in zip(f, xrange(10)):
        for k, v in n.iteritems():
            graph.add_edge(index, k, weight=v['weight'])

    probability = [[1.] for i in xrange(10)]
    print probability
    r = RandomWalk([graph])
    print 'n step...'
    print r.make_n_random_walk_steps(4,probability, 7)
    print 'convergence...'
    ranks, n = r.make_random_walk_steps_till_convergence(4,  probability)
    print ranks
    print n


test_2()
