from networkx import Graph

THRESHOLD = 0.1

class Distances:
    def get_inverted_distance_path(self, graph, threshold=THRESHOLD):
        return {node: self.get_inverted_distance_node(graph, node, threshold)[0] for node in graph.nodes()}

    def get_inverted_distance(self, graph, threshold=THRESHOLD):
        return {node: self.get_inverted_distance_node(graph, node, threshold)[1] for node in graph.nodes()}


    def get_inverted_distance_node(self, graph, node, threshold):
        dist = {node:1}
        path = {node:[node]}
        color = set()
        self.__get_inverted_distance_node(color, graph, dist, path, node, threshold)
        return path, dist

    def __get_inverted_distance_node(self, color, graph, dist, path, current_node, threshold):
        if current_node in color:
            return
        color.add(current_node)
        for neighbour_node in graph[current_node].keys():
            if neighbour_node in dist:
                if dist[current_node] * graph[current_node][neighbour_node]['weight'] <= dist[neighbour_node]:
                    pass
                else:
                    dist[neighbour_node] = dist[current_node] * graph[current_node][neighbour_node]['weight']
                    if dist[neighbour_node] > threshold:
                        if not neighbour_node in path:
                            path[neighbour_node] = [i for i in path[current_node]]
                        path[neighbour_node].append(neighbour_node)
                        self.__get_inverted_distance_node(color, graph, dist, path, neighbour_node, threshold)

            else:
                dist[neighbour_node] = dist[current_node] * graph[current_node][neighbour_node]['weight']
                if dist[neighbour_node] > threshold:
                    if not neighbour_node in path:
                        path[neighbour_node] = [i for i in path[current_node]]
                    path[neighbour_node].append(neighbour_node)
                    self.__get_inverted_distance_node(color, graph, dist, path, neighbour_node, threshold)






if __name__ == '__main__':
    number_nodes = 10
    graph = Graph()
    graph.add_nodes_from(xrange(10))
    # 2: {'weight': 0.7011249819454258}, 0: {'weight': 0.7011249819454258}
    # f = [{1: {'weight': 0.2881826371807402}, 3: {'weight': 0.34637130968632907}, 4: {'weight': 0.3373179295465066}, 5: {'weight': 0.7030568062038827}, 6: {'weight': 0.6594891222589647}, 7: {'weight': 2.9324258260856147e-06}, 8: {'weight': 0.3423324176201711}, 9: {'weight': 0.5055209844804006}},
    #      {0: {'weight': 0.2881826371807402}, 2: {'weight': 0.20813552209346683}, 3: {'weight': 0.1276147823907101}, 4: {'weight': 0.8396978110965048}, 5: {'weight': 0.4169765394357211}, 6: {'weight': 0.2097257037624973}, 7: {'weight': 0.0012854431017619284}, 8: {'weight': 0.14204922253105837}, 9: {'weight': 0.627459408360586}},
    #      {1: {'weight': 0.20813552209346683}, 3: {'weight': 0.29950731559293586}, 4: {'weight': 0.21276454980590903}, 5: {'weight': 0.454212810905543}, 6: {'weight': 0.5522877977826267}, 7: {'weight': 8.536905374239662e-07}, 8: {'weight': 0.43499464362631024}, 9: {'weight': 0.4123855888378498}}, {0: {'weight': 0.34637130968632907}, 1: {'weight': 0.1276147823907101}, 2: {'weight': 0.29950731559293586}, 4: {'weight': 0.2191517039620457}, 5: {'weight': 0.37785736224977384}, 6: {'weight': 0.422838699175566}, 7: {'weight': 1.5333398181677222e-07}, 8: {'weight': 0.6713928437420265}, 9: {'weight': 0.13113707108395709}}, {0: {'weight': 0.3373179295465066}, 1: {'weight': 0.8396978110965048}, 2: {'weight': 0.21276454980590903}, 3: {'weight': 0.2191517039620457}, 5: {'weight': 0.5177722633063939}, 6: {'weight': 0.24083944074024846}, 7: {'weight': 0.00048113689661720963}, 8: {'weight': 0.25365338097875784}, 9: {'weight': 0.5926547364395025}}, {0: {'weight': 0.7030568062038827}, 1: {'weight': 0.4169765394357211}, 2: {'weight': 0.454212810905543}, 3: {'weight': 0.37785736224977384}, 4: {'weight': 0.5177722633063939}, 6: {'weight': 0.7550023506244243}, 7: {'weight': 2.4634718352311803e-05}, 8: {'weight': 0.492554892126609}, 9: {'weight': 0.38876518211543454}}, {0: {'weight': 0.6594891222589647}, 1: {'weight': 0.2097257037624973}, 2: {'weight': 0.5522877977826267}, 3: {'weight': 0.422838699175566}, 4: {'weight': 0.24083944074024846}, 5: {'weight': 0.7550023506244243}, 7: {'weight': 1.8471292550632583e-06}, 8: {'weight': 0.49849003473179043}, 9: {'weight': 0.20550418983997462}}, {0: {'weight': 2.9324258260856147e-06}, 1: {'weight': 0.0012854431017619284}, 2: {'weight': 8.536905374239662e-07}, 3: {'weight': 1.5333398181677222e-07}, 4: {'weight': 0.00048113689661720963}, 5: {'weight': 2.4634718352311803e-05}, 6: {'weight': 1.8471292550632583e-06}, 8: {'weight': 3.596038724859438e-07}, 9: {'weight': 9.381764527550599e-05}}, {0: {'weight': 0.3423324176201711}, 1: {'weight': 0.14204922253105837}, 2: {'weight': 0.43499464362631024}, 3: {'weight': 0.6713928437420265}, 4: {'weight': 0.25365338097875784}, 5: {'weight': 0.492554892126609}, 6: {'weight': 0.49849003473179043}, 7: {'weight': 3.596038724859438e-07}, 9: {'weight': 0.12563826563961714}}, {0: {'weight': 0.5055209844804006}, 1: {'weight': 0.627459408360586}, 2: {'weight': 0.4123855888378498}, 3: {'weight': 0.13113707108395709}, 4: {'weight': 0.5926547364395025}, 5: {'weight': 0.38876518211543454}, 6: {'weight': 0.20550418983997462}, 7: {'weight': 9.381764527550599e-05}, 8: {'weight': 0.12563826563961714}}]
    f = [{1: {'weight':0.5}, 2:{'weight':0.5}}, {3: {'weight':0.5}}]
    for n, index in zip(f, xrange(10)):
        for k,v in n.iteritems():
            graph.add_edge(index, k, weight=v['weight'])

    print graph.edges(data=True)
    node = 0
    labeled_nodes = {1:['a', 'b'], 2:['a', 'c'], 9:['a', 'b', 'd']}
    radius = 2

    d = Distances()
    print d.get_inverted_distance_path(graph, 0.5)

