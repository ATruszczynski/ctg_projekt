from graph_utility import *
from ga_coloring import *
from graph_set_preparator import *


path = 'C:\\Users\\aleks\\Desktop\\inithx.i.3.col'

if __name__ == '__main__':
    graph = read_graph_file(path)
    # graph.remove_nodes_from(list(range(200, 621)))
    graph = weight_graph_randomly(graph, 10, 100).copy()

    # graph = nx.Graph()
    # graph.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    # graph.add_edges_from([(0, 3), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (6, 8), (6, 9)])
    # graph = weight_graph(graph, [20, 20, 20, 1, 1, 1, 1, 20, 20, 20])

    # graph = nx.Graph()
    # graph.add_nodes_from([0, 1, 2, 3, 4, 5, 6, 7, 8])
    # graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 0), (0, 2), (1, 3), (1, 4), (0, 4), (4, 5), (5, 6), (6, 7), (4, 7), (3, 7), (3, 5), (4, 6), (5, 7)])
    # graph = weight_graph(graph, [20, 1, 1, 20, 1, 20, 20, 1, 20])

    graph = greedy_coloring(graph)
    print()
    print_graph(graph)
    print(coloring_cost(graph))

    graph = genetic_coloring(graph, 100, 20, 0.8, 0.8, 5, patience=10)
    print()
    # print_graph(graph)
    print(coloring_cost(graph))


