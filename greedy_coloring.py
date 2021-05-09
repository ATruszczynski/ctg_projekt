from copy import copy

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from graph_set_preparator import read_graph_file, print_graph, weight_graph_randomly, check_if_coloring_is_proper

example_path = '/Users/tomek/Workspace/ctg_projekt/instances/david.col'

graph = read_graph_file(path=example_path)
graph = weight_graph_randomly(graph, 10, 100).copy()
nx.draw(graph, with_labels=True)
plt.show()


def get_greedy_coloring(graph: nx.Graph) -> nx.Graph:
    color_map = [0] * len(graph.nodes)
    colors = range(0,len(color_map))
    max_weights = {color : 0 for color in range(len(colors))}

    for v in graph.nodes():
        available_colors = {i: True for i in range(0,len(colors))}
        for current_color in available_colors.keys():
            for nbr in graph.adj[v]:
                if 'c' in graph.nodes[nbr] and graph.nodes[nbr]['c'] == current_color:
                    available_colors[current_color] = False

        new_sums = [-1] * len(max_weights.keys())

        for color in available_colors.keys():
            if available_colors[color]:
                v_weight = graph.nodes[v]['w']
                tmp_weights = copy(max_weights)
                tmp_weights[color] = max(tmp_weights[color],v_weight)
                new_sums[color]=(sum(tmp_weights.values()))

        color = new_sums.index(min(i for i in new_sums if i > -1))
        graph.nodes[v]['c'] = color
        color_map[v] = colors[color]
        # print(f"Coloured vertex {v} with colour {colors[color]}")
        max_weights[color] = max(max_weights[color],graph.nodes[v]['w'])

    if check_if_coloring_is_proper(graph):
        print(f"Used {sum(1 for i in max_weights.values() if i > 0)} colours")
        nx.draw(graph, with_labels=True, node_color=color_map)
        plt.show()
    else:
        print("Improper colouring has been found")

    return graph


get_greedy_coloring(graph)
