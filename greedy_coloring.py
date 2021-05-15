import random
from copy import copy

import matplotlib.pyplot as plt
import networkx as nx

from graph_set_preparator import check_if_coloring_is_proper


def get_greedy_coloring(graph: nx.Graph) -> nx.Graph:
    color_map = [0] * len(graph.nodes)
    colors = range(0,len(color_map))
    max_weights = {color : 0 for color in range(len(colors))}

    colored_vertices = [-1] * len(graph.nodes)
    uncolored_vertices = list(range(0,len(graph.nodes)))
    while len(uncolored_vertices)>0:
        v_idx = random.randint(0,len(uncolored_vertices)-1)
        v = uncolored_vertices[v_idx]
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
        colored_vertices[v] = v
        uncolored_vertices.remove(v)
        # print(f"Coloured vertex {v} with colour {colors[color]}")
        max_weights[color] = max(max_weights[color],graph.nodes[v]['w'])

    if check_if_coloring_is_proper(graph):
        print(f"Greedy used {sum(1 for i in max_weights.values() if i > 0)} colours")
        # print(f"Greedy vertex coloring = {color_map}")
        nx.draw(graph, with_labels=True, node_color=color_map)
        plt.show()
    else:
        print("Improper colouring has been found")

    return graph
