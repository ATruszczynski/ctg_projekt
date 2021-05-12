from copy import copy

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random

from graph_set_preparator import read_graph_file, print_graph, weight_graph_randomly, check_if_coloring_is_proper
from greedy_coloring import get_greedy_coloring

example_path = '/Users/tomek/Workspace/ctg_projekt/instances/miles1500.col'

graph = read_graph_file(path=example_path)
graph = weight_graph_randomly(graph, 10, 100).copy()
nx.draw(graph, with_labels=True)
plt.show()


def get_dsatur_coloring(graph: nx.Graph) -> nx.Graph :
    color_map = [-1] * len(graph.nodes)
    colors = range(0, len(color_map))
    max_weights = {color: 0 for color in range(len(colors))}
    vertex_saturation_dict = {v: 0 for v in range(len(color_map))}
    colored_vertices = {v: False for v in range(len(color_map))}
    nbr_colors = {v:set() for v in range(len(color_map))}

    while False in colored_vertices.values():

        #if we have no coloured vertex
        if sum(vertex_saturation_dict.values()) == 0:
            v = next(iter(vertex_saturation_dict))
            color = 0
            graph.nodes[v]['c'] = color
            color_map[v] = colors[color]
            colored_vertices[v] = True
        else:

            # Find the vertex with the biggest level of saturation
            biggest_saturation_value = max(vertex_saturation_dict.values())
            most_saturated_vertices = []

            for vertex, saturation_level in vertex_saturation_dict.items():
                if saturation_level == biggest_saturation_value:
                    most_saturated_vertices.append(vertex)

            # print(f"Most saturated vertices are {most_saturated_vertices}")

            #resolve first tie by chosing the biggest degree
            if len(most_saturated_vertices) > 1:
                degs = [graph.degree[v] for v in most_saturated_vertices]
                # print(f"Degree list during the tie {degs}")

                max_degs = []
                max_deg_val = max(degs)

                for idx in range(0,len(most_saturated_vertices)):
                    if degs[idx] == max_deg_val:
                        max_degs.append(most_saturated_vertices[idx])

                # resolve 2nd time randomly
                if len(max_degs) > 1:
                    v = max_degs[random.randint(0, len(max_degs)-1)]
                else:
                    v = max_degs[0]
            else:
                v = most_saturated_vertices[0]

            available_colors = {i: True for i in range(0, len(colors))}
            for current_color in available_colors.keys():
                for nbr in graph.adj[v]:
                    if 'c' in graph.nodes[nbr] and graph.nodes[nbr]['c'] == current_color:
                        available_colors[current_color] = False

            new_sums = [-1] * len(max_weights.keys())

            for color in available_colors.keys():
                if available_colors[color]:
                    v_weight = graph.nodes[v]['w']
                    tmp_weights = copy(max_weights)
                    tmp_weights[color] = max(tmp_weights[color], v_weight)
                    new_sums[color] = (sum(tmp_weights.values()))

            color = new_sums.index(min(i for i in new_sums if i > -1))
            graph.nodes[v]['c'] = color
            color_map[v] = colors[color]

            colored_vertices[v] = True

        # update saturation level

        for nbr in graph.adj[v]:
            nbr_colors[nbr].add(color)
            if nbr in vertex_saturation_dict:
                vertex_saturation_dict[nbr] = len(nbr_colors[nbr])

        # print(f"Coloured vertex {v} with colour {colors[color]}")

        if len(vertex_saturation_dict.keys()) > 0:
            del vertex_saturation_dict[v]
        max_weights[color] = max(max_weights[color], graph.nodes[v]['w'])

    if check_if_coloring_is_proper(graph):
        print(f"DSatur used {sum(1 for i in max_weights.values() if i > 0)} colours")
        nx.draw(graph, with_labels=True, node_color=color_map)
        plt.show()
    else:
        print("Improper colouring has been found")


get_dsatur_coloring(graph)
get_greedy_coloring(graph)