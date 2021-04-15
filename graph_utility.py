import networkx as nx
import random
import string

color = 'c'
weight = 'w'
weight_sum = 'ws'

def color_graph(graph: nx.Graph, colors: [int]) -> nx.Graph:
    graph = graph.copy()

    if len(graph.nodes) != len(colors):
        raise Exception('Numbers of colors and vertices are different!')

    for i in range(0, len(colors)):
        graph.nodes[i][color] = colors[i]

    return graph

def weight_graph(graph: nx.Graph, weights: [int]) -> nx.Graph:
    graph = graph.copy()
    sum_of_wieghts = sum(weights)
    graph.graph[weight_sum] = sum_of_wieghts

    if len(graph.nodes) != len(weights):
        raise Exception('Numbers of weights and vertices are different!')

    for i in range(0, len(weights)):
        graph.nodes[i][weight] = weights[i]

    return graph




def coloring_cost(graph: nx.Graph) -> int:
    colors_weights = {}
    for v in graph.nodes:
        c = graph.nodes[v][color]
        w = graph.nodes[v][weight]

        if c in colors_weights:
            if w > colors_weights[c]:
                colors_weights[c] = w
        else:
            colors_weights[c] = w

    return sum(colors_weights.values())

def get_colors_of_neighbors(graph: nx.Graph, vertex: int) -> [int]:
    adj = graph.adj[vertex]
    adj_colors = []

    for v in adj:
        if color in graph.nodes[v]:
            c = graph.nodes[v][color]
            if c not in adj_colors:
                adj_colors.append(c)

    return adj_colors

def greedy_coloring(graph: nx.Graph, order: [int] = None) -> nx.Graph:
    if order is None:
        order = nx.dfs_preorder_nodes(graph)

    graph = graph.copy()

    for v in order:
        adj_colors = get_colors_of_neighbors(graph, v)
        c_col = 0
        while True:
            if c_col in adj_colors:
                c_col = c_col + 1
            else:
                break

        graph.nodes[v][color] = c_col

    return graph

def random_correct_coloring(graph: nx.Graph) -> nx.Graph:
    graph = graph.copy()

    for v in graph.nodes:
        used_colors = get_colors_used(graph)
        possible_colors = []
        adj_colors = get_colors_of_neighbors(graph, v)
        for c in used_colors:
            if c not in adj_colors:
                possible_colors.append(c)

        possible_colors.append(lowest_unused_color(used_colors))
        choice = random.randrange(len(possible_colors))
        graph.nodes[v][color] = possible_colors[choice]

    return graph

def get_coloring(graph: nx.Graph) -> [int]:
    coloring = []
    for (v, c) in graph.nodes.data(color):
        coloring.append(c)

    return coloring

def get_colors_used(graph: nx.Graph) -> [int]:
    colors = []
    for (v, c) in graph.nodes.data(color):
        if c not in colors and c is not None:
            colors.append(c)

    return colors

def check_if_you_can_color_vertex(graph: nx.Graph, vertex: int, color: int) -> bool:
    neigh_col = get_colors_of_neighbors(graph, vertex)

    return color not in neigh_col

def create_random_permutation(values: [int]) -> [int]:
    options = list(range(len(values)))
    permutation = []
    for i in range(len(values)):
        choice = random.randrange(len(options))
        permutation.append(values[options[choice]])
        del options[choice]

    return permutation

def get_color_counts(graph: nx.Graph):
    color_count = {}

    for v in graph:
        c = graph.nodes[v][color]

        if c in color_count:
            color_count[c] += 1
        else:
            color_count[c] = 1

    return color_count

def print_graph(graph: nx.Graph):
    for v in graph.nodes:
        w = graph.nodes[v][weight]
        c = graph.nodes[v][color]

        print(f'Vertex: {v} - weight: {w} - color: {c}')

def print_many_graphs(graphs: [nx.Graph]):
    for i in range(len(graphs)):
        print(f'Graph {i + 1}')
        print_graph(graphs[i])

def lowest_unused_color(colors: [int]):
    chosen_color = 0
    while True:
        if chosen_color not in colors:
            break
        else:
            chosen_color = chosen_color + 1

    return chosen_color







