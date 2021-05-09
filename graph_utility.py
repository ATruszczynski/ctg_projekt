import networkx as nx
import random
import string

color_key = 'c'
weight_key = 'w'
weight_sum_key = 'ws'


# proste kolorowania

def greedy_coloring(graph: nx.Graph, order: [int] = None) -> nx.Graph:
    if order is None:
        order = get_random_permutation_of_list(list(range(0, len(graph.nodes))))

    graph = graph.copy()

    for v in order:
        adj_colors = get_colors_of_neighbors(graph, v)
        lowest_color = get_lowest_natural_not_on_list(adj_colors)
        graph.nodes[v][color_key] = lowest_color

    return graph


def random_proper_coloring(graph: nx.Graph) -> nx.Graph:
    graph = graph.copy()

    for v in graph.nodes:
        used_colors = get_used_colors(graph)
        adj_colors = get_colors_of_neighbors(graph, v)
        valid_colors = []
        for c in used_colors:
            if c not in adj_colors:
                valid_colors.append(c)

        valid_colors.append(get_lowest_natural_not_on_list(used_colors))
        choice = random.randrange(len(valid_colors))
        graph.nodes[v][color_key] = valid_colors[choice]

    return graph


# operacje na grafach

def weight_graph(graph: nx.Graph, weights: [int]) -> nx.Graph:
    graph = graph.copy()
    graph.graph[weight_sum_key] = sum(weights)

    if len(graph.nodes) != len(weights):
        raise Exception('Numbers of weights and vertices are different!')

    for i in range(0, len(weights)):
        graph.nodes[i][weight_key] = weights[i]

    return graph


# informacje o grafach / kolorowaniach

def get_coloring_cost(graph: nx.Graph) -> int:
    colors_weights = {}
    for v in graph.nodes:
        c = graph.nodes[v][color_key]
        w = graph.nodes[v][weight_key]

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
        if color_key in graph.nodes[v]:
            c = graph.nodes[v][color_key]
            if c not in adj_colors:
                adj_colors.append(c)

    return adj_colors


def check_if_coloring_is_proper(graph: nx.Graph) -> bool:
    for v in graph.nodes:
        cv = graph.nodes[v][color_key]
        neighbors = graph.adj[v]
        for n in neighbors:
            if v == n:
                continue
            cn = graph.nodes[n][color_key]
            if cv == cn:
                return False
    return True


def get_used_colors(graph: nx.Graph) -> [int]:
    colors = []
    for (v, c) in graph.nodes.data(color_key):
        if c not in colors and c is not None:
            colors.append(c)

    return colors


def check_if_you_can_color_vertex(graph: nx.Graph, vertex: int, color: int) -> bool:
    neigh_col = get_colors_of_neighbors(graph, vertex)

    return color not in neigh_col


def get_color_counts(graph: nx.Graph):  # ile wierzchołków jest w poszczególnych kolorach
    color_count = {}

    for v in graph:
        c = graph.nodes[v][color_key]

        if c in color_count:
            color_count[c] += 1
        else:
            color_count[c] = 1

    return color_count


def get_weights_in_color(graph: nx.Graph):  # zwraca słownik przypisujący kolorowi malejącą
    # listę wag wierzchołków w tym kolorze
    weights_in_colors = {}

    for v in graph:
        c = graph.nodes[v][color_key]
        w = graph.nodes[v][weight_key]

        if c in weights_in_colors:
            weights_in_colors[c].append(w)
        else:
            weights_in_colors[c] = [w]

    for v in weights_in_colors:
        weights_in_colors[v].sort(reverse=True)

    return weights_in_colors


def get_lowest_natural_not_on_list(colors: [int]):
    chosen_color = 0
    while True:
        if chosen_color not in colors:
            break
        else:
            chosen_color = chosen_color + 1

    return chosen_color


# różne

def print_graph(graph: nx.Graph):
    for v in graph.nodes:
        w = graph.nodes[v][weight_key]
        c = graph.nodes[v][color_key]

        print(f'Vertex: {v} - weight: {w} - color: {c}')


def print_many_graphs(graphs: [nx.Graph]):
    for i in range(len(graphs)):
        print(f'Graph {i + 1}')
        print_graph(graphs[i])


def get_random_permutation_of_list(values: [int]) -> [int]:
    options = list(range(len(values)))
    permutation = []
    for i in range(len(values)):
        choice = random.randrange(len(options))
        permutation.append(values[options[choice]])
        del options[choice]

    return permutation


# nieużywane

def color_graph(graph: nx.Graph, colors: [int]) -> nx.Graph:
    graph = graph.copy()

    if len(graph.nodes) != len(colors):
        raise Exception('Numbers of colors and vertices are different!')

    for i in range(0, len(colors)):
        graph.nodes[i][color_key] = colors[i]

    return graph


def get_coloring(graph: nx.Graph) -> [int]:
    coloring = []
    for (v, c) in graph.nodes.data(color_key):
        coloring.append(c)

    return coloring
