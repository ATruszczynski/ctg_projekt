import copy

import networkx as nx
from graph_utility import *
from os import listdir
from os.path import isfile, join


def read_graph_file(path: str) -> nx.Graph:
    graph_file = open(path, 'r')
    lines = graph_file.readlines()

    graph = nx.Graph()

    for line in lines:
        elements = line.split()

        if len(elements) < 2:
            continue

        type = elements[0]

        if type == 'p':
            graph.add_nodes_from(range(int(elements[2])))
        elif type == 'e':
            graph.add_edge(int(elements[1]) - 1, int(elements[2]) - 1)
        elif type == 'w':
            graph.nodes[int(elements[1]) - 1][weight_key] = int(elements[2])

    return graph


def weight_graph_randomly(graph: nx.Graph, minWeight: int, maxWeight: int, seed:int) -> nx.Graph:  # uzupełnia brakujące(!) wagi
    g = copy.deepcopy(graph)
    random.seed(seed)
    weights = []
    for v in graph.nodes:
        if weight_key not in graph.nodes[v]:
            weights.append(random.randrange(minWeight, maxWeight))
        else:
            weights.append(graph.nodes[v][weight_key])

    return weight_graph(g, weights)


# zwraca pary (graf, nazwa_grafu)
def read_many_graph_files(mypath: str, minWeight: int = 1, maxWeight: int = 100, *args) -> [
    (nx.Graph, str)]:  # wczytuje pliki z grafami od indeksu w porządku alfabetycznym
    graph_files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".col")]
    graph_files.sort()

    result = []
    if len(args) % 2 == 1 or len(args) <= 0:
        raise Exception("Incorrect amount of parameters")

    for pair_ind in range(0, len(args), 2):
        start_ind = args[pair_ind]
        how_many = args[pair_ind + 1]
        for ind in range(start_ind, start_ind + how_many):
            if ind >= len(graph_files):
                break
            gf = graph_files[ind]
            ftl = gf[0:3]
            seed = ord(ftl[0]) * ord(ftl[1]) * ord(ftl[2]) # jednoznaczne ziarno oparte na nazwie grafu (żeby graf miał zawsze tak samo ważone wierzchołki)

            graph = read_graph_file(join(mypath, gf))
            graph = weight_graph_randomly(graph, minWeight, maxWeight + 1, seed)
            name_parts = gf.split('.')
            name = ""
            for i in range(0, len(name_parts) - 1):
                name += name_parts[i]
            result.append((graph, name))

    return result
