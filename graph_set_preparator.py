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
            graph.nodes[int(elements[1]) - 1][weight] = int(elements[2])

    return graph

def weight_graph_randomly(graph: nx.Graph, minWeight: int, maxWeight: int) -> nx.Graph: # uzupełnia brakujące(!) wagi
    graph = graph.copy()
    weights = []
    for v in graph.nodes:
        if weight not in graph.nodes[v]:
            weights.append(random.randrange(minWeight, maxWeight))
        else:
            weights.append(graph.nodes[v][weight])

    return weight_graph(graph, weights)

def read_many_graph_files(mypath: str, start_ind: int = 0, how_many: int = 10, minWeight: int = 1, maxWeight: int = 100) -> [nx.Graph]: # wczytuje pliki z grafami od indeksu w porządku alfabetycznym
    graph_files = [f for f in listdir(mypath) if isfile(join(mypath, f)) and f.endswith(".col")]

    result = []

    for ind in range(start_ind, start_ind + how_many):
        if ind >= len(graph_files):
            break
        gf = graph_files[ind]
        ftl = gf[0:3]
        random.seed(ord(ftl[0]) * ord(ftl[1]) * ord(ftl[2])) # jednoznaczne ziarno oparte na nazwie grafu (żeby graf miał zawsze tak samo ważone wierzchołki)

        graph = read_graph_file(join(mypath, gf))
        graph = weight_graph_randomly(graph, minWeight, maxWeight)
        name_parts = gf.split('.')
        name = ""
        for i in range(0, len(name_parts) - 1):
            name += name_parts[i]
        result.append((graph, name))

    return result

