from graph_utility import *
import networkx as nx
import random
import math
import multiprocessing as mp

random.seed(1001)

score = 's'

def fitness_func(graph: nx.Graph) -> int:
    cost = coloring_cost(graph)
    weigt_s = graph.graph[weight_sum]

    diff_cols = len(get_colors_used(graph))

    graph_score = (weigt_s - cost)/weigt_s / math.log2(diff_cols)

    graph.graph[score] = graph_score

    return graph_score

def mutation(graph: nx.Graph) -> nx.Graph:
    graph = graph.copy()
    index = random.randrange(len(graph.nodes))
    adj_colors = get_colors_of_neighbors(graph, index)

    used_colors = get_colors_used(graph)

    max_c = None
    max_score = fitness_func(graph)
    for c in used_colors:
        if c is not graph.nodes[index][color] and c not in adj_colors:
            graph_c = graph.copy()
            graph_c.nodes[index][color] = c
            score = fitness_func(graph_c)
            if score > max_score:
                max_score = score
                max_c = c

    if max_c is not None:
        chosen_color = max_c
    else:
        chosen_color = lowest_unused_color(used_colors)

    graph.nodes[index][color] = chosen_color

    return graph

def crossover(graph1: nx.Graph, graph2: nx.Graph) -> (nx.Graph, nx.Graph):
    graph1 = graph1.copy()
    graph2 = graph2.copy()

    used_col_1 = get_colors_used(graph1)
    col_1 = used_col_1[random.randrange(len(used_col_1))]
    used_col_2 = get_colors_used(graph2)
    col_2 = used_col_2[random.randrange(len(used_col_2))]

    for v in graph1.nodes:
        if graph2.nodes[v][color] == col_2 and check_if_you_can_color_vertex(graph1, v, col_1):
            graph1.nodes[v][color] = col_1
        if graph1.nodes[v][color] == col_1 and check_if_you_can_color_vertex(graph2, v, col_2):
            graph2.nodes[v][color] = col_2

    return graph1, graph2

def tournament_selection(colorings: [nx.Graph], participant_count: int) -> nx.Graph:
    if participant_count > len(colorings):
        participant_count = len(colorings)

    options = list(range(len(colorings)))
    chosen = []
    for i in range(participant_count):
        choice = random.randrange(len(options))
        chosen.append(colorings[options[choice]])
        del options[choice]

    chosen = sorted(chosen, key=lambda x: x.graph[score], reverse=True)

    return chosen[0].copy()

def genetic_coloring(graph: nx.Graph, pop_count: int, iterations: int, mprob: float, cprob: float, selected: int, verbal: bool = False) -> nx.Graph:
    population = []

    for i in range(pop_count):
        graph_g = graph.copy()
        graph_g = greedy_coloring(graph_g, create_random_permutation(range(len(graph_g.nodes))))
        # graph_g = random_correct_coloring(graph_g)
        population.append(graph_g)

    if verbal:
        print_many_graphs(population)

    for it in range(iterations):
        print(f'Iteration {it}')
        child_population = []
        for ind in range(len(population)):
            fitness_func(population[ind]) # ustawia score grafu

        while len(child_population) < pop_count:
            mt = random.uniform(0, 1)
            cr = random.uniform(0, 1)

            mutant = tournament_selection(population, selected)

            if mt < mprob:
                mutant = mutation(mutant)

            child_population.append(mutant)

            if len(child_population) >= pop_count:
                break

            cross1 = tournament_selection(population, 3)
            cross2 = tournament_selection(population, 3)

            if cr < cprob:
                cross1, cross2 = crossover(cross1, cross2)

            child_population.append(cross1)

            if len(child_population) >= pop_count:
                break

            child_population.append(cross2)

        population = child_population

        print(coloring_cost(max(population, key=lambda x: coloring_cost(x))))

        if verbal:
            print_many_graphs(population)
            print()


    population = sorted(population, key=lambda x: x.graph[score], reverse=True)

    return population[0].copy()



