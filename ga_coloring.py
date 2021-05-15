from graph_utility import *
import networkx as nx
import random
import math
import multiprocessing as mp
import os

random.seed(1001)

score_id = 's'
mut_id = 'M'
cross_id = 'C'


def genetic_coloring(graph: nx.Graph, pop_count: int, iterations: int, mprob: float, cprob: float, selected: int,
                     verbal: int = 0, pool_count: int = 12, patience: int = 5, fix_prob: float = 0.1,
                     random_init: bool = False) -> nx.Graph:
    pool = mp.Pool(pool_count)

    if verbal >= 1:
        print("Prep start")
    population = []
    for i in range(pop_count):
        population.append(graph.copy())

    if random_init:
        population = quick_graph_coloring_async(pool, color_random_many_graphs, population)
    else:
        population = quick_graph_coloring_async(pool, color_greed_many_graphs, population)

    if verbal >= 1:
        print("Prep end")

    if verbal >= 2:
        print_many_graphs(population)

    curr_best = -math.inf
    curr_patience = 0

    if verbal >= 1:
        print(f'Iteration')

    for it in range(iterations):
        if verbal >= 1:
            if it == 0:
                print(f'{it}', end="")
            else:
                print(f',{it}', end="")

        for ind in range(len(population)):
            fitness_func(population[ind])  # ustawia score grafu

        next_pop_count = 0
        crossover_tasks = []
        to_pass = []

        while next_pop_count < pop_count:
            cr = random.uniform(0, 1)
            cross1 = tournament_selection(population, selected)
            if cr < cprob and next_pop_count <= pop_count - 2:
                cross2 = tournament_selection(population, selected)
                crossover_tasks.append([cross_id, cross1, cross2])
                next_pop_count += 2
            else:
                to_pass.append(cross1)
                next_pop_count += 1

        processed = async_process_genetic_tasks(pool, pool_count, crossover_tasks)
        processed.extend(to_pass)

        assert len(processed) == pop_count

        mutate_tasks = []
        to_pass = []

        for p in processed:
            mutate_tasks.append([mut_id, p, mprob])

        processed = async_process_genetic_tasks(pool, pool_count, mutate_tasks)

        child_population = []
        child_population.extend(processed)
        child_population.extend(to_pass)

        assert len(child_population) == pop_count
        for cg_i in range(len(child_population)):
            cg = child_population[cg_i]
            ok = check_if_coloring_is_proper(cg)
            assert ok

        population = child_population

        local_best = -math.inf
        for g in population:
            if g.graph[score_id] > local_best:
                local_best = g.graph[score_id]

        if local_best > curr_best:
            curr_best = local_best
            curr_patience = 0
        else:
            curr_patience += 1

        if curr_patience == patience:
            if verbal >= 1:
                print("\nPatience run out!")
            break

        if verbal >= 1:
            coloring_cost_min = math.inf
            coloring_cost_avg = 0
            for g in population:
                col_cost = get_coloring_cost(g)
                if col_cost < coloring_cost_min:
                    coloring_cost_min = col_cost
                coloring_cost_avg += col_cost
            coloring_cost_avg = round(coloring_cost_avg / len(population), 2)

            print(
                f"({coloring_cost_min}, {coloring_cost_avg}, {round((coloring_cost_avg - coloring_cost_min) / coloring_cost_avg, 2)})",
                end="")
            if it > 0 and it % 10 == 0:
                print()

        if verbal >= 2:
            print_many_graphs(population)
            print()
    if verbal >= 1:
        print()

    pool.close()
    for ind in range(len(population)):
        fitness_func(population[ind])  # ustawia score graów

    population = sorted(population, key=lambda x: x.graph[score_id], reverse=True)

    best = population[0]

    best = attempt_to_eliminate_small_colors(best, prob=fix_prob)

    return best.copy()


def fitness_func(graph: nx.Graph) -> int:
    cost = get_coloring_cost(graph)
    weigt_s = graph.graph[weight_sum_key]

    diff_cols = len(get_used_colors(graph))

    graph_score = (weigt_s - cost) / weigt_s

    graph.graph[score_id] = graph_score

    return graph_score


def crossover(graph1: nx.Graph, graph2: nx.Graph) -> (nx.Graph, nx.Graph):
    graph1 = graph1.copy()
    graph2 = graph2.copy()

    used_col_1 = get_used_colors(graph1)
    col_1 = used_col_1[random.randrange(len(used_col_1))]
    used_col_2 = get_used_colors(graph2)
    col_2 = used_col_2[random.randrange(len(used_col_2))]

    for v in graph1.nodes:
        if graph2.nodes[v][color_key] == col_2 and check_if_you_can_color_vertex(graph1, v, col_1):
            graph1.nodes[v][color_key] = col_1
        if graph1.nodes[v][color_key] == col_1 and check_if_you_can_color_vertex(graph2, v, col_2):
            graph2.nodes[v][color_key] = col_2

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

    chosen = sorted(chosen, key=lambda x: x.graph[score_id], reverse=True)

    return chosen[0].copy()


def mutation(graph: nx.Graph, ver_mut_prob: float) -> nx.Graph:
    # print('Mutation')
    # print(os.getpid())

    for i in range(len(graph.nodes)):
        mr = random.random()
        if mr < ver_mut_prob:
            graph = local_reduction(graph, index=i)

    return graph


def attempt_to_eliminate_small_colors(graph: nx.Graph, prob: float = 0.1):
    graph = graph.copy()
    color_count = get_color_counts(graph)
    colors_to_eliminate = []

    for col in color_count:
        if color_count[col] < len(graph.nodes) * 0.1:
            colors_to_eliminate.append(col)

    for v in graph.nodes:
        if graph.nodes[v][color_key] in colors_to_eliminate:
            if random.uniform(0, 1) < prob:
                graph = local_reduction(graph, v, allow_increase=False)

    return graph


def local_reduction(graph: nx.Graph, index: int, allow_increase: bool = True):
    graph = graph.copy()
    adj_colors = get_colors_of_neighbors(graph, index)
    curr_color = graph.nodes[index][color_key]
    curr_weigh = graph.nodes[index][weight_key]
    used_colors = get_used_colors(graph)
    values_in_color = get_weights_in_color(graph)

    min_c = None
    min_diff = 0

    # to nie jest ani ładne, ani czytelne, ale za to jest szybkie
    if values_in_color[curr_color][
        0] == curr_weigh:  # wybrany wierzchołek jest najcięższy w swoim kolorze, czyli przerzucenie go do innego koloru może poprawić wynik grafu
        # wpp. przerzucenie do innego koloru jest neutralne lub pogarsza
        for c in used_colors:
            if c is not curr_color and c not in adj_colors:
                curr_diff = math.inf
                other_weight = values_in_color[c][0]

                second_weight = 0
                if len(values_in_color[curr_color]) > 1:
                    second_weight = values_in_color[curr_color][1]

                if curr_weigh < other_weight:  # przerzucenie to tego koloru to zysk
                    curr_diff = second_weight - curr_weigh

                if curr_weigh >= other_weight:  # przerzucenie do tego koloru to MOŻE zysk
                    curr_diff = curr_weigh - other_weight

                if curr_diff < min_diff:
                    min_diff = curr_diff
                    min_c = c

    if min_c is not None:
        chosen_color = min_c
    elif allow_increase:
        chosen_color = get_lowest_natural_not_on_list(used_colors)
    else:
        chosen_color = graph.nodes[index][color_key]

    graph.nodes[index][color_key] = chosen_color

    return graph


def async_process_genetic_tasks(pool, pool_count, tasks):
    number_of_buckets = pool_count
    task_buckets = []
    for b in range(number_of_buckets):
        task_buckets.append([])

    for t_ind in range(len(tasks)):
        task_buckets[t_ind % number_of_buckets].append(tasks[t_ind])
    processed = [pool.apply_async(genetic_tasks, args=(t,)) for t in task_buckets]
    [bucket.wait() for bucket in processed]
    processed = [result for bucket in processed for task in bucket.get(timeout=1) for result in task]

    return processed


def genetic_tasks(graphs: [[nx.Graph]]) -> [[nx.Graph]]:
    results = []
    for g in graphs:
        if g[0] == mut_id:
            results.append([mutation(g[1], g[2])])
        if g[0] == cross_id:
            results.append([item for item in crossover(g[1], g[2])])
    return results


def quick_graph_coloring_async(pool: mp.Pool, color_func, tasks):
    pool_count = pool._processes
    number_of_buckets = pool_count
    task_buckets = []
    for b in range(number_of_buckets):
        task_buckets.append([])

    for t_ind in range(len(tasks)):
        task_buckets[t_ind % number_of_buckets].append(tasks[t_ind])

    processed = [pool.apply_async(color_func, args=(t,)) for t in task_buckets]
    [bucket.wait() for bucket in processed]
    processed = [task for bucket in processed for task in bucket.get(timeout=1)]

    return processed


def color_greed_many_graphs(graphs: [nx.Graph]):
    colored = []
    for g in graphs:
        colored.append(greedy_coloring(g))

    return colored


def color_random_many_graphs(graphs: [nx.Graph]):
    colored = []
    for g in graphs:
        colored.append(random_proper_coloring(g))

    return colored
