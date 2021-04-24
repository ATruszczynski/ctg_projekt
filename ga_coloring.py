from graph_utility import *
import networkx as nx
import random
import math
import multiprocessing as mp
import os

random.seed(1001)

score = 's'
mut_id = 'M'
cross_id = 'C'

def fitness_func(graph: nx.Graph) -> int:
    cost = coloring_cost(graph)
    weigt_s = graph.graph[weight_sum]

    diff_cols = len(get_colors_used(graph))

    graph_score = (weigt_s - cost)/weigt_s / math.log2(diff_cols)

    graph.graph[score] = graph_score

    return graph_score

def mutation(graph: nx.Graph, ver_mut_prob: float) -> nx.Graph:
    # print('Mutation')
    # print(os.getpid())

    count = max(int(len(graph.nodes) * ver_mut_prob), 1)

    for i in range(count):
        index = random.randrange(len(graph.nodes))
        graph = local_reduction(graph, index)

    return graph

def local_reduction(graph: nx.Graph, index: int, allow_increase: bool = True):
    # graph = graph.copy()
    # adj_colors = get_colors_of_neighbors(graph, index)
    #
    # used_colors = get_colors_used(graph)
    #
    # max_c = None
    # max_score = fitness_func(graph)
    # for c in used_colors:
    #     if c is not graph.nodes[index][color] and c not in adj_colors:
    #         graph_c = graph.copy()
    #         graph_c.nodes[index][color] = c
    #         score = fitness_func(graph_c)
    #         if score > max_score:
    #             max_score = score
    #             max_c = c
    #
    # if max_c is not None:
    #     chosen_color = max_c
    # elif allow_increase:
    #     chosen_color = lowest_unused_color(used_colors)
    # else:
    #     chosen_color = graph.nodes[index][color]
    #
    # graph.nodes[index][color] = chosen_color
    #
    # return graph

    graph = graph.copy()
    adj_colors = get_colors_of_neighbors(graph, index)
    curr_color = graph.nodes[index][color]
    curr_weigh = graph.nodes[index][weight]
    used_colors = get_colors_used(graph)
    values_in_color = get_values_in_color(graph)

    min_c = None
    min_diff = 0

    # to nie jest ani ładne, ani czytelne, ale za to jest szybkie
    if values_in_color[curr_color][0] == curr_weigh: # wybrany wierzchołek jest najcięższy w swoim kolorze, czyli przerzucenie go do innego koloru może poprawić wynik grafu
                                                     # wpp. przerzucenie do innego koloru jest neutralne lub pogarsza
        for c in used_colors:
            if c is not curr_color and c not in adj_colors:
                curr_diff = math.inf
                other_weight = values_in_color[c][0]

                second_weight = 0
                if len(values_in_color[curr_color]) > 1:
                    second_weight = values_in_color[curr_color][1]

                if curr_weigh < other_weight: # przerzucenie to tego koloru to zysk
                    curr_diff = second_weight - curr_weigh

                if curr_weigh >= other_weight: # przerzucenie do tego koloru to MOŻE zysk
                    curr_diff = curr_weigh - other_weight

                if curr_diff < min_diff:
                    min_diff = curr_diff
                    min_c = c

    if min_c is not None:
        chosen_color = min_c
    elif allow_increase:
        chosen_color = lowest_unused_color(used_colors)
    else:
        chosen_color = graph.nodes[index][color]

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

#TODO iteracja zerowa wydaje sięzajmować jakoś dużo czasu
def genetic_coloring(graph: nx.Graph, pop_count: int, iterations: int, mprob: float, cprob: float, selected: int, verbal: int = 0, pool_count: int = 12, patience: int = 5, fix_prob: float = 0.1, mutate_ver_prob: float=0.01, random_init: bool = False) -> nx.Graph:
    population = []
    if verbal >= 1:
        print("Prep start")
    for i in range(pop_count):
        graph_g = graph.copy()
        if random_init:
            graph_g = random_correct_coloring(graph_g)
        else:
            graph_g = greedy_coloring(graph_g, create_random_permutation(range(len(graph_g.nodes))))
        population.append(graph_g)

    if verbal >= 1:
        print("Prep end")

    if verbal >= 2:
        print_many_graphs(population)

    curr_best = -math.inf
    curr_patience = 0
    # pool = mp.Pool(mp.cpu_count())
    pool = mp.Pool(pool_count)
    if verbal >= 1:
        print(f'Iteration')
    for it in range(iterations):
        if verbal >= 1:
            if it == 0:
                print(f'{it}', end="")
            else:
                print(f',{it}', end="")
        child_population = []

        # # Step 1: Init multiprocessing.Pool()
        # pool = mp.Pool(mp.cpu_count())
        #
        # # Step 2: `pool.apply` the `howmany_within_range()`
        # results = [pool.apply(fitness_func, args=(g,)) for g in population]
        #
        # # Step 3: Don't forget to close
        # pool.close()
        #
        # print(results)

        for ind in range(len(population)):
            fitness_func(population[ind]) # ustawia score grafu

        # to_mutate = []
        # to_cross = []
        # to_pass = []
        #
        # tasks = []
        #
        # while len(to_mutate) + len(to_cross) + len(to_pass) < pop_count:
        #     mt = random.uniform(0, 1)
        #     cr = random.uniform(0, 1)
        #
        #     mutant = tournament_selection(population, selected)
        #     if mt < mprob:
        #         to_mutate.append(mutant)
        #     else:
        #         to_pass.append(mutant)
        #
        #     if len(to_mutate) + len(to_cross) + len(to_pass) >= pop_count:
        #         break
        #
        #     if len(to_mutate) + len(to_cross) + len(to_pass) < pop_count - 2:
        #         cross1 = tournament_selection(population, selected)
        #         cross2 = tournament_selection(population, selected)
        #
        #         if cr < cprob:
        #             to_cross.append(['C', cross1, cross2])
        #         else:
        #             to_pass.append(cross1)
        #             to_pass.append(cross2)
        #
        # for m in to_mutate:
        #     tasks.append(['M', m, mutate_ver_prob])
        # for c in to_cross:
        #     tasks.append(c)

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
            mt = random.uniform(0, 1)
            if mt < mprob:
                mutate_tasks.append([mut_id, p, mutate_ver_prob])
            else:
                to_pass.append(p)

        print(f"<<{len(mutate_tasks)}>>", end="")

        processed = async_process_genetic_tasks(pool, pool_count, mutate_tasks)

        child_population = []
        child_population.extend(processed)
        child_population.extend(to_pass)
        assert len(child_population) == pop_count

        for cg in child_population:
            ok = check_correctness_of_coloring(cg)
            assert ok


        population = child_population

        local_best = -math.inf
        for g in population:
            if g.graph[score] > local_best:
                local_best = g.graph[score]

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
            print(f"({coloring_cost(min(population, key=lambda x: coloring_cost(x)))})", end="")
            if it > 0 and it % 10 == 0:
                print()

        if verbal >= 2:
            print_many_graphs(population)
            print()
    if verbal >= 1:
        print()

    pool.close()
    for ind in range(len(population)):
        fitness_func(population[ind]) # ustawia score grafu
    population = sorted(population, key=lambda x: x.graph[score], reverse=True)

    best = population[0]

    best = attempt_to_eliminate_small_colors(best, prob=fix_prob)

    return best.copy()

def attempt_to_eliminate_small_colors(graph: nx.Graph, prob: float=0.1):
    graph = graph.copy()
    color_count = get_color_counts(graph)
    colors_to_eliminate = []

    for col in color_count:
        if color_count[col] < len(graph.nodes) * 0.1:
            colors_to_eliminate.append(col)

    for v in graph.nodes:
        if graph.nodes[v][color] in colors_to_eliminate:
            if random.uniform(0, 1) < prob:
                # print(f'Attempt reduction {v}')
                graph = local_reduction(graph, v, allow_increase=False)

    return graph

# def genetic_task(graphs: [nx.Graph]) -> [nx.Graph]:
#     if len(graphs) == 1:
#         return [mutation(graphs[0])]
#     if len(graphs) == 2:
#         return [item for item in crossover(graphs[0], graphs[1])]

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

# def genetic_task(graph1: nx.Graph, graph2: nx.Graph) -> [nx.Graph]:
#     cross1, cross2 = crossover(graph1, graph2)
#     return [cross1, cross2]



