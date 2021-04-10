from graph_utility import *
import networkx as nx
import random
import math
import multiprocessing as mp
import os

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
    # print('Mutation')
    # print(os.getpid())
    index = random.randrange(len(graph.nodes))
    return local_reduction(graph, index)

def local_reduction(graph: nx.Graph, index: int, allow_increase: bool = True):
    graph = graph.copy()
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

def genetic_coloring(graph: nx.Graph, pop_count: int, iterations: int, mprob: float, cprob: float, selected: int, verbal: bool = False, pool: int = 12, patience: int = 5, fix_prob: float = 0.1) -> nx.Graph:
    population = []

    print("Prep start")
    for i in range(pop_count):
        graph_g = graph.copy()
        graph_g = greedy_coloring(graph_g, create_random_permutation(range(len(graph_g.nodes))))
        # graph_g = random_correct_coloring(graph_g)
        population.append(graph_g)

    print("Prep end")

    if verbal:
        print_many_graphs(population)

    curr_best = -math.inf
    curr_patience = 0
    # pool = mp.Pool(mp.cpu_count())
    pool = mp.Pool(pool)
    for it in range(iterations):
        print(f'Iteration {it}')
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

        to_mutate = []
        to_cross = []
        to_pass = []

        tasks = []

        while len(to_mutate) + len(to_cross) + len(to_pass) < pop_count:
            mt = random.uniform(0, 1)
            cr = random.uniform(0, 1)

            mutant = tournament_selection(population, selected)
            if mt < mprob:
                to_mutate.append(mutant)
            else:
                to_pass.append(mutant)

            if len(to_mutate) + len(to_cross) + len(to_pass) >= pop_count:
                break

            if len(to_mutate) + len(to_cross) + len(to_pass) < pop_count - 2:
                cross1 = tournament_selection(population, 3)
                cross2 = tournament_selection(population, 3)

                if cr < cprob:
                    to_cross.append([cross1, cross2])
                else:
                    to_pass.append(cross1)
                    to_pass.append(cross2)

        for m in to_mutate:
            tasks.append([m])
        for c in to_cross:
            tasks.append(c)

        number_of_buckets = 10
        task_buckets = []
        for b in range(number_of_buckets):
            task_buckets.append([])

        for t_ind in range(len(tasks)):
            task_buckets[t_ind % number_of_buckets].append(tasks[t_ind])


        # processed = [pool.apply(genetic_task, args=(t,)) for t in tasks]
        processed = [pool.apply_async(genetic_tasks, args=(t,)) for t in task_buckets]
        [bucket.wait() for bucket in processed]

        processed = [result for bucket in processed for task in bucket.get(timeout=1) for result in task]

        child_population = []
        # child_population.extend(mutants)
        # child_population.extend(crossovers_t)
        child_population.extend(processed)
        child_population.extend(to_pass)

        ori = 1



        # while len(child_population) < pop_count:
        #     mt = random.uniform(0, 1)
        #     cr = random.uniform(0, 1)
        #
        #     mutant = tournament_selection(population, selected)
        #
        #     if mt < mprob:
        #         mutant = mutation(mutant)
        #
        #     child_population.append(mutant)
        #
        #     if len(child_population) >= pop_count:
        #         break
        #
        #     cross1 = tournament_selection(population, 3)
        #     cross2 = tournament_selection(population, 3)
        #
        #     if cr < cprob:
        #         cross1, cross2 = crossover(cross1, cross2)
        #
        #     child_population.append(cross1)
        #
        #     if len(child_population) >= pop_count:
        #         break
        #
        #     child_population.append(cross2)

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
            print("Patience run out!")
            break

        print(coloring_cost(min(population, key=lambda x: coloring_cost(x))))

        if verbal:
            print_many_graphs(population)
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

def genetic_tasks(graphs: [[nx.Graph]]) -> [[nx.Graph]]:
    results = []
    for g in graphs:
        if len(g) == 1:
            results.append([mutation(g[0])])
        if len(g) == 2:
            results.append([item for item in crossover(g[0], g[1])])
    return results

# def genetic_task(graph1: nx.Graph, graph2: nx.Graph) -> [nx.Graph]:
#     cross1, cross2 = crossover(graph1, graph2)
#     return [cross1, cross2]


