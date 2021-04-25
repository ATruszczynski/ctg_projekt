from ga_coloring import *
import networkx as nx
import time
import datetime
import os
from graph_set_preparator import *
from algo_tuples import *

path = 'C:\\Users\\aleks\\Desktop\\inithx.i.3.col'
test_dir = "tests"

algorithms_extensive = []

verb = 1
pool_c = 12

algorithms_extensive.append(GA_Tuple(repetitions=3, pop_count=50, iterations=100, mprob=0.02, cprob=0.8, selected=3,
                                     patience=300, fix_prob=1, mutate_ver_prob=0.02,
                                     random_init=False, pool_count=pool_c, verbal=verb))

# algorithms.append(GA_Tuple(repetitions=2, pop_count=30, iterations=10, mprob=0.02, cprob=0.8, selected=10,
#                                                                                                          patience=100, fix_prob=0.1, mutate_ver_prob=0.01,
#                                                                                                          random_init=True, pool_count=pool_c, verbal=verb))
# algorithms.append(RVC_Tuple(repetitions=5))
# algorithms.append(Greed_Tuple(repetitions=6))
# algorithms.append(DSatur_Tuple(repetitions=7))
# algorithms.append(GA_Tuple(repetitions=3, pop_count=100, iterations=100, mprob=0.01, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.01, mutate_ver_prob=0.01,
#                            random_init=False, pool_count=pool_c, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=3, pop_count=100, iterations=100, mprob=0.01, cprob=0.5, selected=10,
#                            patience=100, fix_prob=0.01, mutate_ver_prob=0.01,
#                            random_init=False, pool_count=pool_c, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=3, pop_count=100, iterations=100, mprob=0.05, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.01, mutate_ver_prob=0.01,
#                            random_init=False, pool_count=pool_c, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=3, pop_count=100, iterations=100, mprob=0.01, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.01, mutate_ver_prob=0.05,
#                            random_init=False, pool_count=pool_c, verbal=verb))

def test(graph: nx.Graph, graph_name: str, algos: [AlgoTuple], stu: float) -> [str]:
    records = []
    for ind in range(len(algos)):
        algo = algos[ind]
        print(f"Test {ind+1}/{len(algos)}")
        record = f"{graph_name}"
        if type(algo) is GA_Tuple:
            record += ",GA"
        elif type(algo) is Pure_Greed_Tuple:
            record += ",PGreed"
        elif type(algo) is Greed_Tuple:
            record += ",Greed"
        elif type(algo) is DSatur_Tuple:
            record += ",DSatur"
        elif type(algo) is RVC_Tuple:
            record += ",RVC"

        repetitions = algo.repetitions

        min_cost = math.inf
        total_cost = 0
        total_time = 0

        for i in range(repetitions):
            start = end = 0
            coloring = None

            if type(algo) is GA_Tuple:
                start = time.time()
                coloring = genetic_coloring(graph=graph, pop_count=algo.pop_count, iterations=algo.iterations,
                                            mprob=algo.mprob, cprob=algo.cprob, selected=algo.selected, verbal=algo.verbal,
                                            pool_count=algo.pool_count, patience=algo.patience, fix_prob=algo.fix_prob,
                                            mutate_ver_prob=algo.mutate_ver_prob, random_init=algo.random_init)
                end = time.time()

            elif type(algo) is Pure_Greed_Tuple:
                start = time.time()
                coloring = greedy_coloring(graph=graph)
                end = time.time()

            elif type(algo) is RVC_Tuple:
                start = time.time()
                coloring = random_proper_coloring(graph=graph)
                end = time.time()

            elif type(algo) is Greed_Tuple:
                start = time.time()
                # TODO Podmienić na wywołanie algorytmu greedy
                coloring = greedy_coloring(graph=graph)
                end = time.time()

            elif type(algo) is DSatur_Tuple:
                start = time.time()
                # TODO Podmienić na wywołanie algorytmu DSatur
                coloring = greedy_coloring(graph=graph)
                end = time.time()

            assert check_if_coloring_is_proper(coloring)
            cost = get_coloring_cost(coloring)
            if cost < min_cost:
                min_cost = cost

            total_cost += cost
            total_time += end - start

        record += f",{round(total_cost/repetitions, 3)},{min_cost}"
        record += f",{round(total_time/repetitions/stu, 3)}"

        record += f",{algo.repetitions}"

        if type(algo) is GA_Tuple:
            record += f"|{algo.pop_count}|{algo.iterations}|{algo.mprob}|{algo.cprob}|" \
                      f"{algo.selected}|{algo.patience}|{algo.fix_prob}|{algo.mutate_ver_prob}|{algo.random_init}"
        if type(algo) is Greed_Tuple:
            # TODO Dodać parametry Greedy do record (jeśli jakieś są)
            pass
        if type(algo) is DSatur_Tuple:
            # TODO Dodać parametry DSatur do record (jeśli jakieś są)
            pass

        records.append(record)

    return records

# graphs to lista par (graf, nazwa_grafu)
def test_for_graphs(graphs: [(nx.Graph, str)], algos: [AlgoTuple], test_dir: str, stu_size: int = 500) -> [str]:
    stu = calculate_stu(stu_size)
    result = []

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)
    now = datetime.datetime.now()
    file = open(f"{test_dir}{os.path.sep}tests_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.csv", "w")

    file.write("name,algorithm,avg_cost,min_cost,avg_time,parameters\n")

    for ind in range(len(graphs)):
        g = graphs[ind]
        print(f"Tests for graph {ind + 1}")
        res = test(g[0], g[1], algos, stu)
        result.extend(res)
        for r in res:
            file.write(r + "\n")
        file.flush()

    return result


def calculate_stu(size: int = 500, rep: int = 10):
    vert = range(0, size)
    graph = nx.Graph()
    graph.add_nodes_from(vert)
    for i in range(len(vert)):
        for j in range(i+1, len(vert)):
            graph.add_edge(i, j)

    time_sum = 0

    for i in range(rep):
        graph_to_col = graph.copy()
        start = time.time()
        _ = greedy_coloring(graph_to_col)
        end = time.time()

        print(f'STU - {i+1}/{rep}')

        time_sum += end - start

    return time_sum/rep


if __name__ == '__main__':
    graphs = read_many_graph_files("instances", 1, 100, 0, 2, 5, 1, 7, 1, 11, 1, 26, 1, 34, 5, 46, 3)
    # graphs = read_many_graph_files("instances", 1, 100, 2, 100)
    test_for_graphs(graphs, algorithms_extensive, test_dir, 50)
