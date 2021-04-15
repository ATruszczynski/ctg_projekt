from ga_coloring import *
import networkx as nx
import time
import datetime
import os
from graph_set_preparator import *
from algo_tuples import *

path = 'C:\\Users\\aleks\\Desktop\\inithx.i.3.col'
test_dir = "tests"

algorithms = []

verb = 1
pool_c = 10

# algorithms.append(GA_Tuple(repetitions=1, pop_count=2, iterations=2, mprob=0.2, cprob=0.2, selected=10,
#                            patience=100, fix_prob=0.01, mutate_ver_prob=0.001,
#                            random_init=True, pool_count=12, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=5, pop_count=10, iterations=100, mprob=0.01, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.1, mutate_ver_prob=0.25,
#                            random_init=True, pool_count=pool_c, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=5, pop_count=100, iterations=100, mprob=0.01, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.1, mutate_ver_prob=0.1,
#                            random_init=True, pool_count=pool_c, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=5, pop_count=100, iterations=100, mprob=0.8, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.1, mutate_ver_prob=0.01,
#                            random_init=True, pool_count=pool_c, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=5, pop_count=200, iterations=100000, mprob=0.01, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.1, mutate_ver_prob=0.01,
#                            random_init=True, pool_count=pool_c, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=5, pop_count=200, iterations=100000, mprob=0.02, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.1, mutate_ver_prob=0.01,
#                            random_init=True, pool_count=pool_c, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=5, pop_count=200, iterations=100000, mprob=0.01, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.1, mutate_ver_prob=0.02,
#                            random_init=True, pool_count=pool_c, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=5, pop_count=100, iterations=100, mprob=0.0, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.01, mutate_ver_prob=0.01,
#                            random_init=False, pool_count=pool_c, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=5, pop_count=100, iterations=100, mprob=0.01, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.1, mutate_ver_prob=0.01,
#                            random_init=True, pool_count=pool_c, verbal=verb))
# algorithms.append(GA_Tuple(repetitions=5, pop_count=100, iterations=100, mprob=0.0, cprob=0.8, selected=10,
#                            patience=100, fix_prob=0.01, mutate_ver_prob=0.01,
#                            random_init=True, pool_count=pool_c, verbal=verb))
algorithms.append(GA_Tuple(repetitions=5, pop_count=10, iterations=10, mprob=1, cprob=1, selected=10,
                           patience=100, fix_prob=0.0, mutate_ver_prob=0.01,
                           random_init=True, pool_count=pool_c, verbal=verb))


def test(graph: nx.Graph, graph_name: str, algos: [AlgoTuple], stu: float) -> [str]:
    records = []
    for ind in range(len(algos)):
        algo = algos[ind]
        print(f"Test {ind+1}/{len(algos)}")
        record = f"{graph_name}"
        if type(algo) is GA_Tuple:
            record += ",GA"

            iterations = algo.repetitions

            min_cost = math.inf
            total_cost = 0
            total_time = 0

            for i in range(iterations):
                start = time.time()
                coloring = genetic_coloring(graph=graph, pop_count=algo.pop_count, iterations=algo.iterations,
                                            mprob=algo.mprob, cprob=algo.cprob, selected=algo.selected, verbal=algo.verbal,
                                            pool_count=algo.pool_count, patience=algo.patience, fix_prob=algo.fix_prob,
                                            mutate_ver_prob=algo.mutate_ver_prob, random_init=algo.random_init)
                end = time.time()

                cost = coloring_cost(coloring)
                if cost < min_cost:
                    min_cost = cost

                total_cost += cost
                total_time += end - start

            record += f",{round(total_cost/iterations, 3)},{min_cost}"
            record += f",{round(total_time/iterations/stu, 3)}"
            record += f",{algo.repetitions}|{algo.pop_count}|{algo.iterations}|{algo.mprob}|{algo.cprob}|" \
                      f"{algo.selected}|{algo.patience}|{algo.fix_prob}|{algo.mutate_ver_prob}|{algo.random_init}"

        records.append(record)

    return records

def test_for_graphs(graphs: [nx.Graph], algos: [AlgoTuple], test_dir: str, stu_size: int = 500) -> [str]:
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
    # print(test(graph, "desu", test_tuples, ""))
    graphs = read_many_graph_files("instances", 0, 3)
    test_for_graphs(graphs, algorithms, test_dir, 500)
