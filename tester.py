from dsatur import get_dsatur_coloring
from ga_coloring import *
import networkx as nx
import time
import datetime
import os
from graph_set_preparator import *
from algo_tuples import *
from greedy_coloring import get_greedy_coloring
from dsatur import *

# path = 'C:\\Users\\aleks\\Desktop\\inithx.i.3.col'
path = '/Users/tomek/Workspace/ctg_projekt/instances/inithx.i.3.col'
test_dir = "tests"


verb = 0
pool_c = 12

def test(graphh: nx.Graph, graph_name: str, graph_num: int, algos: [AlgoTuple], stu: float) -> [str]:
    records = []
    for ind in range(len(algos)):
        algo = algos[ind]
        print(f"Graph - {graph_num + 1} - test {ind + 1}/{len(algos)}")
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
        min_cols = math.inf
        total_cost = 0
        total_colors = 0
        total_time = 0

        for i in range(repetitions):
            graph = copy.deepcopy(graphh)
            start = end = 0
            coloring = None

            if type(algo) is GA_Tuple:
                start = time.time()
                coloring = genetic_coloring(graph=graph, pop_count=algo.pop_count, iterations=algo.iterations,
                                            mprob=algo.mprob, cprob=algo.cprob, selected=algo.selected,
                                            verbal=algo.verbal,
                                            pool_count=algo.pool_count, patience=algo.patience, fix_prob=algo.fix_prob,
                                            random_init=algo.random_init)
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
                coloring = get_greedy_coloring(graph=graph)
                end = time.time()

            elif type(algo) is DSatur_Tuple:
                start = time.time()
                # TODO Podmienić na wywołanie algorytmu DSatur
                coloring = get_dsatur_coloring(graph=graph)
                end = time.time()

            assert check_if_coloring_is_proper(coloring)
            cost = get_coloring_cost(coloring)
            if cost < min_cost:
                min_cost = cost

            col = count_of_colors_used(coloring)
            if col < min_cols:
                min_cols = col

            total_cost += cost
            total_colors += count_of_colors_used(coloring)
            total_time += end - start

        record += f",{round(total_colors / repetitions, 3)},{min_cols},{round(total_cost / repetitions, 3)},{min_cost}"
        record += f",{round(total_time / repetitions / stu, 3)}"

        record += f",{algo.repetitions}"

        if type(algo) is GA_Tuple:
            record += f"|{algo.pop_count}|{algo.iterations}|{algo.mprob}|{algo.cprob}|" \
                      f"{algo.selected}|{algo.patience}|{algo.fix_prob}|{algo.random_init}"
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
    file = open(
        f"{test_dir}{os.path.sep}tests_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.csv", "w")

    file.write("name,algorithm,avg_col_used,min_col_used,avg_cost,min_cost,avg_time,parameters\n")

    for ind in range(len(graphs)):
        g = graphs[ind]
        print(f"Tests for graph {ind + 1}")
        res = test(g[0], g[1], ind,  algos, stu)
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
        for j in range(i + 1, len(vert)):
            graph.add_edge(i, j)

    time_sum = 0

    for i in range(rep):
        graph_to_col = graph.copy()
        start = time.time()
        _ = greedy_coloring(graph_to_col)
        end = time.time()

        print(f'STU - {i + 1}/{rep}')

        time_sum += end - start

    return time_sum / rep


if __name__ == '__main__':
    algorithms_to_test = [] # lista algo_tuples, które określają algorytm do użycia, jego parametry i liczbę powtórzeń testów

    # przykładowa krotka dla ga (który jest powodem dla którego w ogóle są potrzebne krotki)
    algorithms_to_test.append(GA_Tuple(repetitions=7, pop_count=100, iterations=100, mprob=0.0005, cprob=0.8,
                                         selected=3, patience=300, fix_prob=1,
                                         random_init=False, pool_count=pool_c, verbal=verb))
    algorithms_to_test.append(DSatur_Tuple(10)) # dodanie innych algo
    algorithms_to_test.append(Greed_Tuple(10))
    algorithms_to_test.append(Pure_Greed_Tuple(10))

    # wczytywanie grafów. pierwszy arg to folder z grafami, dwa następne regulują losowe wagi, następny ciag to pary
    # (indeks_pliku, liczba_plików_do_wczytania_od_tego_indeksu) czyli np. '0, 2, 4, 4' to wczytanie dwóch plików 0, 1, 4, 5, 6, 7
    # można wpisać za dużą liczbe plików do wczytania i się nie obrazi
    graphs = read_many_graph_files("instances", # mypath
                                   1,   #minWeight
                                   100, #maxWeight
                                   0, 100)
    # graphs = read_many_graph_files("instances", 1, 100, 2, 100)

    test_for_graphs(graphs=graphs, algos=algorithms_to_test, test_dir=test_dir, stu_size=500)
