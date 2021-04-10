from ga_coloring import *
import networkx as nx
import time
import datetime
import os

path = 'C:\\Users\\aleks\\Desktop\\inithx.i.3.col'
test_dir = "tests"
graph = read_graph_file(path)
# graph.remove_nodes_from(list(range(200, 621)))
graph = weight_graph_randomly(graph, 10, 100).copy()

test_tuples = []

test_tuples.append(('GA', 5, 100, 100, 0.8, 0.8, 5, 10, 1))

def test(graph: nx.Graph, graph_name: str, tuples, path_to_dir: str):
    stu = calculate_stu(500)
    records = []
    for test in tuples:
        record = f"{graph_name}"
        if test[0] == 'GA':
            record += ",GA"

            fix_prob = 0
            if len(test) == 9:
                fix_prob = test[8]
            else:
                fix_prob = 0.1

            iterations = test[1]

            total_cost = 0
            total_time = 0

            for i in range(iterations):
                start = time.time()
                coloring = genetic_coloring(graph=graph, pop_count=test[2], iterations=test[3], mprob=test[4], cprob=test[5], selected=test[6], verbal=False, patience=test[7], fix_prob=fix_prob)
                end = time.time()

                total_cost += coloring_cost(coloring)
                total_time += end - start

            record += f",{round(total_cost/iterations, 3)}"
            record += f",{round(total_time/iterations/stu, 3)}"

        records.append(record)

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    now = datetime.datetime.now()
    file = open(f"{test_dir}{os.path.sep}tests_{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}.csv", "w")

    for r in records:
        file.write(r + "\n")

    return records


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
    print(test(graph, "desu", test_tuples, ""))
