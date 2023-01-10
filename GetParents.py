import numpy as np


def get_parents(graph):
    nodes = graph.shape[0]
    parents_list = []
    for i in range(nodes):
        parents_list.append([])
    for i in range(nodes):
        for j in range(nodes):
            if graph[i][j]==-1:
                parents_list[j].append(i)
    return parents_list


if __name__ == '__main__':
    graph = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, -1, 0, ],
         [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, ],
         [0, 0, 0, 0, -1, 0, 0, -1, -1, -1, 0, 0, -1, 0, ],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, ],
         [0, 0, 1, 0, 0, 0, -1, -1, 1, 0, 1, 0, 1, 0, ],
         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, ],
         [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, ],
         [0, -1, 1, 0, 1, 0, -1, 0, 0, 0, 1, 0, 1, 1, ],
         [-1, 0, -1, 0, -1, 0, 0, 0, 0, -1, -1, -1, 0, 0, ],
         [0, -1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, ],
         [0, -1, 0, 0, -1, 0, 0, -1, 1, 0, 0, 0, 0, 1, ],
         [0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 1, ],
         [1, 0, 1, 0, -1, -1, -1, -1, 0, 0, 0, 0, 0, -1, ],
         [0, -1, 0, 1, 0, -1, 0, -1, 0, -1, -1, -1, 1, 0, ], ])
    parent_list=get_parents(graph)
    for i in range(len(parent_list)):
        for j in range(len (parent_list[i])):
            parent_list[i][j]+=1
    print(parent_list)
