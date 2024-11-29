import random
from solution import *
from classlib import *
from collections import defaultdict

def create_test_files():
    visibility_map = {'v': "visible", 'p': "after_published",
                      'd': "after_due_date", 'h': "hidden"} # for gradescope UI
    params = [(5, 'v'), (5, 'v'), (200, 'v'),
              (5, 'p'), (5, 'p'), (200, 'p'),
              (5, 'd'), (5, 'd'), (200, 'd'),
              (123, 'h'), (123, 'h'), (123, 'h')]
    for i, param in enumerate(params):
        n, v = param
        visibility = visibility_map[v]
        G = randomDigraphDegreeBound(n, 3, seed=123+i)
        prefix = f"./autograder/tests/{visibility}"
        writeGraph(G, f"{prefix}/inputs/input{i:02d}.txt")
        doubleEdgeWeights(G)
        writeGraph(G, f"{prefix}/outputs/output{i:02d}.txt")

if __name__ == "__main__":
    create_test_files()
    create_starter_library()    
