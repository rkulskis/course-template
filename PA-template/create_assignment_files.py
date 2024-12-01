import random
from solution.submission import *
from classlib import *
from collections import defaultdict
from functools import partial

def create_test_files():
    visibility_map = {'v': "visible", 'p': "after_published",
                      'd': "after_due_date", 'h': "hidden"} # for gradescope UI

    def testPath(i, visibility, fType):
        assert fType in ['input', 'output']
        return f"./autograder/tests/{visibility}/{fType}s/{fType}{i:02d}.txt"

    def testGeneric(i, sol, n, vmap_key='v'):
        visibility = visibility_map[vmap_key]
        G = randomDigraphDegreeBound(n, 3, seed=123+i)        
        writeGraph(G, testPath(i, visibility, 'input'))
        doubleEdgeWeights(G)        
        writeGraph(G, testPath(i, visibility, 'output'))        
        
    test_descriptions = []
    def testDoubleEdgeWeights(i, param):
        (n, vmap_key) = param
        testGeneric(i, doubleEdgeWeights, n, vmap_key=vmap_key)
        test_descriptions.append(f"Double Edge Weights: n={n}")

    tests = {}
    tests[testDoubleEdgeWeights] = [
        (5, 'v'), (10, 'v'), (100, 'v'),
        (50, 'v'), (80, 'v'), (200, 'p')
    ]

    i = 0
    for test in tests:
        params = tests[test]
        for param in params:
            test(i, param)
            i += 1
        
    with open ("./autograder/tests/test_descriptions.txt", 'w') as f:
        for desc in test_descriptions:
            f.write(f"{desc}\n")
            
if __name__ == "__main__":
    create_test_files()
    create_starter_library()    
