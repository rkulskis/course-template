from classlib import *
from problems.DoubleEdgeWeights import DoubleEdgeWeights

problems = [
    DoubleEdgeWeights([
        (5, 'v'), (10, 'v'), (100, 'v'),
        (50, 'v'), (80, 'v'), (200, 'p')
    ])
]

# for generating {starter_code,solution}/submission.py
def interpretCommandLineArgs(args = []): 
    USAGE = "Usage: python3 sumbission.py input_file output_file"
    assert len(args) == 2, USAGE
    input_file = args[0]
    output_file = args[1]
    G = readGraph(input_file)
    doubleEdgeWeights(G)
    writeGraph(G, output_file)
    return

create_all_files(problems, interpretCommandLineArgs)
