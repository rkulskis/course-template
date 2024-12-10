############################################################
# Starter code for PA-template
# Created on December 10, 2024
############################################################

from classlib import *

def doubleEdgeWeights(G):
    return # TODO: implement

############################################################ for course staff
sys.path.insert(0, os.path.abspath('..'))
if os.path.exists('../solution/submission.py'):
    from solution.submission import *
############################################################

def interpretCommandLineArgs(args = []): 
    USAGE = "Usage: python3 sumbission.py input_file output_file"
    assert len(args) == 2, USAGE
    input_file = args[0]
    output_file = args[1]
    G = readGraph(input_file)
    doubleEdgeWeights(G)
    writeGraph(G, output_file)
    return

if __name__ == "__main__":
    interpretCommandLineArgs(sys.argv[1:])