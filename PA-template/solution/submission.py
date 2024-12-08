############################################################
# Solution code for PA-template
# Created on December 08, 2024
############################################################

from classlib import *

def doubleEdgeWeights(G):
    def scale(u,v):
        G[u][v] *= 2
    foreachEdge(G, scale)
    return

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