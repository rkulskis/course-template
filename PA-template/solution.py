from classlib import *

def doubleEdgeWeights(G):
    def scale(u,v):
        G[u][v] *= 2
    foreachEdge(G, scale)

USAGE = "Usage: python3 sumbission.py input_file output_file"
def interpretCommandLineArgs(args = []):
    assert len(args) == 2, USAGE
    input_file = args[0]
    output_file = args[1]
    G = readGraph(input_file)
    doubleEdgeWeights(G)
    writeGraph(G, output_file)
    return

if __name__ == "__main__":
    interpretCommandLineArgs(sys.argv[1:])    
