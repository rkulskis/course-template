############################################################
# Template starter code
# Ross Mikulskis, rkulskis@bu.edu
############################################################

import sys, os
from classlib import *

def doubleEdgeWeights(G):
    # TODO: implement
    return

############################################################ for course staff
sys.path.insert(0, os.path.abspath(".."))
if os.path.exists(os.path.join(os.path.abspath("../solution"), 'submission.py')):
    from solution.submission import *
############################################################

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
