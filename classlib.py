############################################################
# Simple graphs: v3.4
# Adam Smith, Ross Mikulskis
############################################################
import random
import numpy as np
############################################################ Adjacency list
def emptyGraph(n=0):
    return {i: {} for i in range(n)}
############################################################ Nodes
def N(G): return len(G)                    # N = nodes cardinality
def V(G): return {u for u in G}            # V = nodes set

def addNode(G, u):
    if u not in G:
        G[u] = {}
    
def delNode(G, u, delIncomingEdges=False):
    if u not in G:
        return
    if delIncomingEdges:
        foreachEdge(G, lambda a,b: delEdge(a,b) if b == u else None)
    del G[u]

def foreachNode(G, f_v, safe=False):
    s = lambda G: V(G) if safe else G
    for u in s(G): f_v(u)
############################################################ Edges
def E(G):                                  # E = edges set
    E = set()
    foreachEdge(G, lambda u,v: E.add((u,v)))
    return E

def addEdge(G, u, v, label=True, f=None, undir=False):
    if u not in G: addNode(G, u)
    if v not in G: addNode(G, v)
    
    if f and (v in G[u]):       # f is an optional lambda to decide how to merge
        G[u][v] = f(G[u][v], label)
    else:                       
        G[u][v] = label
    if undir:
        G[v][u] = G[u][v]

def delEdge(G, u, v, undirected=False):
    if not undirected and v in G[u]:
        del G[u][v]
    elif undirected and v in G[u] and u in G[v]:
        del G[u][v]
        del G[v][u]
        
def foreachEdge(G, f_e, safe=False):
    if safe:
        for (u,v) in E(G): f_e(u,v)
    else:
        traverseGraph(G, lambda u: None, f_e)
############################################################ Set-like operations
def copyGraph(G):               # deep copy
    C = emptyGraph()
    traverseGraph(G,
                  lambda u: addNode(C, u),
                  lambda u,v: addEdge(C, u, v, G[u][v]))
    return C
############################################################ Traversals
def traverseGraph(G, f_v, f_e, safe=False):
    if safe:
        traverseGraphSafe(G, f_v, f_e)
        return
    for u in G:
        f_v(u)
        for v in G[u]:
            f_e(u, v)

def traverseGraphSafe(G, f_v, f_e): # safe since iterate over list
    for (u,v) in E(G): f_e(u, v) # edges first since may delete nodes
    for u in V(G): f_v(u)
############################################################ Boolean operations
def graphForAll(G, f):
    for u in G:
        for v in G[u]:
            if not f(u, v):
                return False
    return True

def graphIsSubset(G1, G2, f=lambda w1, w2: w1 == w2): # G1 subset_eq G2
    def isSubset(u, v):
        return u in G2 and v in G2[u] and f(G1[u][v], G2[u][v])
    return graphForAll(G1, isSubset)

def graphsEqual(G1, G2, f=lambda w1, w2: w1 == w2):
    return graphIsSubset(G1, G2, f) and graphIsSubset(G2, G1, f)
############################################################ Graph generators
def sampleWoutReplace(n,d, exclude, rng):
    # generate d numbers without replacement from {0,...,n-1} - {exclude}.
    sample = [exclude]
    for j in range(d):
        nbr = rng.integers(n)
        while nbr in sample:
            nbr = rng.integers(n)
        sample.append(nbr)
    return sample[1:]

def randomDigraphDegreeBound(n, d, seed=None):
    # each vertex given d random incoming and outgoing edges (repeats ignored)
    rng = np.random.default_rng(seed)
    G = emptyGraph(n)
    for i in G:
        out_list = sampleWoutReplace(n,d,i, rng)
        for nbr in out_list:
            addEdge(G, i, nbr, label=rng.random())
        in_list = sampleWoutReplace(n,d,i, rng)
        for nbr in in_list:
            addEdge(G, nbr, i, label=rng.random())
    def scale_edge(u, v):
        G[u][v] = int(G[u][v] * 50)
    traverseGraph(G,
                  lambda u: None,
                  scale_edge)    
    return G
############################################################ I/O functions
def writeGraphF(G, f):          # sorted so identical graphs formatted same
    for u in sorted(G):
        f.write(f"{u}\n")
        for v in sorted(G[u]):
            if type(G[u][v]) == bool:
                f.write(f"{u}, {v}\n") # unweighted
            else:
                f.write(f"{u}, {v}, {G[u][v]}\n") # weighted
                
def readGraph(input_file):
    with open(input_file, 'r') as f:
        return readGraphF(f)

def readGraphF(f, intWeights=True):
    G = emptyGraph()
    cast = int if intWeights else float
    raw = [[cast(x) for x in s.split(',')] for s in f.read().splitlines()]
    for entry in raw:
        if len(entry) == 1:  # node name
            addNode(G, int(entry[0]))
        elif len(entry) == 2:  # unweighted
            addEdge(G, int(entry[0]), int(entry[1]))
        elif len(entry) == 3:  # weighted
            addEdge(G, int(entry[0]), int(entry[1]), label=cast(entry[2]))
        else:
            print("Incorrectly formatted entry ignored:", entry)
    return G
    
def writeGraph(G, output_file):
    with open(output_file, 'w') as f:
        writeGraphF(G, f)
    return
