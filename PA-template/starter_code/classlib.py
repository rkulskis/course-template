############################################################
# Simple graphs: v3.4
# Adam Smith, Ross Mikulskis
############################################################
import re
import os, sys, subprocess, time
from gradescope_utils.autograder_utils.decorators import weight,visibility,number
import unittest
from collections import defaultdict
import random
import heapq, queue
import numpy as np
############################################################ Adjacency list
def emptyGraph(n=0):
    return {i: {} for i in range(n)}
############################################################ Nodes
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