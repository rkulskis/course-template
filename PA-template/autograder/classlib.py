############################################################
# Simple graphs: v3.4
# Adam Smith, Ross Mikulskis
############################################################
import random
import sys, os
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

def foreachEdge(G, f_e, safe=False):
    if safe:
        for (u,v) in E(G): f_e(u,v)
    else:
        traverseGraph(G, lambda u: None, f_e)

############################################################ Boolean operations
def graphForAll(G, f_v=None, f_e=None):
    for u in G:
        if f_v and not f_v(u):
            return False
        if not f_e: continue
        for v in G[u]:
            if not f_e(u, v):
                return False
    return True

def graphIsSubset(G1, G2, f=lambda w1, w2: w1 == w2): # G1 subset_eq G2
    def isSubset(u, v):
        return u in G2 and v in G2[u] and f(G1[u][v], G2[u][v])
    return graphForAll(G1, f_e=isSubset)

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

def writeGraph(G, output_file, args=()):
    with open(output_file, 'w') as f:
        args_header = 'args: '
        for i, arg in enumerate(args):
            if i == len(args) - 1:
                args_header += f'{arg}\n'
            else:
                args_header += f'{arg}, '
        if len(args) > 0:
            f.write(args_header)
        writeGraphF(G, f)

def readGraph(input_file):
    args = None
    with open(input_file, 'r') as f:
        firstline = f.readline().strip()
        if "args:" in firstline:
            args = (map(int, firstline[len("args:"):].split(',')))
        else:
            f.seek(0)
        G = readGraphF(f)
    if args is None:
        return G
    else:
        return G, args

############################################################ Gradescope
import re
from collections import defaultdict
import timeout_decorator
import subprocess, time
from gradescope_utils.autograder_utils.decorators import *
import unittest
from problems.problem import Problem
def execTest(i, visibility, checkAnswer):
    in_file = f"tests/{visibility}/inputs/input{i:02d}.txt"
    out_file = f"tests/{visibility}/outputs/output{i:02d}.txt"
    student_out_file = f"student_output.txt"
    
    time_bound = 60
    start = time.time()
    try:
        ret = subprocess.run(['python3', f"submission.py",
                                    in_file, student_out_file],
                                   stderr=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   timeout=time_bound)
    except subprocess.TimeoutExpired:
        return False, f'Program ran longer than {time_bound:.2f} seconds.'
    end = time.time()
    runtime = end - start
    
    if ret.returncode != 0:
        error_string = ret.stderr.decode() if ret.stderr else ""
        return False, f"Submitted program crashed.\n === Error messages:\n \
        {error_string[:min(1000, len(error_string))]}"
    
    passed, feedback = checkAnswer(i, in_file, out_file, student_out_file)
    info = f"Submitted program terminated in {runtime:.2f} seconds.\n{feedback}"
    return passed, info

class Test(unittest.TestCase):  # auto generate tests
    def formatted_test_run(self, i, visibility, checkAnswer, test_description):
        print(f"==========Testing input {i:02d} ({visibility})==========")
        print(test_description)
        passed, info = execTest(i, visibility, checkAnswer)
        self.fail(info) if not passed else print(info)

    def generate_test(self, i, visibility, w, checkAnswer, test_description):
        @weight(w)
        @number(f'({i:02d}: {test_description}') # display metadata on test
        @timeout_decorator.timeout(5)
        def test_func(self):
            self.formatted_test_run(i, visibility, checkAnswer, test_description)
        return test_func

def generate_test_methods(checkAnswer):
    with open("tests/test_descriptions.txt", 'r') as f:
        test_descriptions = [line for line in f]
    
    for visibility in set(Problem.VISIBILITY_MAP.values()) | {""}:
        dirpath = f"./tests/{visibility}/inputs"
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            i = int(re.findall(r'\d+', fname)[0])
            w = 1
            test_description = test_descriptions[i] if test_descriptions else "" 
            setattr(Test, f"test_{i}",
                    Test().generate_test(i, visibility, w,
                                         checkAnswer, test_description))

