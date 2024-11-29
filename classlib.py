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
############################################################ Gradescope
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
        {error_string[:min(1000, len(error_string))]}\n WITH {in_file}"
    
    passed, feedback = checkAnswer(in_file, out_file, student_out_file)
    info = f"Submitted program terminated in {runtime:.2f} seconds.\n{feedback}"
    return passed, info

class Test(unittest.TestCase):  # auto generate tests
    def formatted_test_run(self, i, visibility, checkAnswer):
        print(f"==========Testing input {i:02d} ({visibility})==========")
        passed, info = execTest(i, visibility, checkAnswer)
        self.fail(info) if not passed else print(info)

    def generate_test(self, i, visibility, w, checkAnswer):
        @weight(w)
        @number(i)
        def test_func(self):
            self.formatted_test_run(i, visibility, checkAnswer)
        return test_func
def generate_test_methods(default_weights, custom_weights, checkAnswer):
    for visibility in default_weights:
        dirpath = f"./tests/{visibility}/inputs"
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            i = int(re.findall(r'\d+', fname)[0])
            if i in custom_weights:
                w = custom_weights[i]
            else:
                w = default_weights[visibility]
            setattr(Test, f'test_{i}', Test().generate_test(i, visibility,
                                                            w, checkAnswer))
############################################################ Assignment creation
def create_starter_library(sol_file="solution.py", lib_file="classlib.py"):
    regex_fcall = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    regex_fdef = r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    regex_section_header = r'^#{60}'
    
    with open(lib_file, "r") as f:
        lib_code = f.read()
    with open(sol_file, "r") as sol_file:
        sol_code = sol_file.read()
    sol_fcalls = set(re.findall(regex_fcall, sol_code))
        
    # foreach function get name, start, scope=indentation
    lib_fdefs = {}              # name -> (start, indentation)
    for match in re.finditer(regex_fdef, lib_code, flags=re.MULTILINE):
        fname = match.group(1)
        start = match.start()
        line_start = lib_code.rfind('\n', 0, start)+1
        line = lib_code[line_start:lib_code.find('\n', line_start)]
        scope = len(line) - len(line.lstrip())
        lib_fdefs[fname] = (start, scope)
    G_dep = defaultdict(set)
    defs_inorder = list(sorted(lib_fdefs.items(), key=lambda i: i[1][0]))
    for i, (fname, (start, scope)) in enumerate(defs_inorder): # ordered by start
        j = i + 1
        end = len(lib_code)
        while j < len(defs_inorder):
            j_start = defs_inorder[j][1][0]
            j_scope = defs_inorder[j][1][1]
            if j_scope <= scope: # indentation, so less than means greater scope
                end = j_start
                break
            j += 1
        f_code = lib_code[start:end]
        fcalls = re.findall(regex_fcall, f_code)
        for f in fcalls:
            if f in lib_fdefs:
                G_dep[fname].add(f)
                
    def resolve_deps(funcs, graph): # resolve deps recursively
        resolved = set(funcs)
        stack = list(funcs)
        while stack:
            func = stack.pop()
            for dep in graph[func]:
                if dep not in resolved:
                    resolved.add(dep)
                    stack.append(dep)
        return resolved

    required_f = resolve_deps(sol_fcalls, G_dep)
    lines = lib_code.splitlines() # remove unused functions
    starter_lib = []
    skip = False
    for i, line in enumerate(lines):
        match = re.match(regex_fdef, line)
        if match:
            fname = match.group(1)
            skip = fname not in required_f
        elif re.match(regex_section_header, line):
            skip = False
        if not skip:
            starter_lib.append(line)
            
    i = 0; pops = 0
    while i + pops < len(starter_lib) - 1: # remove headers of empty sections
        if re.match(regex_section_header, starter_lib[i]) and \
           (re.match(regex_section_header, starter_lib[i+1])):
            starter_lib.pop(i)
            pops += 1
            i -= 1
        i += 1
    while re.match(regex_section_header, starter_lib[-1]):
        starter_lib.pop(-1)

    start_lib_path = f"./starter_code/{lib_file}"
    if os.path.exists(start_lib_path):
        os.remove(start_lib_path)    # get rid of temporary sym link to full lib
    with open(start_lib_path, "w") as f:
        f.write("\n".join(starter_lib)) # create starter_lib = subset of full lib
    print(f"Compiled starter_code/{lib_file} only containing functions needed.")
