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
    
def xorNode(G, u):
    addNode(G,u) if u not in G else delNode(G,u)
    
def foreachNode(G, f_v, safe=False):
    s = lambda G: V(G) if safe else G
    for u in s(G): f_v(u)

def degree(G, u):
    return len(G[u])
############################################################ Edges
def M(G): return sum(len(G[u]) for u in G) # M = edges cardinality
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
        
def xorEdge(G, u, v, label=True, f=None, undir=False):
    if u in G and v in G[u]:
        delEdge(G, u, v)
    else:
        addEdge(G, u, v, label, f, undir)

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
            args = list(map(int, firstline[len("args:"):].split(',')))
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
import subprocess, time
import timeout_decorator
from gradescope_utils.autograder_utils.decorators import *
import unittest
from problems.problem import Problem

def execTest(i, visibility, checkAnswer):
    cwd = os.getcwd()
    in_file = f"{cwd}/tests/{visibility}/inputs/input{i:02d}.txt"
    out_file = f"{cwd}/tests/{visibility}/outputs/output{i:02d}.txt"
    student_out_file = f"student_output.txt"
    with open(student_out_file, "w") as f:
        f.write("w")
    time_bound = 60
    start = time.time()
    try:
        firejail = [
            'firejail',
            '--nonewprivs',
            '--nogroups',
            '--seccomp',
            '--net=none',
            f'--whitelist={in_file}',
            f'--whitelist={cwd}/submission.py',
            f'--whitelist={cwd}/problems',
            f'--whitelist={cwd}/classlib.py',
            f'--whitelist={cwd}/{student_out_file}',
        ]
        ret = subprocess.run(
            firejail + ['python3', '-c',
                        f"import sys; sys.path.append('{cwd}'); \
                        from submission import *; \
                        interpretCommandLineArgs(sys.argv[1:])",
                        in_file, student_out_file],
            stderr=subprocess.PIPE, stdout=subprocess.PIPE, timeout=time_bound
        )
    except subprocess.TimeoutExpired:
        return False, f'Program ran longer than {time_bound:.2f} seconds.'
    end = time.time()
    runtime = end - start
    
    if ret.returncode != 0:
        error_string = ret.stderr.decode() if ret.stderr else ""
        return False, f"Submitted program crashed.\n === Error messages:\n \
        {error_string[:min(1000, len(error_string))]}"
    
    passed, feedback = checkAnswer(in_file, out_file, student_out_file)
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

def generate_test_methods(problemClasses):
    with open("tests/test_descriptions.txt", 'r') as f:
        test_descriptions = [line for line in f]

    pname_to_pclass = {cls.__name__: cls for cls in problemClasses}
    test_classes = [s.split(":")[0] for s in test_descriptions] # i.e. "ClassName"
    
    for visibility in set(Problem.VISIBILITY_MAP.values()) | {""}:
        dirpath = f"./tests/{visibility}/inputs"
        if not os.path.isdir(dirpath):
            continue
        for fname in sorted(os.listdir(dirpath)):
            i = int(re.findall(r'\d+', fname)[0])
            w = 1
            test_description = test_descriptions[i]
            pclass = pname_to_pclass[test_classes[i]]
            setattr(Test, f"test_{i}",
                    Test().generate_test(i, visibility, w,
                                         pclass.checkAnswer, test_description))
############################################################ Assignment creation
import tree_sitter_python as ts
import ast
import inspect, textwrap
from datetime import datetime
from tree_sitter import Language, Parser

def get_problem_paths(problem_instances):
    pclass_names = [instance.__class__.__name__ for instance in problem_instances]
    return [f"problems/{cls_name}.py" for cls_name in pclass_names]

def create_test_files(problems):
    test_descriptions = []; index = 0; classname_i = 0
    pclass_names = [instance.__class__.__name__ for instance in problems]
    for p in problems:
        for param in p.getTestParams():
            test_descriptions.append(
                f"{pclass_names[classname_i]}: {p.createTest(index, param)}"
            )
            index += 1
        classname_i += 1
    with open("autograder/tests/test_descriptions.txt", 'w') as f:
        for p in test_descriptions:
            f.write(f"{p}\n")

def extract_calls(u, code):
    calls = set()
    def rec_helper(u, code):
        nonlocal calls
        for v in u.children:
            if u.type == "call" and v.type == "identifier":
                name = code[v.start_byte:v.end_byte]
                calls.add(name)
            rec_helper(v, code)
    rec_helper(u, code)
    return calls

def traverse_ast(root, f_uv):
    def rec_helper(u):
        for v in u.children:
            if f_uv(u, v): # only explore subtree if return True
                rec_helper(v)   
    rec_helper(root)

section_demarcation = "#" * 60
def create_sublib(lib_path="classlib.py",
                           sublib_path="starter_code/classlib.py",
                           dep_file_paths=["solution/submission.py"]):
    LANGUAGE = Language(ts.language())
    parser = Parser(LANGUAGE)
    if os.path.exists(sublib_path):
        os.remove(sublib_path) # remove temporary soft link
    with open(lib_path, "r") as f:
        lib_code = f.read()
        lib_tree = parser.parse(bytes(lib_code, "utf8"))
    
    calls = set()
    for path in dep_file_paths:
        with open(path, "r") as f:
            code = f.read()
            tree = parser.parse(bytes(code, "utf8"))
        calls = calls.union(extract_calls(tree.root_node, code))
    def get_defs(tree, code):
        defs = {}
        comments_and_imports = {}
        i = 0
        for u in tree.root_node.children:
            if "definition" in u.type:
                id_v = next((v for v in u.children if v.type=="identifier"), None)
                if not id_v: continue
                name = code[id_v.start_byte:id_v.end_byte]
                defs[name] = u
            elif "import" in u.type or u.type == "comment":
                comments_and_imports[f"{u.type}_{i}"] = u
                i += 1
        return defs, comments_and_imports

    defs, comments_and_imports = get_defs(lib_tree, lib_code)
    processed = {}
    while calls:
        name = calls.pop()
        if name not in defs or name in processed:
            continue
        u = defs[name]
        processed[name] = u
        nested_calls = extract_calls(u, lib_code)
        calls.update(nested_calls)

    sorted_deps = sorted(processed.items() | comments_and_imports.items(),
                         key=lambda i: i[1].start_byte)
    constructs = {}
    # always include first imports before any defs
    # for future imports, only include if fdef in section included in sublib
    after_first_def = False
    not_def = lambda u: "definition" not in u.type
    section = set()
    added_def = False
    for i, (name, u) in enumerate(sorted_deps):
        block = lib_code[u.start_byte:u.end_byte]
        if (section_demarcation in block or i == len(sorted_deps) - 1) and \
           not added_def:
            for j in section:
                del constructs[j]
        if i == len(sorted_deps) - 1:
            break
        if section_demarcation in block:
            section.clear()
            added_def = False
        if not_def(u):
            constructs[name] = f"{block}\n"
            if after_first_def:
                section.add(name)
        else:
            after_first_def = True
            added_def = True
            constructs[name] = f"{block}\n\n"
    with open(sublib_path, "w") as f:
        for construct in constructs.values():
            f.write(construct)

def unindent_once(code, indent_size=4):
    lines = code.splitlines()
    adjusted_lines = []
    for line in lines:
        if line.startswith(" " * indent_size):
            adjusted_lines.append(line[indent_size:])
        elif line.strip():
            adjusted_lines.append(line)
    return "\n".join(adjusted_lines)
    
def create_submissions(problem_paths, interpretCommandLineArgs):
    LANGUAGE = Language(ts.language())
    parser = Parser(LANGUAGE)
    sol_import_stub = f"\n{section_demarcation} for course staff\nsys.path.insert(0, os.path.abspath('..'))\nif os.path.exists('../solution/submission.py'):\n    from solution.submission import *\n{section_demarcation}\n\n"
    constructs = {}
    imports = set()
    for problem_path in problem_paths:
        with open(problem_path, "r") as file:
            code = file.read()
        tree = parser.parse(bytes(code, "utf8"))
        imports.update(
            f"{code[u.start_byte:u.end_byte]}\n"
            for u in tree.root_node.children
            if "import" in u.type
        )
        def f_uv(u, v):
            if u.type != "function_definition" or v.type != "identifier":
                return True
            fname = code[v.start_byte:v.end_byte]
            if fname != "solution" and fname != "starter":
                return True
            w = next((w for w in u.children if w.type == "block"), None)
            new_fname = problem_path.split('/')[-1].split('.')[0]
            if not new_fname[1].isupper():
                new_fname = new_fname[0].lower() + new_fname[1:]
            if  fname == "starter":
                new_fname = f"{fname}_{new_fname}"
            params = code[v.end_byte:w.start_byte]
            new_params = "(" + ", " \
                .join(part.strip() for part in params.strip("()").split(",") \
                      if part.strip() != "self")
            body = code[w.start_byte:u.end_byte]
            new_body = unindent_once(body) + "\n"
            fhead = f"def {new_fname}{new_params}"
            constructs[new_fname] = (
                f"{fhead}\n    {new_body}\n",
                len(fhead)
            )
            return True
        traverse_ast(tree.root_node, f_uv)

    for dir in ["solution", "starter_code"]:
        with open(f"{dir}/submission.py", "w") as f:
            pset = os.path.basename(os.getcwd())
            timestamp = datetime.now().strftime("Created on %B %d, %Y")
            section_header = f"{section_demarcation}\n# {'Starter' if dir == 'starter_code' else 'Solution'} code for {pset}\n# {timestamp}\n{section_demarcation}\n\n"
            f.write(section_header)
            # for i in imports:
                # if ".problem" in i: continue
                # f.write(i)
            f.write("from classlib import *\n\n")
            for fname, construct in constructs.items():
                if f"starter_{fname}" in constructs and dir == "starter_code":
                    continue
                if fname.startswith("starter_"):
                    if dir == "solution":
                        continue
                    fdef = construct[0]
                    fdef = re.sub(r"starter_", "", fdef)
                else:
                    fdef = construct[0] if dir == "solution" else \
                        construct[0][:construct[1]] \
                        + "\n    return # TODO: implement\n"
                fdef = re.sub(r"\.solution\(self,", "(", fdef)
                f.write(fdef)
            if dir == "starter_code":
                f.write(sol_import_stub)
            f.write(inspect.getsource(interpretCommandLineArgs))
            f.write('\nif __name__ == "__main__":\n    interpretCommandLineArgs(sys.argv[1:])')

def create_problems_subset(problems):
    os.makedirs("autograder/tests/problems", exist_ok=True)
    problem_paths = get_problem_paths(problems) + ["problems/problem.py"]
    LANGUAGE = Language(ts.language())
    parser = Parser(LANGUAGE)
    for p in problem_paths:
        with open(p, "r") as file:
            code = file.read()
            tree = parser.parse(bytes(code, "utf8"))
        exclude_bounds = []
        def f_uv(u, v):
            nonlocal exclude_bounds
            if u.type == "function_definition" and v.type == "identifier":
                fname = code[v.start_byte:v.end_byte]
                if fname == "solution":
                    exclude_bounds.append((u.start_byte, u.end_byte))
                    return False
                elif fname == "createTest":
                    exclude_bounds.append((u.start_byte, u.end_byte))             
                    return False                
            if len(exclude_bounds) == 2:
                return False
            return True
        traverse_ast(tree.root_node, f_uv)
        filtered_code = ""; curr = 0
        for (start,end) in sorted(exclude_bounds, key=lambda t: t[0]):
            filtered_code += code[curr:start]
            curr = end
        filtered_code += code[curr:]
        problem_name = os.path.basename(p)
        output_path = os.path.join("autograder/tests/problems", problem_name)
        with open(output_path, "w") as f_out:
            f_out.write(filtered_code)

def create_test_module(problems):
    with open("autograder/tests/test.py", "w") as f:
        f.write("from classlib import *\nimport unittest\n")
        pclass_names = [instance.__class__.__name__ for instance in problems]
        for p in pclass_names:
            f.write(f"from problems.{p} import *\n")
        f.write(f"\ngenerate_test_methods([{', '.join(pclass_names)}])\n\n")
        f.write("if __name__ == '__main__':\n    unittest.main()")
        
def create_all_files(problems, interpretCommandLineArgs):
    # problem files for this PA with solution() and createTest() removed, for
    # autograder and starter code tests, which are obsfucated with pyarmor
    create_problems_subset(problems)
    problem_paths = [p for p in get_problem_paths(problems)]
    with open("autograder/tests/problem_paths.txt", "w") as f:
        for p in problem_paths:
            f.write(f"{p}\n")
        f.write(f"problems/problem.py\n") # for use in Makefile script; pyarmor
    create_test_files(problems)           # {in,out}puts based on problem params
    # create autograder/tests/test.py based on problems array. Each one has
    # a checkAnswer and we can match test to class using test_descriptions.txt
    # generated by create_test_files()
    create_test_module(problems)
    # {solution,starter_code}/submission.py
    create_submissions(problem_paths, interpretCommandLineArgs)
    # only include necessary functions for each autogen sublib
    create_sublib(     
        sublib_path="starter_code/classlib.py",
        dep_file_paths=problem_paths + ["solution/submission.py"]
    )
    create_sublib(
        sublib_path="autograder/tests/classlib.py",
        dep_file_paths=problem_paths + \
        ["solution/submission.py", "autograder/tests/test.py"]
    )
############################################################ PA3
def subtractGraph(G1, G2, subtractNodes=False):      # G1 - G2
    def rmG_n(u):
        delNode(G1, u)     
    traverseGraph(G2,
                  lambda u: rmG_n if subtractNodes else None,
                  lambda u,v: delEdge(G1, u, v),
                  safe=True)
    return G1

def reverseGraph(G):
    def reverseEdge(u,v):
        addEdge(G, v, u, G[u][v])
        delEdge(G, u, v)
    foreachEdge(G, reverseEdge, safe=True)
    return G

def BFS(G, s, exploreZeroEdges = True):
    distances = {}                     # in terms of number of edges
    parents = {}                       # parent of node in BFS tree
    layers = [[] for d in range(N(G))] # lists of nodes at each distance.

    finalized = set()
    Q = queue.Queue(); Q.put(s)
    distances[s] = 0; parents[s] = None;
    while not(Q.empty()):
        u = Q.get()
        if u in finalized: continue
        finalized.add(u)
        layers[distances[u]].append(u) 
        for v in G[u]:
            if v in distances or (not exploreZeroEdges and G[u][v]==0): continue
            parents[v] = u
            distances[v] = distances[u] + 1
            Q.put(v)

    return distances, parents, layers

def findAugmentingPath(G, s, t):
    _, parents, _ = BFS(G, s, exploreZeroEdges=False)
    if t not in parents:
        return [], 0
    v = t
    ts_path = [v]
    bottleneck = float('inf')
    while v != s:
        u = parents[v]
        ts_path.append(u)
        if G[u][v] < bottleneck:
            bottleneck = G[u][v]
        v = u
    st_path = ts_path[::-1]
    return st_path, bottleneck

def makeConnected(G):
    def addChild(u):  # adds edge to node that is not parent of u
        for v in G:
            if u not in G[v]:
                addEdge(G, u, v, label=4)
                
    foreachNode(G, lambda u: addChild if degree(G,u) == 0 else None)
    return G

def makeDirected(G):
    foreachEdge(G, lambda u,v: delEdge(G, v,u), safe=True)
    return G

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

def randomDigraphWithSourceSink(n, d, greedy_fail, seed=None): # for FF
    G = randomDigraphDegreeBound(n, d, seed)
    makeDirected(G)
    makeConnected(G)    
    scale = lambda x: int(x*50)
    def scale_edge(u, v):
        G[u][v] = scale(G[u][v])
    traverseGraph(G,
                  lambda u: None,
                  scale_edge)
    s, t = n, n + 1
    addNode(G, s)
    addNode(G, t)
    rng = np.random.default_rng(seed)
    out_list = sampleWoutReplace(n, d, s, rng)
    for nbr in out_list:
        addEdge(G, s, nbr, label=scale(rng.random()))
    in_list = sampleWoutReplace(n, d, t, rng)
    for nbr in in_list:
        addEdge(G, nbr, t, label=scale(rng.random()))
    if greedy_fail:
        addGreedyFailureFF(G, s, t, seed=seed)
    return G, s, t

def assertValidFlow(exF, C, s, t, k, flow): # exF=flow graph, C=capacity graph
    assert graphIsSubset(exF, C, lambda w1, w2: True), \
        f"Nodes {N(exF)} != {N(C)}; Edges {M(exF)} != {M(C)} where F != G"
    assert graphForAll(exF,f_e=lambda u,v: exF[u][v]>=0 and exF[u][v]<=C[u][v]),\
        f"flow not in range [0, cap] on at least one edge"
    xorEdge(exF, t, s, flow)
    inF = copyGraph(exF)
    reverseGraph(inF)
    for u in inF:
        in_f = sum(inF[u][v] for v in inF[u])
        ex_f = sum(exF[u][v] for v in exF[u])
        assert in_f == ex_f, f"Flow conservation failed on {u}"
        if k != 0 and u not in (s,t):
            assert in_f <= k, f"Node {u} surpassed capacity {k}"
    xorEdge(exF, t, s, flow)

def addGreedyFailureFF(G, s, t, seed=123):
    random.seed(seed)
    n = N(G)
    L = 100
    S = L - random.randrange(4, 60)
    for i in range(6):
        addNode(G, n + i)
    addEdge(G, s, n, label=S)
    addEdge(G, n, n + 1, label=S)
    addEdge(G, n + 1, n + 2, label=S)
    addEdge(G, n + 2, t, label=L)
    addEdge(G, s, n + 3, label=L)
    addEdge(G, n + 3, n + 2, label=L)
    addEdge(G, n + 3, n + 4, label=S)
    addEdge(G, n + 4, n + 5, label=S)
    addEdge(G, n + 5, t, label=S)
