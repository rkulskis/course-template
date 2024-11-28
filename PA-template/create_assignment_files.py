import sys, os
import random
import re
from solution import *
from classlib import *
from collections import defaultdict

# auto generates subset of library used in solution to place in ./starter_code/
def create_starter_library(sol_file="solution.py", lib_file="classlib.py"):
    regex_fcall = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    regex_fdef = r'^\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    regex_section_header = r'^#{60}'
    
    with open(lib_file, "r") as f:
        lib_code = f.read()
    with open(sol_file, "r") as sol_file:
        sol_code = sol_file.read()
    sol_fcalls = set(re.findall(regex_fcall, sol_code))
    
    fdefs = {}
    G_dep = defaultdict(set)
    for match in re.finditer(regex_fdef, lib_code, re.MULTILINE):
        fname = match.group(1)
        fdefs[fname] = match.start()
    sorted_tups = list(sorted(fdefs.items(), key=lambda i: i[1]))
    for i, (fname, start) in enumerate(sorted_tups):
        end = sorted_tups[i+1][1] if i+1 < len(sorted_tups) else len(lib_code)
        f_code = lib_code[start:end]
        fcalls = re.findall(regex_fcall, f_code)
        for f in fcalls:
            if f in fdefs:
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
    if re.match(regex_section_header, starter_lib[-1]):
        starter_lib.pop(-1)
            
    with open(f"./starter_code/{lib_file}", "w") as f:
        f.write("\n".join(starter_lib))
    print(f"Compiled starter_code/{lib_file} only containing functions needed.")

def create_test_files():
    visibility_map = {'v': "visible", 'p': "after_published",
                      'd': "after_due_date", 'h': "hidden"} # for gradescope UI
    params = [(5, 'v'), (5, 'v'), (200, 'v'),
              (5, 'p'), (5, 'p'), (200, 'p'),
              (5, 'd'), (5, 'd'), (200, 'd'),
              (123, 'h'), (123, 'h'), (123, 'h')]
    for i, param in enumerate(params):
        n, v = param
        visibility = visibility_map[v]
        G = randomDigraphDegreeBound(n, 3, seed=123+i)
        prefix = f"./autograder/tests/{visibility}"
        writeGraph(G, f"{prefix}/inputs/input{i:02d}.txt")
        doubleEdgeWeights(G)
        writeGraph(G, f"{prefix}/outputs/output{i:02d}.txt")

if __name__ == "__main__":
    create_test_files()
    create_starter_library()    
