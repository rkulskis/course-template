import os, subprocess, time
import random, re
import unittest
from gradescope_utils.autograder_utils.decorators import weight,visibility,number
from classlib import *

def checkAnswer(in_file, out_file, student_out_file):
    try:
        G1 = readGraph(student_out_file)
        G2 = readGraph(out_file)
        G0 = readGraph(in_file)
    except:
        return False, "Error reading files"
    try:
        assert graphsEqual(G1, G2), "Student output is incorrect!"
    except Exception as e:
        return False, e
    return True, "Student output is correct!"

prefix = ""                     # necessary for local testing
def execTest(i, visibility):
    in_file = f"tests/{visibility}/inputs/input{i:02d}.txt"
    out_file = f"tests/{visibility}/outputs/output{i:02d}.txt"
    student_out_file = f"{prefix}student_output.txt"
    
    time_bound = 60
    start = time.time()
    try:
        ret = subprocess.run(['python3', f"{prefix}submission.py",
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
    
    passed, feedback = checkAnswer(in_file, out_file, student_out_file)
    info = f"Submitted program terminated in {runtime:.2f} seconds.\n{feedback}"
    return passed, info

class Test(unittest.TestCase):  # auto generate tests
    def formatted_test_run(self, i, visibility):
        print(f"==========Testing input {i:02d} ({visibility})==========")
        passed, info = execTest(i, visibility)
        self.fail(info) if not passed else print(info)

    def generate_test(self, i, visibility, w):
        @weight(w)
        @number(i)
        def test_func(self):
            self.formatted_test_run(i, visibility)
        return test_func

default_weights = {"visible": 1, # weights of gradescope test categories
                   "after_published": 2,
                   "after_due_date": 3,
                   "hidden": 4}
custom_weights = {1: 0.5, 9: 2.5} # based on global test index, e.g. test09.txt
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
        setattr(Test, f'test_{i}', Test().generate_test(i, visibility, w))
    
if __name__ == "__main__":
    prefix = "../starter_code/"
    unittest.main()        
