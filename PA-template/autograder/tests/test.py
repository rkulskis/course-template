from classlib import *
# from submission import tripleEdgeWeights

def checkAnswer(test_i, in_file, out_file, student_out_file):
    # can use test descriptions for custom tests, e.g.:
    # if "triple" in test_descriptions[test_i]:
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

default_weights = {"visible": 1, # weights of gradescope test categories
                   "after_published": 2,
                   "after_due_date": 3,
                   "hidden": 4}
custom_weights = {1: 0.5, 9: 2.5} # based on global test index, e.g. test09.txt

with open("tests/test_descriptions.txt", 'r') as f:
    test_descriptions = [s for s in f.read().splitlines()]
generate_test_methods(default_weights, custom_weights,
                      checkAnswer, test_descriptions)
if __name__ == "__main__":
    unittest.main()        
