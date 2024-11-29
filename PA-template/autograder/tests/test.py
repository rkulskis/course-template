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

default_weights = {"visible": 1, # weights of gradescope test categories
                   "after_published": 2,
                   "after_due_date": 3,
                   "hidden": 4}
custom_weights = {1: 0.5, 9: 2.5} # based on global test index, e.g. test09.txt
generate_test_methods(default_weights, custom_weights, checkAnswer)
if __name__ == "__main__":
    unittest.main()        
