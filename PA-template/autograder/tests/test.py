from classlib import *
from problems.DoubleEdgeWeights import *
import unittest

def checkAnswer(i, in_file, out_file, student_out_file):
    return DoubleEdgeWeights.checkAnswer(in_file, out_file, student_out_file)

generate_test_methods(checkAnswer)

if __name__ == "__main__":
    unittest.main()        
