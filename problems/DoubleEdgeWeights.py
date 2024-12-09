from classlib import *
from .problem import Problem

class DoubleEdgeWeights(Problem):
    def solution(self, G):
        def scale(u,v):
            G[u][v] *= 2
        foreachEdge(G, scale)
        return
    
    def createTest(self, index, param):
        (n, visibility_key) = param
        visibility = self.VISIBILITY_MAP[visibility_key]
        G = randomDigraphDegreeBound(n, 3, seed=123 + index)
        writeGraph(G, self.testPath(index, visibility, 'input'))
        self.solution(G)
        writeGraph(G, self.testPath(index, visibility, 'output'))
        return f"n={n}"

    @classmethod
    def checkAnswer(cls, in_file, out_file, student_file):
        try:
            G0, G1, G2 = map(readGraph, [in_file, out_file, student_file])
        except:
            return False, cls.E_FILE_READ
        try:
            assert graphsEqual(G1, G2), cls.E_CORRECTNESS
        except Exception as e:
            return False, e
        return True, cls.CORRECT_MESSAGE
