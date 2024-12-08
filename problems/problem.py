from classlib import *  # Assuming necessary imports
from abc import ABC, abstractmethod

class Problem(ABC):
    def __init__(self, test_params):
        self.test_params = test_params
    
    VISIBILITY_MAP = {
        'v': "visible",
        'p': "after_published",
        'd': "after_due_date",
        'h': "hidden",
    }
    E_FILE_READ = "Error reading files"
    CORRECT_MESSAGE = "Student output is correct!"
    E_CORRECTNESS = "Student output is incorrect!"

    @staticmethod
    def testPath(i, visibility, fType):
        assert fType in ['input', 'output']
        return f"./autograder/tests/{visibility}/{fType}s/{fType}{i:02d}.txt"
    def getTestParams(self):
        return self.test_params
    
    @abstractmethod
    def solution(self): pass
    @abstractmethod
    def createTest(self, index, param): pass
    @abstractmethod
    def checkAnswer(cls, in_file, out_file, student_file): pass
