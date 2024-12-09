# Autograder template

## About
This template allows for course staff to easily create new assignments with
maximal automation and speed. The goal is to standardize the creation of
autograders and minimize the footprint of non-boiler-plate code.

Currently this is a python-based framework for python assignments; however, I
want to extend it to other languages as well as add:
 * **enhanced security features**: currently code has 5 second timeout and is ran
 as a subprocess; however, I want to add socket and filesystem reduced privilege
 for the student code as well as nonce hashes of output to ensure student code
 cannot fabricate fake results in the results.json.
 * **plagiarism checkers**: against the solution and all past submissions. Probably
 do this asynchronously as a daemon process since only the course staff should 
 see the results of whether a student submission was possibly plagiarized.
 * **portability to grade assignments on your own cluster** to circumvent potential
   kernel incompatibilities with a custom interface for self-hosting and grading
   code (c.f. [NERC_autograder](https://github.com/OpenOSOrg/NERCautograder))
 * **support for other languages**, e.g. Java, C, C++, Go, Rust, SQL. This requires
 polymorphism in the code AST parsing because tree-sitter has different node
 names for different languages. Hopefully this shouldn't be too difficult.
 * **Integration with GitHub repos/organizations** so you can have an organization
 for your course (e.g. Boston-University-CS330) and a repo for each year with a
 starter code directory for each assignment in the student repo, and a private
 course repo for each year with corresponding source code
 * **Integration with GitHub classroom** (since it's free! and gradescope is not
 free). Additionally, this requires student to use git which is an essential
 programming skill. Also, the GitHub actions show live results so students
 can see which tests they pass/fail in real-time rather than looking at the 
 gradescope loading screen, waiting for all their results to arrive at the end.
 
 
## Porting to existing framework of code
Because forks are not private, to pull this framework into your private course
repository run:
```
git remote add upstream git@github.com:rkulskis/autograder-template.git
git remote set-url --push upstream no_push
git config --global merge.keeporigin.driver "cp -f %A %O"
echo "classlib.py merge=keeporigin" > .gitattributes
git pull upstream main --allow-unrelated-histories
```

## Getting started
### Demo
```bash
# install some packages with pip, e.g. tree-sitter parser for code autogeneration
make install 
# demo all of the code generation with PA-template, test running, and zipping
make demo
```

### Customization
To create a new assignment do the following - note that step 7 is equivalent to
running steps 3 to 6 in one command:
0. `cp -r PA-template PA-assignment-name` to make your new programming
assignment directory
1. Add any necessary library functions to `problems/classlib.py`
2. For each part in your PA, create a new classfile in `problems/`
   * Inherits from the problem abstract base class
   * Implements:
	 * solution(): solution for this problem
		 * if you make `starter_solution()` then this will be included as the
		 starter code function for this problem. These solution methods are
		 compiled to the name of the class with the first letter in lowercase if
		 the second letter of the class name is lowercase. E.g.:
			 * `FindCycles.solution` becomes `findCycles()`
			 * `BFS.solution` becomes `BFS()`
	 * createTest(): create inputs and expected outputs as well as test
	 descriptions
	 * checkAnswer(): given {in,out}put.txt and student submission, return a
	 bool for whether student passed as well as test run info
3. In `PA-assignment-name/create_assignment_files.py`:
   * Import all of your problem classes
   * Create a `problems` list of class instances with test parameter lists for
   each instance as the constructor agument
   * Implement `interpretCommandLineArgs(args)` to read input and based on the
   input file, apply one of your problem functions and write the output
   * Run `make` in the same directory to autogenerate:
	 * `autograder/` template code, this should work for any assignment, as if
	 you need additional python packages you can add them to the
	 `PA-assignment-name/Makefile` definition of `REQUIREMENTS_TXT`
	 * test files
	 * `test.py`: autograder test script
	 * `{starter_code,solution}/submission.py`
	 * `classlib.py` files that are subsets of the entire classlib such that
	 only the necessary functions are included in each compilation (e.g. student
	 starter_code should only have necessary functions for the assignment as
	 dictated by the `solution/submission.py`
	 * Obsfucated version of autograder, i.e. undecipherable version with
	 compiled libraries for each platform: darwin64, linux, windows. This is in
	 `starter_code` for students, with only the visible inputs from
	 autograder. Also, we exclude `createTest(),solution()` from problem class
	 definitions to be absolutely sure there's no critical information in the
	 starter code distribution
4. Run `make` in `PA-assignment-name/autograder/` to run all tests
5. Run `make` in `PA-assignment-name/starter_code/` to run all visible tests
   * This runs the imported solution from `solution/submission.py` but you can
   comment this out if you want for local testing
6. Run `make zip` inside `PA-assignment-name` to make `starter_code.zip` (to
distribute to the students) and `autograder.zip` to upload to gradescope
7. To run all of steps 3-6 in one command, in the root of the repo run: `make
DEMO_DIR=PA-assignment-name`

## Background
Below is a `tree` of the code including descriptions for the files and how to
use them. The template code just generates random graph tests for an assignment
which has one function, `doubleEdgeWeights()`.

This structure is targeted toward comparing output files from python code
submissions, but can be extended to other languages as needed.
 
### Tree
```bash
⭐: Modify these files to make a new assignment
.
├── Makefile
├── PA-template
│   ├── assignment-description.{tex,pdf} ⭐
│   ├── create_assignment_files.py ⭐
│   └── Makefile
├── problems ⭐
│   ├── classlib.py
│   ├── {problems_for_this_PA,...}.py
│   └── problem.py
└── README.md
```
### Tree (with autogenerated files)
```bash
KEY:
⚙️: Autogenerated files and directories using Makefile
.
├── Makefile
├── PA-template
│   ├── assignment-description.{tex,pdf}
│   ├── autograder ⚙️
│   │   ├── classlib.py
│   │   ├── Makefile
│   │   ├── requirements.txt
│   │   ├── run_autograder
│   │   ├── run_tests.py
│   │   ├── setup.sh
│   │   ├── submission.py -> ../starter_code/submission.py
│   │   └── tests
│   │       ├── {after_due_date, after_published, hidden, visible}
│   │       │   ├── inputs
│   │       │   │   └── input{%2d}.txt
│   │       │   └── outputs
│   │       │       └── output{%2d}.txt
│   │       ├── problems
│   │       │   ├── {problems_for_this_PA,...}.py
│   │       │   └── problem.py
│   │       ├── test_descriptions.txt
│   │       └── test.py
│   ├── autograder.zip ⚙️
│   ├── classlib.py -> ../problems/classlib.py ⚙️
│   ├── create_assignment_files.py
│   ├── Makefile
│   ├── problems -> ../problems ⚙️
│   ├── solution ⚙️
│   │   └── submission.py
│   ├── starter_code ⚙️
│   │   ├── assignment-description.pdf -> ../assignment-description.pdf
│   │   ├── classlib.py
│   │   ├── dist
│   │   │   ├── classlib.py
│   │   │   ├── problems
│   │   │   │   ├── {problems_for_this_PA,...}.py
│   │   │   │   └── problem.py
│   │   │   ├── pyarmor_runtime_000000
│   │   │   │   ├── darwin_x86_64
│   │   │   │   │   └── pyarmor_runtime.so
│   │   │   │   ├── __init__.py
│   │   │   │   ├── linux_x86_64
│   │   │   │   │   └── pyarmor_runtime.so
│   │   │   │   ├── __pycache__
│   │   │   │   │   └── __init__.cpython-312.pyc
│   │   │   │   └── windows_x86_64
│   │   │   │       └── pyarmor_runtime.pyd
│   │   │   └── test.py
│   │   ├── Makefile
│   │   ├── student_output.txt
│   │   ├── submission.py
│   │   └── tests
│   │       ├── inputs -> ../../autograder/tests/visible/inputs
│   │       ├── outputs -> ../../autograder/tests/visible/outputs
│   │       └── test_descriptions.txt -> ../../autograder/tests/test_descriptions.txt
│   └── starter_code.zip ⚙️
├── problems
│   ├── classlib.py
│   ├── DoubleEdgeWeights.py
│   └── problem.py
└── README.md
```
