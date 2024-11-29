all: demo
demo:
	$(MAKE) -C PA-template				# generates assignment files and starter_code lib
	$(MAKE) -C PA-template/starter_code # run all starter tests on submission.py
	$(MAKE) T=1 -C PA-template/starter_code # run only starter test 1
	$(MAKE) -C PA-template/autograder # run all autograder tests on submission.py
	$(MAKE) zip -C PA-template				# get autograder.zip and starter_code.zip
