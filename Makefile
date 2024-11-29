# can change to your new PA to make everything
DEMO_DIR = PA-template
all: demo
demo:
	$(MAKE) -C $(DEMO_DIR)				# generates assignment files and starter_code lib
	$(MAKE) -C $(DEMO_DIR)/starter_code # run all starter tests on submission.py
	$(MAKE) T=1 -C $(DEMO_DIR)/starter_code # run only starter test 1
	$(MAKE) -C $(DEMO_DIR)/autograder # run all autograder tests on submission.py
	$(MAKE) zip -C $(DEMO_DIR)				# get autograder.zip and starter_code.zip
