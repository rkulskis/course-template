SHELL=/bin/bash
# can change to your new PA to make everything
DEMO_DIR = PA-template
all: demo
demo:
	$(MAKE) -C $(DEMO_DIR)				# generates assignment files and starter_code lib
	$(MAKE) -C $(DEMO_DIR)/starter_code # run all starter tests on submission.py
	$(MAKE) -C $(DEMO_DIR)/autograder # run all autograder tests on submission.py
	$(MAKE) zip -C $(DEMO_DIR)				# get autograder.zip and starter_code.zip
clean:
	cat .gitignore | while read -r pattern; \
	do find . -name "$$pattern" -exec rm -rf {} +; done
	cat .gitignore | while read -r pattern; \
	do find . \( -type f -o -type l \) -wholename "$$pattern" -exec rm {} +; done
	cat .gitignore | while read -r pattern; \
	do find . -type d -wholename "$$pattern" -exec rm -rf {} +; done

