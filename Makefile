SHELL=/bin/bash
DIR = pa/template-python
PA-DIRS := $(filter-out problems,$(wildcard pa/*/))
REQUIREMENTS = timeout-decorator gradescope-utils numpy tree-sitter tree-sitter-python
all: demo
publish:
demo:
	$(MAKE) -C $(DIR)				# generates assignment files and starter_code lib
	$(MAKE) -C $(DIR)/starter_code # run all starter tests on submission.py
	$(MAKE) -C $(DIR)/autograder # run all autograder tests on submission.py
	$(MAKE) zip -C $(DIR)				# get autograder.zip and starter_code.zip
install:														# only need to run once
  # can change to pipx, but cannot find tree-sitter for some reason	
	echo $(REQUIREMENTS) | xargs -n 1 pip install	--break-system-packages
clean:
	@cat .gitignore | while read -r pattern; \
	do find . -name "$$pattern" -exec rm -rf {} +; done
	@cat .gitignore | while read -r pattern; \
	do find . \( -type f -o -type l \) -wholename "$$pattern" -exec rm {} +; done
	@cat .gitignore | while read -r pattern; \
	do find . -type d -wholename "$$pattern" -exec rm -rf {} +; done
	@for dir in $(PA-DIRS); do \
		if [ -d "$$dir" ]; then \
			$(MAKE) -C "$$dir" clean; \
		fi \
	done
