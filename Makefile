SHELL=/bin/bash
# can change to your new PA to make everything
DIR = PA-template
PUBLIC_REPO = ../autograder-template-students
REQUIREMENTS = timeout-decorator gradescope-utils numpy tree-sitter tree-sitter-python
all: demo
publish:
	if [ -f $(DIR)/Makefile ]; \
		then $(MAKE) -C $(DIR); \
	fi # WA (written assignments) don't have Makefiles	
	if [ -d $(PUBLIC_REPO)/$(DIR) ]; \
		then rm -rf $(PUBLIC_REPO)/$(DIR); \
	fi # want to get rid of old files
	cp -rL $(DIR) $(PUBLIC_REPO)
	cd $(PUBLIC_REPO); git add . ; git commit -m "publish" ; git push
demo:
	$(MAKE) -C $(DIR)				# generates assignment files and starter_code lib
	$(MAKE) -C $(DIR)/starter_code # run all starter tests on submission.py
	$(MAKE) -C $(DIR)/autograder # run all autograder tests on submission.py
	$(MAKE) zip -C $(DIR)				# get autograder.zip and starter_code.zip
install:														# only need to run once
  # can change to pipx, but cannot find tree-sitter for some reason	
	echo $(REQUIREMENTS) | xargs -n 1 pip install	--break-system-packages
clean:
	cat .gitignore | while read -r pattern; \
	do find . -name "$$pattern" -exec rm -rf {} +; done
	cat .gitignore | while read -r pattern; \
	do find . \( -type f -o -type l \) -wholename "$$pattern" -exec rm {} +; done
	cat .gitignore | while read -r pattern; \
	do find . -type d -wholename "$$pattern" -exec rm -rf {} +; done
