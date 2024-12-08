VISIBILITIES = {visible,after_published,after_due_date,hidden}
TEST_DIRS = ./autograder/tests/$(VISIBILITIES)/{inputs,outputs}
CLASS_LIB = classlib.py

define STARTER_MAKEFILE
all: test
test:
	python3 dist/test.py
zip: clean
	zip -r starter_code.zip .
clean:
	rm -rf *~ __pycache__ $(STUDENT_OUTPUT)
endef
export STARTER_MAKEFILE

all: clean
	bash -c "mkdir -p $(TEST_DIRS) starter_code/tests solution"
	bash -c "ln -sfrt ./ ../problems/$(CLASS_LIB)"
	bash -c "ln -sfrt ./ ../problems"
	bash -c "ln -sfrt ./autograder ./starter_code/submission.py"
	bash -c "ln -sfrt ./autograder/tests ./autograder/$(CLASS_LIB)"
	bash -c "ln -sfrt ./starter_code ./assignment-description.pdf"	
	bash -c "ln -sfrt ./starter_code/tests ./autograder/tests/test_descriptions.txt"
	bash -c "ln -sfrt ./starter_code/tests \
./autograder/tests/visible/{inputs,outputs}"
	python3 create_assignment_files.py # also creates problem_paths.txt
	rm -rf starter_code/dist
	make obsfucate-test-code
	echo "$$STARTER_MAKEFILE" > starter_code/Makefile
obsfucate-test-code:
	cd autograder/tests && \
	pipx run pyarmor gen  \
	--platform windows.x86_64 \
	--platform linux.x86_64 \
	--platform darwin.x86_64 \
	test.py classlib.py $$(cat problem_paths.txt) && \
	mkdir -p dist/problems && \
	cd dist && \
	mv $$(cat ../problem_paths.txt | xargs -n 1 basename) problems && \
	cd .. && \
	rm -rf problem_paths.txt && \
	cd ..
	mv autograder/tests/dist starter_code
zip:
	$(MAKE) zip -C autograder
	$(MAKE) zip -C starter_code
	mv autograder/*.zip starter_code/*.zip .
clean:
	cat ../.gitignore | while read -r pattern; \
	do find . -name "$$pattern" -exec rm -rf {} +; done
