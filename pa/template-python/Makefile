SHELL=/bin/bash
VISIBILITIES = {visible,after_published,after_due_date,hidden}
TEST_DIRS = ./autograder/tests/$(VISIBILITIES)/{inputs,outputs}
CLASS_LIB = classlib.py

define STARTER_MAKEFILE
all: test
test: 
	sudo docker run \
		-v `pwd | xargs realpath`:/starter \
		-it rkulskis/course-template-toolchain:latest \
		/bin/bash -c "cd starter; python3 dist/test.py"
zip: clean
	zip -r starter_code.zip .
clean:
	rm -rf *~ __pycache__ $(STUDENT_OUTPUT)
endef
export STARTER_MAKEFILE

define AUTOGRADER_MAKEFILE
all: test
zip: clean
	zip -r autograder.zip . -x "*submission.py"
test:
	@python3 tests/test.py
clean:
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*~" -exec rm -rf {} +
	find . -name "*.zip" -exec rm -rf {} +
	find . -name "student_out*" -exec rm -rf {} +
endef
export AUTOGRADER_MAKEFILE

define RUN_AUTOGRADER
#!/usr/bin/env bash
# Set up autograder files
cp /autograder/submission/submission.py /autograder/source/ &> /dev/null
cp /autograder/submission/submission.py /autograder/source/tests/
cd /autograder/source
python3 run_tests.py > /autograder/results/results.json
endef
export RUN_AUTOGRADER

define RUN_TESTS
import unittest
from gradescope_utils.autograder_utils.json_test_runner import JSONTestRunner
if __name__ == '__main__':
    suite = unittest.defaultTestLoader.discover('tests')
    JSONTestRunner(visibility='visible').run(suite)
endef
export RUN_TESTS

define SETUP_SH
#!/usr/bin/env bash
apt-get -y install openjdk-11-jdk
apt-get install -y python3 python3-pip python3-dev firejail
pip3 install -r /autograder/source/requirements.txt
mkdir -p /run/firejail/dbus			# necessary for firejail mount
chmod 444 /autograder/source/classlib.py # make classlib.py read-only
chmod -R 444 /autograder/source/tests		 # make all test files read-only
endef
export SETUP_SH

define REQUIREMENTS_TXT
gradescope-utils
numpy
timeout-decorator
endef
export REQUIREMENTS_TXT

all: clean
	mkdir -p autograder
	echo "$$AUTOGRADER_MAKEFILE" > autograder/Makefile
	echo "$$RUN_AUTOGRADER" > autograder/run_autograder
	echo "$$RUN_TESTS" > autograder/run_tests.py
	echo "$$SETUP_SH" > autograder/setup.sh
	echo "$$REQUIREMENTS_TXT" > autograder/requirements.txt

	-rm -rf starter_code/tests		# since we copy over files and don't symlink
	mkdir -p $(TEST_DIRS) starter_code/tests solution
	cd autograder && \
	ln -sf ../starter_code/classlib.py classlib.py && \
	ln -sf ../solution/submission.py submission.py;
	ln -sf ../problems/classlib.py classlib.py && \
	ln -sf ../problems problems;

	ln -sf ../problems/classlib.py classlib.py && \
	ln -sf ../problems/ problems
	python3 create_assignment_files.py # also creates problem_paths.txt
	cd starter_code && \
	ln -sf ../assignment-description.pdf assignment-description.pdf && \
	cd tests && \
	cp -r ../../autograder/tests/visible/inputs . && \
	cp -r ../../autograder/tests/visible/outputs . && \
	cp ../../autograder/tests/test_descriptions.txt .
	rm -rf starter_code/dist
	make obsfucate-test-code
	echo "$$STARTER_MAKEFILE" > starter_code/Makefile
obsfucate-test-code:
	cd autograder/tests && \
	pyarmor gen  \
	--platform windows.x86_64 \
	--platform linux.x86_64 \
	--platform darwin.x86_64 \
  --platform darwin.arm64 \
  --platform linux.aarch64 \
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
	cat ../../.gitignore | while read -r pattern; \
	do find . -name "$$pattern" -exec rm -rf {} +; done
	cat ../../.gitignore | while read -r pattern; \
	do find . \( -type f -o -type l \) -wholename "$$pattern" -exec rm {} +; done
	cat ../../.gitignore | while read -r pattern; \
	do find . -type d -wholename "$$pattern" -exec rm -rf {} +; done
	-rm classlib.py problems
