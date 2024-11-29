VISIBILITIES = {visible,after_published,after_due_date,hidden}
TEST_DIRS = ./autograder/tests/$(VISIBILITIES)/{inputs,outputs}
CLASS_LIB = classlib.py

all: tests
tests: clean
	bash -c "mkdir -p $(TEST_DIRS)"
	bash -c "ln -sfrt ./starter_code/ ./$(CLASS_LIB)" # temporary
	bash -c "ln -sfrt ./starter_code/ ./autograder/tests/visible/{inputs,outputs}"
	python3 create_assignment_files.py # will override library with subset of funcs
zip: tests
	$(MAKE) zip -C autograder
	$(MAKE) zip -C starter_code
	mv autograder/*.zip starter_code/*.zip .
clean:
	find . -name "inputs" -o -name "outputs" -exec rm -rf {} +
