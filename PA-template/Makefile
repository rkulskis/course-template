VISIBILITIES = {visible,after_published,after_due_date,hidden}
TEST_DIRS = ./autograder/tests/$(VISIBILITIES)/{inputs,outputs}

all: tests
tests: clean
	bash -c "mkdir -p $(TEST_DIRS)"
	bash -c "ln -sfrt ./starter_code/ ./autograder/tests/visible/{inputs,outputs}"
	python3 create_assignment_files.py
clean:
	cat ../.gitignore | while read -r pattern; \
	do find . -name "$$pattern" -exec rm -rf {} +; done
