all: test
zip: clean
	zip -r autograder.zip . -x "*submission.py"
test:
	python3 tests/test.py
clean:
	find . -name "__pycache__" -exec rm -rf {} +
	find . -name "*~" -exec rm -rf {} +
	find . -name "*.zip" -exec rm -rf {} +
	find . -name "student_out*" -exec rm -rf {} +
