all: test
zip: clean
	zip -r autograder.zip . -x submission.py
test:
	python3 tests/test.py
clean:
	rm -rf __pycache__ *~ *.zip student_out*
