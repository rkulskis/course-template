all: test
zip: clean
	zip -r autograder.zip .
test:
	python3 tests/test.py
clean:
	rm -rf __pycache__ *~ *.zip student_out*
