T ?= -1
STUDENT_OUTPUT = student_output.txt
CORRECT_OUTPUT = ./outputs/output$(T_f).txt

diff_check = bash -c "diff <(head -n 1 $(STUDENT_OUTPUT)) \
<(head -n 1 $(CORRECT_OUTPUT)) > /dev/null && \
echo 'Test $(T) passed!' || echo 'Test $(T) failed.'"

ifneq ($(T), -1)
  T_f = $(shell printf "%02d" $(T))
  INPUT = ./inputs/input$(T_f).txt
all:
	@python3 submission.py $(INPUT) $(STUDENT_OUTPUT)
	@$(diff_check)
else
all:
	@for input in inputs/input*.txt; do \
		T=$$(echo $$input | tr -d -c 0-9); \
		CORRECT_OUTPUT=$$(echo $$input | sed 's|input|output|g'); \
		python3 submission.py $$input $(STUDENT_OUTPUT); \
		bash -c "diff <(head -n 1 $(STUDENT_OUTPUT)) \
		<(head -n 1 $$CORRECT_OUTPUT) > /dev/null && \
		echo 'Test $$T passed!' || echo 'Test $$T failed.'"; \
	done

endif

zip: clean
	zip -r start_code.zip .

clean:
	rm -rf *~ __pycache__ $(STUDENT_OUTPUT)
