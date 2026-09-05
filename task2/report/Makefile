# Builds task2/report/main.tex -> task2/task2_report.pdf (the actual submission file).
# Kept separate from the graded task2/Makefile, which must not be modified.

MAIN := main
OUT  := ../task2_report.pdf

.PHONY: all clean

all: $(OUT)

$(OUT): $(MAIN).tex
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(MAIN).tex
	cp $(MAIN).pdf $(OUT)

clean:
	latexmk -C
	rm -f $(OUT)
