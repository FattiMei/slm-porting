CXX        = g++
WARNINGS   = -Wall -Wextra -Wpedantic # -Waddress -Wbool-compare -Wconversion -Wdeprecated
INCLUDE    = -I ./include
OPT        = -O2 -march=native
OPENMP     = -fopenmp
PYTHON     = python3.8
CONFIG     = -DREMOVE_EXP


src        = $(wildcard src/*.cpp)
obj        = $(patsubst src/%.cpp,build/%.o,$(src))


targets += porting benchmark analysis regression


all: $(targets)


porting: build/main.o build/kernels.o build/units.o build/utils.o build/slm.o
	$(CXX) $(OPENMP) -o $@ $^


benchmark: build/benchmark.o build/kernels.o build/units.o build/utils.o build/slm.o
	$(CXX) $(OPENMP) -o $@ $^


analysis: build/analysis.o build/slm.o build/kernels.o build/utils.o build/units.o
	$(CXX) $(OPENMP) -o $@ $^


regression: build/regression.o build/slm.o build/kernels.o build/utils.o build/units.o
	$(CXX) $(OPENMP) -o $@ $^


bench: benchmark
	./$^ | tee bench.txt


output.bin: porting
	./$^ $@


report: output.bin
	$(PYTHON) python/compare_with_serial.py $^


# for now don't include header file dependencies
build/%.o: src/%.cpp
	$(CXX) -c $(WARNINGS) $(INCLUDE) $(OPT) $(OPENMP) $(OPT) $(CONFIG) -o $@ $<


.PHONY clean:
clean:
	rm -f $(obj) $(targets) output.bin
