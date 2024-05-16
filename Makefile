CXX        = g++
WARNINGS   = -Wall -Wextra -Wpedantic -Waddress -Wbool-compare -Wconversion -Wdeprecated
OPTFLAGS   = -O2 -march=native
OMPFLAGS   = -fopenmp
PROFFLAGS  = -pg
INCLUDE    = -I ./include
PYTHON     = python3.8


src        = $(wildcard src/*.cpp)
obj        = $(patsubst src/%.cpp,build/%.o,$(src))


targets += porting benchmark analysis regression


all: $(targets)


porting: build/main.o build/kernels.o build/units.o build/utils.o build/slm.o
	$(CXX) $(OMPFLAGS) -o $@ $^


benchmark: src/benchmark.cpp src/kernels.cpp src/units.cpp src/utils.cpp src/slm.cpp
	$(CXX) $(INCLUDE) $(WARNINGS) $(OPTFLAGS) $(OMPFLAGS) -o $@ $^


analysis: build/analysis.o build/slm.o build/kernels.o build/utils.o build/units.o
	$(CXX) $(OMPFLAGS) -o $@ $^


regression: build/regression.o build/slm.o build/kernels.o build/utils.o build/units.o
	$(CXX) $(OMPFLAGS) -o $@ $^


bench: benchmark
	./$^ | tee bench.txt


output.bin: porting
	./$^ $@


report: output.bin
	$(PYTHON) python/compare_with_serial.py $^


profile:
	$(CXX) $(OPTFLAGS) $(OMPFLAGS) $(PROFFLAGS) $(INCLUDE) -o $@ $(src)


# (INCOMPLETE) in the future this will definetely be automatically generated
build/%.o: src/%.cpp include/utils.hpp include/slm.hpp include/units.hpp
	$(CXX) -c $(WARNINGS) $(OMPFLAGS) $(OPTFLAGS) $(INCLUDE) -o $@ $<


.PHONY clean:
clean:
	rm -f $(obj) $(targets) output.bin
