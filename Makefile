CXX        = g++
CXXFLAGS   = -Wall -Wextra -Wpedantic
OPTFLAGS   = -O2 -march=native
PROFFLAGS  = -pg
INCLUDE    = -I ./include
PYTHON     = python3.8


src        = $(wildcard src/*.cpp)
obj        = $(patsubst src/%.cpp,build/%.o,$(src))


targets += porting benchmark


all: $(targets)


porting: build/main.o build/kernels.o build/units.o build/utils.o build/slm.o
	$(CXX) -o $@ $^


benchmark: build/benchmark.o build/kernels.o build/units.o build/utils.o build/slm.o
	$(CXX) -o $@ $^


bench: benchmark
	./$^ | tee bench.txt


output.bin: porting
	./$^ $@


regression: output.bin
	$(PYTHON) python/compare.py reference.bin $^


report: output.bin
	$(PYTHON) python/compare_with_serial.py $^


profile:
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(PROFFLAGS) $(INCLUDE) -o $@ $(src)


# (INCOMPLETE) in the future this will definetely be automatically generated
build/%.o: src/%.cpp include/utils.hpp include/slm.hpp include/units.hpp
	$(CXX) -c $(CXXFLAGS) $(OPTFLAGS) $(INCLUDE) -o $@ $<


.PHONY clean:
clean:
	rm -f $(obj) $(targets) output.bin
