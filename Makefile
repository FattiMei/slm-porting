CXX        = g++
WARNINGS   = -Wall -Wextra -Wpedantic -Waddress -Warith-conversion -Wbool-compare -Wconversion -Wdeprecated
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


benchmark: src/benchmark.cpp src/kernels.cpp src/units.cpp src/utils.cpp src/slm.cpp
	$(CXX) $(WARNINGS) $(OPTFLAGS) -o $@ $^


bench: benchmark
	./$^ | tee bench.txt


output.bin: porting
	./$^ $@


regression: output.bin
	$(PYTHON) python/compare.py reference.bin $^


report: output.bin
	$(PYTHON) python/compare_with_serial.py $^


profile:
	$(CXX) $(OPTFLAGS) $(PROFFLAGS) $(INCLUDE) -o $@ $(src)


# (INCOMPLETE) in the future this will definetely be automatically generated
build/%.o: src/%.cpp include/utils.hpp include/slm.hpp include/units.hpp
	$(CXX) -c $(WARNINGS) $(OPTFLAGS) $(INCLUDE) -o $@ $<


.PHONY clean:
clean:
	rm -f $(obj) $(targets) output.bin
