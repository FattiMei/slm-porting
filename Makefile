CXX        = g++
CXXFLAGS   = -Wall -Wextra -Wpedantic
OPTFLAGS   = -O2
PROFFLAGS  = -pg
BENCHFLAGS = -lbenchmark -lpthread
INCLUDE    = -I ./include
PYTHON     = python3.8


src        = src/main.cpp src/serial.cpp src/utils.cpp src/units.cpp
obj        = $(patsubst src/%.cpp,build/%.o,$(src))
benchmarks = $(wildcard benchmark/*.cpp)


targets += porting
targets += $(patsubst benchmark/%.cpp,build/%,$(benchmarks))


all: $(targets)


porting: $(obj)
	$(CXX) -o $@ $^


output.bin: porting
	./$^ $@


regression: output.bin
	$(PYTHON) python/compare.py reference.bin $^


report: output.bin
	$(PYTHON) python/compare_with_serial.py $^


profile:
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(PROFFLAGS) $(INCLUDE) -o $@ $(src)


build/%: benchmark/%.cpp build/serial.o build/utils.o build/units.o
	$(CXX) $(CXXFLAGS) $(OPTFLAGS) $(INCLUDE) -o $@ $^ $(BENCHFLAGS)


# (INCOMPLETE) in the future this might be automatically generated
build/%.o: src/%.cpp include/utils.hpp include/slm.hpp include/units.hpp
	$(CXX) -c $(CXXFLAGS) $(OPTFLAGS) $(INCLUDE) -o $@ $<


.PHONY clean:
clean:
	rm -f $(obj) porting profile output.bin
