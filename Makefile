CXX        = g++
WARNINGS   = -Wall -Wextra -Wpedantic # -Waddress -Wbool-compare -Wconversion -Wdeprecated
INCLUDE    = -I ./include
OPT        = -O2 -march=native
OPENMP     = -fopenmp
# TODO: automatically detect python version
PYTHON     = python3.8


# requires installation of https://github.com/google/benchmark
GOOGLE_BENCHMARK_INC_DIR = /home/matteo/hpc/programs/benchmark/include
GOOGLE_BENCHMARK_LIB_DIR = /home/matteo/hpc/programs/benchmark/build/src


# program specific configuration, to be put in separate file
CONFIG          = -DREMOVE_EXP
RESOLUTION      = 512
OMP_NUM_THREADS = 8


src        = $(wildcard src/*.cpp)
obj        = $(patsubst src/%.cpp,build/%.o,$(src))


targets += porting benchmark analysis regression


all: $(targets)


porting: build/main.o build/kernels.o build/units.o build/utils.o build/slm.o build/scheduling.o
	$(CXX) $(OPENMP) -o $@ $^


benchmark: build/benchmark.o build/kernels.o build/units.o build/utils.o build/slm.o build/pupil.o build/scheduling.o
	$(CXX) $(OPENMP) -o $@ $^ -L $(GOOGLE_BENCHMARK_LIB_DIR) -lbenchmark -lpthread


analysis: build/analysis.o build/slm.o build/kernels.o build/utils.o build/units.o build/scheduling.o
	$(CXX) $(OPENMP) -o $@ $^


regression: build/regression.o build/slm.o build/kernels.o build/utils.o build/units.o build/scheduling.o
	$(CXX) $(OPENMP) -o $@ $^


bench: benchmark
	OMP_NUM_THREADS=$(OMP_NUM_THREADS) ./$^


output.bin: porting
	./$^ $@


report: output.bin
	$(PYTHON) python/compare_with_serial.py $^


godbolt: src/kernels.cpp include/kernels.hpp include/utils.hpp
	$(CXX) -E $(INCLUDE) $< -o $@


# for now don't include header file dependencies
build/%.o: src/%.cpp
	$(CXX) -c $(WARNINGS) $(INCLUDE) $(OPT) $(OPENMP) $(CONFIG) -o $@ $<


build/benchmark.o: src/benchmark.cpp
	$(CXX) -c $(WARNINGS) $(INCLUDE) -I $(GOOGLE_BENCHMARK_INC_DIR) $(OPT) -o $@ $<


# don't exclude to have in the future only a python driver/compiler
src/pupil.cpp: generator/pupil_index_generator.py
	RESOLUTION=$(RESOLUTION) OMP_NUM_THREADS=$(OMP_NUM_THREADS) $(PYTHON) $^ > $@


src/scheduling.cpp: generator/scheduling.py
	RESOLUTION=$(RESOLUTION) OMP_NUM_THREADS=$(OMP_NUM_THREADS) $(PYTHON) $^ > $@


.PHONY clean:
clean:
	rm -f $(obj) $(targets) output.bin src/pupil.cpp src/scheduling.cpp
