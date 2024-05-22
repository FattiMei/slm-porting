CXX        = g++
WARNINGS   = -Wall -Wextra -Wpedantic # -Waddress -Wbool-compare -Wconversion -Wdeprecated
INCLUDE    = -I ./include
OPT        = -O2 -march=native
OPENMP     = -fopenmp
PYTHON     = python3.8


# requires installation of https://github.com/google/benchmark
GOOGLE_BENCHMARK_INC_DIR = /home/matteo/hpc/programs/benchmark/include
GOOGLE_BENCHMARK_LIB_DIR = /home/matteo/hpc/programs/benchmark/build/src


CONFIG = -DREMOVE_EXP


src = src/kernels.cpp src/units.cpp src/utils.cpp src/slm.cpp src/pupil.cpp src/scheduling.cpp
obj = $(patsubst src/%.cpp,build/%.o,$(src))


targets += benchmark regression


all: $(targets)


benchmark: build/benchmark.o $(obj)
	$(CXX) $(OPENMP) -o $@ $^ -L $(GOOGLE_BENCHMARK_LIB_DIR) -lbenchmark -lpthread


regression: build/regression.o $(obj)
	$(CXX) $(OPENMP) -o $@ $^


bench: benchmark
	./$^


generator/pupil: build/pupil.gen.o build/slm.o build/utils.o
	$(CXX) -o $@ $^


generator/scheduling: build/scheduling.gen.o build/slm.o build/utils.o
	$(CXX) -o $@ $^


output.bin: porting
	./$^ $@


report: output.bin
	$(PYTHON) python/compare_with_serial.py $^


build/%.o: src/%.cpp include/config.hpp
	$(CXX) -c $(WARNINGS) $(INCLUDE) $(OPT) $(OPENMP) $(CONFIG) -o $@ $<


build/benchmark.o: src/benchmark.cpp
	$(CXX) -c $(WARNINGS) $(INCLUDE) -I $(GOOGLE_BENCHMARK_INC_DIR) $(OPT) -o $@ $<


src/pupil.cpp: generator/pupil
	$< > $@


src/scheduling.cpp: generator/scheduling
	$< > $@


.PHONY clean:
clean:
	rm -f $(obj) $(targets) output.bin src/pupil.cpp src/scheduling.cpp
