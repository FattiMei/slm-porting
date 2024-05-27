CXX        = g++-13
SYCLCC     = acpp
WARNINGS   = -Wall -Wextra -Wpedantic
INCLUDE    = -I ./include
OPT        = -O2 -march=native -ftree-vectorize -ffast-math
OPENMP     = -fopenmp
PYTHON     = python3.8


# requires installation of https://github.com/google/benchmark
GOOGLE_BENCHMARK_INC_DIR = /home/matteo/hpc/programs/benchmark/include
GOOGLE_BENCHMARK_LIB_DIR = /home/matteo/hpc/programs/benchmark/build/src


CONFIG = -DREMOVE_EXP


src = src/kernels.cpp src/units.cpp src/utils.cpp src/slm.cpp src/pupil.cpp src/scheduling.cpp
obj = $(patsubst src/%.cpp,build/%.o,$(src))


targets += benchmark regression porting test


all: $(targets)


benchmark: build/benchmark.o $(obj)
	$(CXX) $(OPENMP) -o $@ $^ -L $(GOOGLE_BENCHMARK_LIB_DIR) -lbenchmark -lpthread


regression: build/regression.o $(obj)
	$(CXX) $(OPENMP) -o $@ $^


porting: build/main.o $(obj)
	$(CXX) $(OPENMP) -o $@ $^


test: src/sycl.cpp src/units.cpp src/utils.cpp src/pupil.cpp
	$(SYCLCC) $(INCLUDE) -O3 -o $@ $^


bench: benchmark
	./$^


generator/pupil: build/pupil.gen.o build/slm.o build/utils.o
	$(CXX) -o $@ $^


generator/scheduling: build/scheduling.gen.o build/slm.o build/utils.o
	$(CXX) -o $@ $^


output.bin: test
	./$^ $@


report: output.bin
	$(PYTHON) python/compare_with_serial.py $^


asm: src/kernels.cpp include/config.hpp
	$(CXX) -S $(WARNINGS) $(INCLUDE) $(OPT) $(OPENMP) $(CONFIG) -o kernels.asm $<


build/%.o: src/%.cpp include/config.hpp
	$(CXX) -c $(WARNINGS) $(INCLUDE) $(OPT) $(OPENMP) $(CONFIG) -o $@ $<


build/benchmark.o: src/benchmark.cpp
	$(CXX) -c $(WARNINGS) $(INCLUDE) -I $(GOOGLE_BENCHMARK_INC_DIR) $(OPT) -o $@ $<


build/sycl.o: src/sycl.cpp
	$(SYCLCC) $(INCLUDE) -O3 -c -o $@ $^


src/pupil.cpp: generator/pupil
	$< > $@


src/scheduling.cpp: generator/scheduling
	$< > $@


.PHONY clean:
clean:
	rm -f build/*.o $(targets) output.bin src/pupil.cpp src/scheduling.cpp
