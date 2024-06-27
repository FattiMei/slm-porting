CXX        = g++ -fPIE
SYCLCC     = /home/mmei/programs/acpp/bin/acpp --acpp-targets="cuda:sm_80" --acpp-platform=cuda --acpp-gpu-arch=sm_80 -fPIE
WARNINGS   = -Wall -Wextra -Wpedantic
INCLUDE    = -I ./include
OPT        = -O2 -march=native -ftree-vectorize -ffast-math
OPENMP     = -fopenmp


# requires installation of https://github.com/google/benchmark
GOOGLE_BENCHMARK_INC_DIR = /home/mmei/benchmark/include
GOOGLE_BENCHMARK_LIB_DIR = /home/mmei/benchmark/build/src


CONFIG = -DREMOVE_EXP


src = src/kernels.cpp src/units.cpp src/utils.cpp src/slm.cpp src/pupil.cpp src/scheduling.cpp
obj = $(patsubst src/%.cpp,build/%.o,$(src))


targets += benchmark benchmark_sycl regression porting


all: $(targets)


benchmark: build/benchmark.o $(obj)
	$(CXX) $(OPENMP) -o $@ $^ -L $(GOOGLE_BENCHMARK_LIB_DIR) -lbenchmark -lpthread


benchmark_sycl: build/benchmark.sycl.o build/kernels.sycl.o build/units.o build/utils.o build/pupil.o
	$(SYCLCC) -O3 -o $@ $^ -L $(GOOGLE_BENCHMARK_LIB_DIR) -lbenchmark -lpthread


regression: build/regression.o $(obj)
	$(CXX) $(OPENMP) -o $@ $^


porting: build/main.o $(obj)
	$(CXX) $(OPENMP) -o $@ $^


bench: benchmark
	./benchmark --benchmark_time_unit=ms
	./benchmark --benchmark_filter="^(rs_static_scheduling|rs_pupil_indices|rs_static_index_bounds|rs_computed_index_bounds)" --benchmark_time_unit=ms
	./benchmark --benchmark_filter=".*scheduling" --benchmark_time_unit=ms
	./benchmark --benchmark_filter=".*branch.*" --benchmark_time_unit=ms
	./benchmark --benchmark_filter="^(rs_pupil_indices|rs_simd)" --benchmark_time_unit=ms


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


build/%.sycl.o: src/%.sycl.cpp
	$(SYCLCC) -c $(INCLUDE) -O3 -o $@ $^


build/benchmark.sycl.o: src/benchmark.sycl.cpp
	$(SYCLCC) -c $(WARNINGS) $(INCLUDE) -I $(GOOGLE_BENCHMARK_INC_DIR) -O3 -o $@ $<


src/pupil.cpp: generator/pupil
	$< > $@


src/scheduling.cpp: generator/scheduling
	$< > $@


.PHONY clean:
clean:
	rm -f build/*.o $(targets) output.bin src/pupil.cpp src/scheduling.cpp
