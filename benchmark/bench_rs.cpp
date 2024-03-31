#include <benchmark/benchmark.h>
#include "slm.hpp"
#include "utils.hpp"


const int WIDTH   = 512;
const int HEIGHT  = 512;
const int NPOINTS = 100;
const Length WAVELENGTH  (0.488, Unit::Micrometers);
const Length PITCH       ( 15.0, Unit::Micrometers);
const Length FOCAL_LENGTH( 20.0, Unit::Millimeters);


static void bench_rs(benchmark::State& state) {
	SLM slm(WIDTH, HEIGHT, WAVELENGTH, PITCH, FOCAL_LENGTH);

	std::vector<double> pists(NPOINTS);
	generate_random_vector(pists, 0.0, 2.0 * M_PI, 1);
	std::vector<Point3D> spots;
	generate_grid_spots(10, 10.0, spots);

	for (auto _ : state) {
		slm.rs(spots, pists);
	}
}


BENCHMARK(bench_rs);
BENCHMARK_MAIN();
