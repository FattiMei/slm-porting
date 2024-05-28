#include <benchmark/benchmark.h>
#include <vector>
#include "config.hpp"
#include "slm.hpp"
#include "utils.hpp"
#include "units.hpp"
#include <CL/sycl.hpp>
#include "kernels.sycl.hpp"


constexpr int n      = 100;
constexpr int width  = 512;
constexpr int height = 512;


const SLM::Parameters parameters(width, height, Length(20.0, Unit::Millimeters), Length(15.0, Unit::Micrometers), Length(488.0, Unit::Nanometers));
std::vector<Point3D> spots(n);
std::vector<double>  pists(n);
std::vector<double>  phase(width * height);

cl::sycl::queue q;



static void rs_sycl_naive(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists.data(), 0.0, 2.0 * M_PI, 1);
		rs_kernel_naive(q, spots, pists, phase, parameters);
		q.wait();
	}
}


static void rs_sycl_pupil(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists.data(), 0.0, 2.0 * M_PI, 1);
		rs_kernel_pupil(q, spots, pists, phase, parameters);
		q.wait();
	}
}


// @TODO: set the time unit in the command invocation
BENCHMARK(rs_sycl_naive)->Unit(benchmark::kMillisecond);
BENCHMARK(rs_sycl_pupil)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
