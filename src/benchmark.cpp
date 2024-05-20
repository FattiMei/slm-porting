#include <benchmark/benchmark.h>
#include "slm.hpp"
#include "utils.hpp"
#include "kernels.hpp"


constexpr int n      = 100;
constexpr int width  = 512;
constexpr int height = 512;


const SLM::Parameters parameters(width, height, Length(20.0, Unit::Millimeters), Length(15.0, Unit::Micrometers), Length(488.0, Unit::Nanometers));
Point3D spots[n];
double  pists[n];
double  phase[width * height];

extern const int pupil_count;
extern const int pupil_indices[];



static void rs_static_scheduling(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_static_scheduling(n, spots, pists, phase, &parameters);
	}
}


static void rs_dynamic_scheduling(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_dynamic_scheduling(n, spots, pists, phase, &parameters);
	}
}


static void rs_branchless(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_branchless(n, spots, pists, phase, &parameters);
	}
}


static void rs_branch_delay_slot(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_branchless(n, spots, pists, phase, &parameters);
	}
}


static void rs_pupil_indices(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_pupil_indices(n, spots, pists, phase, pupil_count, pupil_indices, &parameters);
	}
}


// @TODO: set the time unit in the command invocation
BENCHMARK(rs_static_scheduling)->Unit(benchmark::kMillisecond);
BENCHMARK(rs_dynamic_scheduling)->Unit(benchmark::kMillisecond);
BENCHMARK(rs_pupil_indices)->Unit(benchmark::kMillisecond);
BENCHMARK(rs_branchless)->Unit(benchmark::kMillisecond);
BENCHMARK(rs_branch_delay_slot)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
