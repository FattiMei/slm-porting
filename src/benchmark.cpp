#include <benchmark/benchmark.h>
#include "slm.hpp"
#include "utils.hpp"
#include "kernels.hpp"


const SLM::Parameters parameters(
	512,
	512,
	Length(20.0,  Unit::Millimeters),
	Length(15.0,  Unit::Micrometers),
	Length(488.0, Unit::Nanometers)
);


static void BM_rs_naive(benchmark::State &state) {
	const auto spots = generate_grid_spots(10, 10.0);
	const int  N	 = spots.size();
	auto pists = generate_random_vector(N, 0.0, 2.0 * M_PI, 1);
	std::vector<double> phase(parameters.width * parameters.height);

	for (auto _ : state) {
		rs_kernel_naive(N, spots.data(), pists.data(), phase.data(), &parameters);
	}
}


static void BM_rs_branchless(benchmark::State &state) {
	const auto spots = generate_grid_spots(10, 10.0);
	const int  N	 = spots.size();
	auto pists = generate_random_vector(N, 0.0, 2.0 * M_PI, 1);
	std::vector<double> phase(parameters.width * parameters.height);

	for (auto _ : state) {
		rs_kernel_branchless(N, spots.data(), pists.data(), phase.data(), &parameters);
	}
}


static void BM_rs_indices(benchmark::State &state) {
	const auto spots = generate_grid_spots(10, 10.0);
	const int  N	 = spots.size();
	auto pists = generate_random_vector(N, 0.0, 2.0 * M_PI, 1);
	std::vector<double> phase(parameters.width * parameters.height);

	auto pupil_indices = generate_pupil_indices(parameters);


	for (auto _ : state) {
		rs_kernel_pupil_indices(N, spots.data(), pists.data(), phase.data(), pupil_indices.size(), pupil_indices.data(), &parameters);
	}
}


// @TODO: set the time unit in the command invocation
BENCHMARK(BM_rs_branchless)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_rs_naive)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_rs_indices)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
