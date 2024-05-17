#include <iostream>
#include <chrono>
#include "slm.hpp"
#include "utils.hpp"
#include "kernels.hpp"


#define NSAMPLES 5
#define SEED     1


template <typename Run>
void benchmark(int samples, const std::string &name, Run run_cmd) {
	std::cout << name << std::endl;

	for (int i = 0; i < samples; ++i) {
		const auto start_time = std::chrono::high_resolution_clock::now();

		run_cmd();

		const auto end_time = std::chrono::high_resolution_clock::now();
		const auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

		std::cout << delta << std::endl;
	}
}


int main() {
	SLM::Parameters parameters(
		512,
		512,
		Length(20.0, Unit::Millimeters),
		Length(15.0, Unit::Micrometers),
		Length(488.0, Unit::Nanometers)
	);

	const auto spots = generate_grid_spots(10, 10.0);
	const int  N	 = spots.size();
	auto pists = generate_random_vector(N, 0.0, 2.0 * M_PI, 1);

	// specific for rs testing
	const auto pupil_coordinates  = generate_pupil_coordinates(parameters);
	const auto pupil_indices      = generate_pupil_indices(parameters);
	const auto pupil_index_bounds = generate_pupil_index_bounds(parameters);

	// specific for gs testing
	std::vector<std::complex<double>> spot_fields(N);
	std::vector<double> p_phase_cache(N);

	std::vector<double> phase(parameters.width * parameters.height);


	const auto rs_naive_invocation = [&] {
		rs_kernel_naive(N, spots.data(), pists.data(), phase.data(), &parameters);
	};

	const auto rs_manual_invocation = [&] {
		rs_kernel_manual(N, spots.data(), pists.data(), phase.data(), &parameters);
	};

	const auto rs_upper_bound_invocation = [&] {
		rs_upper_bound(N, spots.data(), pists.data(), phase.data(), &parameters);
	};

	const auto rs_pupil_coordinates_invocation = [&] {
		rs_kernel_pupil_coordinates(N, spots.data(), pists.data(), phase.data(), pupil_coordinates.size(), pupil_coordinates.data(), &parameters);
	};

	const auto rs_pupil_indices_invocation = [&] {
		rs_kernel_pupil_indices(N, spots.data(), pists.data(), phase.data(), pupil_indices.size(), pupil_indices.data(), &parameters);
	};

	const auto rs_pupil_index_bounds_invocation = [&] {
		rs_kernel_pupil_index_bounds(N, spots.data(), pists.data(), phase.data(), pupil_index_bounds.data(), &parameters);
	};

	const auto rs_static_index_bounds_invocation = [&] {
		rs_kernel_static_index_bounds(N, spots.data(), pists.data(), phase.data(), &parameters);
	};

	// i know that these invocations will modify pists data, it shouldn't be a problem since here we are testing only performance and the algorithms are static
	const auto gs_naive_invocation = [&] {
		gs_kernel_naive(N, spots.data(), pists.data(), spot_fields.data(), phase.data(), &parameters, 30);
	};

	const auto gs_cached_invocation = [&] {
		gs_kernel_cached(N, spots.data(), pists.data(), p_phase_cache.data(), spot_fields.data(), phase.data(), &parameters, 30);
	};

	const auto gs_reordered_invocation = [&] {
		gs_kernel_reordered(N, spots.data(), pists.data(), spot_fields.data(), phase.data(), &parameters, 30);
	};


	benchmark(NSAMPLES, "rs upper bound",                   rs_upper_bound_invocation);
	benchmark(NSAMPLES, "rs naive",				rs_naive_invocation);
	benchmark(NSAMPLES, "rs all manual/inlined functions",  rs_manual_invocation);
	benchmark(NSAMPLES, "rs precomputed pupil coordinates", rs_pupil_coordinates_invocation);
	benchmark(NSAMPLES, "rs precomputed pupil indices",	rs_pupil_indices_invocation);
	benchmark(NSAMPLES, "rs precomputed index bounds",	rs_pupil_index_bounds_invocation);
	benchmark(NSAMPLES, "rs runtime computed index bounds", rs_static_index_bounds_invocation);

#if 0
	benchmark(1, "gs naive",	gs_naive_invocation);
	benchmark(1, "gs cached", 	gs_cached_invocation);
	benchmark(1, "gs reordered",	gs_reordered_invocation);
#endif


	return 0;
}
