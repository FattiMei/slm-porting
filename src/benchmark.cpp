#include <iostream>
#include <chrono>
#include "slm.hpp"
#include "utils.hpp"
#include "kernels.hpp"


#define NSAMPLES 1000
#define SEED     1


// we don't care about correctness in this experiment
int main() {
	SLM::Parameters parameters(
		512,
		512,
		Length(20.0, Unit::Millimeters),
		Length(15.0, Unit::Micrometers),
		Length(488.0, Unit::Nanometers)
	);

	const auto spots              = generate_grid_spots(10, 10.0);
	const auto pists              = generate_random_vector(spots.size(), 0.0, 2.0 * M_PI, 1);
	const auto pupil_indices      = generate_pupil_indices(parameters);
	const auto pupil_index_bounds = generate_pupil_index_bounds(parameters);
	std::vector<double> phase(parameters.width * parameters.height);


	std::cout << "# rs kernel naive (ms)" << std::endl;
	for (int i = 0; i < NSAMPLES; ++i) {
		const auto start_time = std::chrono::high_resolution_clock::now();

		rs_kernel_naive(spots.size(), spots.data(), pists.data(), phase.data(), &parameters);

		const auto end_time = std::chrono::high_resolution_clock::now();
		const auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

		std::cout << delta << std::endl;
	}


	std::cout << "# rs kernel precomputed pupil indices (ms)" << std::endl;
	for (int i = 0; i < NSAMPLES; ++i) {
		const auto start_time = std::chrono::high_resolution_clock::now();

		rs_kernel_pupil_indices(spots.size(), spots.data(), pists.data(), phase.data(), pupil_indices.size(), pupil_indices.data(), &parameters);

		const auto end_time = std::chrono::high_resolution_clock::now();
		const auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

		std::cout << delta << std::endl;
	}


	std::cout << "# rs kernel precomputed pupil index bounds (ms)" << std::endl;
	for (int i = 0; i < NSAMPLES; ++i) {
		const auto start_time = std::chrono::high_resolution_clock::now();

		rs_kernel_pupil_index_bounds(spots.size(), spots.data(), pists.data(), phase.data(), pupil_index_bounds.data(), &parameters);

		const auto end_time = std::chrono::high_resolution_clock::now();
		const auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

		std::cout << delta << std::endl;
	}

	return 0;
}
