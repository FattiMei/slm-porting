#include "slm.hpp"
#include "kernels.hpp"
#include "utils.hpp"


std::vector<int>
generate_pupil_indices(const int resolution) {
	std::vector<int> result;

	for (int j = 0; j < resolution; ++j) {
		for (int i = 0; i < resolution; ++i) {
			const double x = linspace(-1.0, 1.0, resolution,  i);
			const double y = linspace(-1.0, 1.0, resolution, j);

			if (x*x + y*y < 1.0) {
				result.push_back(j * resolution + i);
			}
		}
	}

	return result;
}


std::vector<std::pair<int,int>>
generate_pupil_index_bounds(const int resolution) {
	std::vector<std::pair<int,int>> result(resolution);

	for (int j = 0; j < resolution; ++j) {
		const double y = linspace(-1.0, 1.0, resolution, j);
		int lower = 0;
		int upper = 0;

		for (lower = 0; lower < resolution / 2; ++lower) {
			const double x = linspace(-1.0, 1.0, resolution, lower);

			if (x*x + y*y < 1.0) {
				break;
			}
		}

		for (upper = resolution - 1; upper > resolution / 2; --upper) {
			const double x = linspace(-1.0, 1.0, resolution, upper);

			if (x*x + y*y < 1.0) {
				++upper;
				break;
			}
		}

		result[j] = {lower, upper};
	}

	return result;
}


std::vector<std::pair<int,int>>
compute_pupil_index_bounds(const SLM::Parameters &parameters) {
	// returns [lower, upper) bounds

	const int &WIDTH  = parameters.width;
	const int &HEIGHT = parameters.height;
	std::vector<std::pair<int,int>> result(HEIGHT);

	for (int j = 0; j < HEIGHT; ++j) {
		const double y = linspace(-1.0, 1.0, HEIGHT, j);
		const int upper = static_cast<int>(std::ceil(0.5 * (1.0 + std::sqrt(1.0 - y*y)) * static_cast<double>(WIDTH - 1)));
		const int lower = WIDTH - upper;

		result[j] = {lower, upper};
	}

	return result;
}
