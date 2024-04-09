#include "slm.hpp"
#include "kernels.hpp"
#include "utils.hpp"


// @DESIGN: I should copy the parameters structure or store the reference?
// @DESIGN: what can I do for allocating std::vectors? resize?
// @DESIGN (for reference problems) use std::move or unique or shared pointer
SLM::Wrapper::Wrapper(const SLM::Parameters parameters_, const std::vector<Point3D> &spots_) :
	parameters(parameters_),
	spots     (spots_),
	n         (spots_.size()),
	phase     (parameters_.width * parameters_.height)
{
	pists = generate_random_vector(n, 0.0, 2.0 * M_PI, 1);
}


void SLM::Wrapper::rs() {
	rs_kernel(n, spots.data(), pists.data(), phase.data(), &parameters, NULL);
}


std::vector<int> generate_pupil_indices(const SLM::Parameters &parameters) {
	const int &WIDTH  = parameters.width;
	const int &HEIGHT = parameters.height;
	std::vector<int> result;

	for (int j = 0; j < HEIGHT; ++j) {
		for (int i = 0; i < WIDTH; ++i) {
			const double x = linspace(-1.0, 1.0, WIDTH,  i);
			const double y = linspace(-1.0, 1.0, HEIGHT, j);

			if (x*x + y*y < 1.0) {
				result.push_back(j * WIDTH + i);
			}
		}
	}

	return result;
}


std::vector<std::pair<int,int>> generate_pupil_index_bounds(const SLM::Parameters &parameters) {
	// returns [lower, upper) bounds

	const int &WIDTH  = parameters.width;
	const int &HEIGHT = parameters.height;
	std::vector<std::pair<int,int>> result(HEIGHT);

	// is there a modern and readable C++ way to solve this problem?
	// if this approach is useful I will put more attention to it
	for (int j = 0; j < HEIGHT; ++j) {
		const double y = linspace(-1.0, 1.0, HEIGHT, j);
		int lower;
		int upper;

		for (lower = 0; lower < WIDTH; ++lower) {
			const double x = linspace(-1.0, 1.0, WIDTH, lower);

			if (x*x + y*y < 1.0) {
				break;
			}
		}

		for (upper = WIDTH - 1; upper >= 0; --upper) {
			const double x = linspace(-1.0, 1.0, WIDTH, upper);

			if (x*x + y*y < 1.0) {
				++upper;
				break;
			}
		}

		result[j] = {lower, upper};
	}

	return result;
}


std::vector<std::pair<int,int>> compute_pupil_index_bounds(const SLM::Parameters &parameters) {
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
