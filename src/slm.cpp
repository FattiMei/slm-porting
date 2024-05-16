#include "slm.hpp"
#include "kernels.hpp"
#include "utils.hpp"


// @DESIGN: I should copy the parameters structure or store the reference?
// @DESIGN: what can I do for allocating std::vectors? resize?
// @DESIGN (for reference problems) use std::move or unique or shared pointer


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

	// @ADVICE: is there a modern and readable C++ way to solve this problem?
	for (int j = 0; j < HEIGHT; ++j) {
		const double y = linspace(-1.0, 1.0, HEIGHT, j);
		int lower;
		int upper;

		for (lower = 0; lower < WIDTH / 2; ++lower) {
			const double x = linspace(-1.0, 1.0, WIDTH, lower);

			if (x*x + y*y < 1.0) {
				break;
			}
		}

		for (upper = WIDTH - 1; upper > WIDTH / 2; --upper) {
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


SLM::PupilIterator::PupilIterator(SLM::Parameters &parameters_) : parameters(parameters_) {
	pupil_indices = generate_pupil_indices(parameters);
	current_index = pupil_indices.begin();
}


std::pair<double, double>* SLM::PupilIterator::operator*() {
	// this operation has many vulnerabilities: the PupilIterator class has to live longer than this returned pointer
	const int i = *current_index % parameters.width;
	const int j = *current_index / parameters.width;

	cache = std::make_pair(
		parameters.pixel_size_um * linspace(-1.0, 1.0, parameters.width,  i) * static_cast<double>(parameters.width)  / 2.0,
		parameters.pixel_size_um * linspace(-1.0, 1.0, parameters.height, j) * static_cast<double>(parameters.height) / 2.0
	);

	return &cache;
}


SLM::PupilIterator& SLM::PupilIterator::operator++() {
	++current_index;
	empty = (current_index == pupil_indices.end());

	return *this;
}


bool SLM::PupilIterator::operator!=(SLM::PupilIterator &other) {
	return empty and other.empty;
}
