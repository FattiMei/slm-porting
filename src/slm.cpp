#include "slm.hpp"
#include "kernels.hpp"
#include "utils.hpp"


std::vector<int>
generate_pupil_indices(const SLM::Parameters &parameters) {
	std::vector<int> result;

	const int &WIDTH  = parameters.width;
	const int &HEIGHT = parameters.height;

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


std::vector<std::pair<int,int>>
generate_pupil_index_bounds(const SLM::Parameters &parameters) {
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


SLM::PupilIterator::PupilIterator(const SLM::Parameters &parameters_) {
	// pupil_coordinates = generate_pupil_coordinates(parameters_);
	current_point = pupil_coordinates.begin();
}


Point2D SLM::PupilIterator::operator*() {
	return *current_point;
}


SLM::PupilIterator& SLM::PupilIterator::operator++() {
	++current_point;
	empty = (current_point == pupil_coordinates.end());

	return *this;
}


bool SLM::PupilIterator::operator!=(SLM::PupilIterator &other) {
	return empty and other.empty;
}
