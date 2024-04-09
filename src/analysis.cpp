#include <iostream>
#include <cassert>
#include "slm.hpp"


int main() {
	const SLM::Parameters parameters(
		512,
		512,
		Length(20.0, Unit::Millimeters),
		Length(15.0, Unit::Micrometers),
		Length(488.0, Unit::Nanometers)
	);

	// @ADVICE: can we make the computation of such bounds at compile time? constexpr can be of use in this case
	const auto reference   = generate_pupil_index_bounds(parameters);
	const auto alternative =  compute_pupil_index_bounds(parameters);

	if (reference == alternative) {
		std::cout << "Good job" << std::endl;
	}
	else {
		std::cout << "Something is wrong" << std::endl;
	}

	return 0;
}
