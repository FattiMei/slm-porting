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

	const auto reference   = generate_pupil_index_bounds(parameters);
	const auto alternative =  compute_pupil_index_bounds(parameters);


	assert(reference.size() == alternative.size());


	for (size_t i = 0; i < reference.size(); ++i) {
		std::cout
			<< "("
			<< reference[i].first
			<< ", "
			<< reference[i].second
			<< ") "
			<< "("
			<< alternative[i].first
			<< ", "
			<< alternative[i].second
			<< ") "
			<< std::endl;
	}


	return 0;
}
