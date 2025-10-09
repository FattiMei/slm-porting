#include "units.h"
#include <cmath>
#include <cassert>


bool are_close(double x, double y, double atol = 1e-10) {
	return std::abs(x - y) < atol;
}


int main() {
	Length wavelength = Length(488.0, Unit::NANOMETERS);

	assert(are_close(wavelength.convert_to(Unit::METERS), 488.0e-9));
	assert(are_close(wavelength.convert_to(Unit::MICROMETERS), 0.488));

	return 0;
}
