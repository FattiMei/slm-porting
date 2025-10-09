#include "units.h"
#include <cmath>


Length::Length(const double mantissa, const Unit unit) : value(
	mantissa * std::pow(10.0, static_cast<int>(unit))) {}


double Length::convert_to(Unit unit) {
	return value * std::pow(10.0, -static_cast<int>(unit));
}
