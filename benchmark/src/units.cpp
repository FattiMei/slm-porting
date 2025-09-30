#include "units.hpp"
#include <cmath>


Length::Length(const double value_, const Unit unit_) : value(value_), unit(unit_) {}


double Length::as(const Unit target) const {
	return value * std::pow(10.0, static_cast<int>(unit) - static_cast<int>(target));
}
