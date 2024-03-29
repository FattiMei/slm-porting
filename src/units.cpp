#include "units.hpp"
#include <cmath>


Length::Length(const double value_, const enum Unit unit_) : value(value_), unit(unit_) {}


double Length::as(const enum Unit target) const {
	return value * std::pow(10.0, unit - target);
}
