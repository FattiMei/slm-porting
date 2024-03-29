#ifndef __UNITS_HPP__
#define __UNITS_HPP__


// cosa cambia tra enum e enum class?
enum Unit {
	Meters      =  0,
	Millimeters = -3,
	Micrometers = -6,
	Nanometers  = -9
};


class Length {
	public:
		Length(const double value_, const enum Unit unit_);

		double as(const enum Unit target) const;


	private:
		const double value;
		const enum Unit unit;
};


#endif
