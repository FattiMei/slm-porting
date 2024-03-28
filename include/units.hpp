#ifndef __UNITS_HPP__
#define __UNITS_HPP__


// cosa cambia tra enum e enum class?
enum Unit {
	meters      =  0,
	millimeters = -3,
	micrometers = -6,
	nanometers  = -9
};


class Length {
	public:
		Length(const double value_, const enum Unit unit_);

		double as(const enum Unit requested) const;


	private:
		const double value;
		const enum Unit unit;
};


#endif
