#ifndef __UNITS_HPP__
#define __UNITS_HPP__


enum class Unit {
	Meters      =  0,
	Millimeters = -3,
	Micrometers = -6,
	Nanometers  = -9
};


class Length {
	public:
		Length(const double value_, const Unit unit_);

		double as(const Unit target) const;


	private:
		const double value;
		const Unit unit;
};


#endif
