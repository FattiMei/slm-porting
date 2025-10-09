#ifndef __UNITS_H__
#define __UNITS_H__


enum class Unit {
	METERS = 1,
	MILLIMETERS = -3,
	MICROMETERS = -6,
	NANOMETERS = -9
};


class Length {
	public:
		Length(double mantissa, Unit unit);
		double convert_to(Unit unit) const;

	private:
		const double value;
};


#endif // __UNITS_H__
