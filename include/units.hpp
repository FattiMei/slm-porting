#ifndef __UNITS_HPP__
#define __UNITS_HPP__


// implementazione con le classi
class Length {
	public:
		enum units {
			meters,
			millimeters,
			micrometers,
			nanometers
		};

		Length(double x, units u) {
			double exponent;

			switch (u) {
				case meters     : exponent = 1e+0; break;
				case millimeters: exponent = 1e-3; break;
				case micrometers: exponent = 1e-6; break;
				case nanometers : exponent = 1e-9; break;
			}

			value = x * exponent;
		}

		double as_meters() {
			return value;
		}

		double as_millimeters() {
			return value * 1e3;
		}

		double as_micrometers() {
			return value * 1e6;
		}

		double as_nanometers() {
			return value * 1e9;
		}


	private:
		double value;
};


// mi piacerebbe chiamare il costruttore con questa sintassi
// auto x = Length::Meters(10.0);
//
// ci sono problemi di precisione?


// implementazione con i template?
// cosa cambia tra enum e enum class?
enum units {
	meters = 0,
	millimeters,
	micrometers,
	nanometers
};


// secondo me si può fare di meglio, tipo dei pattern matching o semplicemente un array da indicizzare con la enum
double exponent(enum units u) {
	switch (u) {
		case units::meters     : return 1e+0;
		case units::millimeters: return 1e-3;
		case units::micrometers: return 1e-6;
		case units::nanometers : return 1e-9;
	}

	return 0.0;
}


// si potrebbe avere anche la sicurezza che questi calcoli siano fatti il più possibile a compile time?
template <enum units U>
class LengthTemplated {
	public:
		LengthTemplated(double value_) : value(value_) {};

		// forse queste sono duplicate, posso fare diversamente
		double as_meters() {
			return value * exponent(U) / exponent(units::meters);
		}

		double as_millimeters() {
			return value * exponent(U) / exponent(units::millimeters);
		}

		double as_micrometers() {
			return value * exponent(U) / exponent(units::micrometers);
		}

		double as_nanometers() {
			return value * exponent(U) / exponent(units::nanometers);
		}

		double as(enum units requested) {
			return value * exponent(U) / exponent(requested);
		}


	private:
		const double value;
};


#endif
