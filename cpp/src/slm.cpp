#include "slm.h"
#include "units.h"
#include <cmath>


SLM::SLM(Length focal_,
	 Length pixel_size_,
	 Length wavelength_,
	 int resolution_) : focal(focal_),
	                    pixel_size(pixel_size_),
	                    wavelength(wavelength_),
	                    resolution(resolution_),
	                    C1(2.0 * M_PI / (wavelength.convert_to(Unit::MICROMETERS) * focal.convert_to(Unit::MICROMETERS))),
	                    C2(M_PI / (wavelength.convert_to(Unit::MICROMETERS) * std::pow(focal.convert_to(Unit::MICROMETERS), 2.0))) {

	// pupil point generation
	for (int i = 0; i < resolution; ++i) {
		for (int j = 0; j < resolution; ++j) {
			const double x = -1.0 + 2.0 * j / (resolution-1.0);
			const double y = -1.0 + 2.0 * i / (resolution-1.0);

			if (x*x + y*y < 1.0) {
				pupil_idx.push_back(i*resolution + j);
				xx.push_back(x * pixel_size.convert_to(Unit::MICROMETERS) * resolution / 2.0);
				yy.push_back(y * pixel_size.convert_to(Unit::MICROMETERS) * resolution / 2.0);
			}
		}
	}
}


SLM get_standard_slm() {
	Length focal(20.0, Unit::MILLIMETERS);
	Length pixel_size(15.0, Unit::MICROMETERS);
	Length wavelength(488.0, Unit::NANOMETERS);
	int resolution = 512;

	return SLM(focal, pixel_size, wavelength, resolution);
}
