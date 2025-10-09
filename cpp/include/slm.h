#ifndef __SLM_H__
#define __SLM_H__


#include "units.h"
#include <vector>


struct SLM {
	SLM(Length focal,
	    Length pixel_size,
	    Length wavelength,
	    int resolution);

	const Length focal;
	const Length pixel_size;
	const Length wavelength;
	const int resolution;

	const double C1;
	const double C2;

	std::vector<double> xx;
	std::vector<double> yy;
	std::vector<int> pupil_idx;
};


SLM get_standard_slm();


#endif //  __SLM_H__
