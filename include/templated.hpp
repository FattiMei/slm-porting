#ifndef __TEMPLATED_HPP__
#define __TEMPLATED_HPP__


#include "utils.hpp"


using namespace std::complex_literals;


inline double compute_p_phase(const double wavelength, const double focal_length, const Point3D spot, const double x, const double y) {
	const double c1 = 2.0 * M_PI / (wavelength * focal_length * 1000.0);
	const double c2 = M_PI * spot.z / (wavelength * focal_length * focal_length * 1e6);

	return c1 * (spot.x * x + spot.y * y) + c2 * (x * x + y * y);
}


// template definition and declaration must be in the same file
template <const int n, const int WIDTH, const int HEIGHT>
void rs_kernel_templated(
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	SLM::Parameters*	par
) {
	const double &FOCAL_LENGTH = par->focal_length_mm;
	const double &PIXEL_SIZE   = par->pixel_size_um;
	const double &WAVELENGTH   = par->wavelength_um;


	for (int j = 0; j < HEIGHT; ++j) {
		for (int i = 0; i < WIDTH; ++i) {
			double x = linspace(-1.0, 1.0, WIDTH,  i);
			double y = linspace(-1.0, 1.0, HEIGHT, j);

			if (x*x + y*y < 1.0) {
				std::complex<double> total_field(0.0, 0.0);
				x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
				y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

				for (int ispot = 0; ispot < n; ++ispot) {
					const double p_phase = compute_p_phase(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

					total_field += std::exp(1.0i * (p_phase + pists[ispot]));
				}

				phase[j * WIDTH + i] = std::arg(total_field);
			}
		}
	}
}


#endif
