#include "slm.hpp"
#include "utils.hpp"
#include <random>
#include <cmath>
#include <complex>
#include <cassert>


// @TODO: see correct linspace implementation
double linspace(double inf, double sup, int n, int i) {
	return inf + static_cast<double>(i) * (sup - inf) / static_cast<double>(n);
}


SLM::SLM(int width_, int height_, double wavelength_um_, double pixel_size_um_, double focal_length_mm_) : par(width_, height_, focal_length_mm_, pixel_size_um_, wavelength_um_), phase_buffer(width_ * height_), texture_buffer(width_ * height_) {
}


void SLM::write_on_file(std::ofstream &out) {
	write_vector_on_file(phase_buffer, par.width, par.height, out);
}


void SLM::rs_kernel(int n, const Point3D spots[], double pists[], double phase[], const SLMParameters *par, Performance *perf) {
	/* Before we start:
	 *  - the first implementation will be very inefficient and explicit, also there will be many memory allocations
	 *  - also I will use C++ features (vector, complex, exp)
	 * Let's go!
	 */


	(void) perf;
	const int    &WIDTH        = par->width;
	const int    &HEIGHT       = par->height;
	const double &FOCAL_LENGTH = par->focal_length_mm;
	const double &WAVELENGTH   = par->wavelength_um;
	const std::complex<double> IOTA(0.0, 1.0);


	for (int j = 0; j < HEIGHT; ++j) {
		for (int i = 0; i < WIDTH; ++i) {
			double x = linspace(-1.0, 1.0, WIDTH,  i);
			double y = linspace(-1.0, 1.0, HEIGHT, j);

			if (x*x + y*y < 1.0) {
				std::complex<double> total_field(0.0, 0.0);
				x = x * FOCAL_LENGTH * static_cast<double>(WIDTH) / 2.0;
				y = y * FOCAL_LENGTH * static_cast<double>(WIDTH) / 2.0;

				for (int ispot = 0; ispot < n; ++ispot) {
					const double p_phase = 2.0 * M_PI / (WAVELENGTH * FOCAL_LENGTH * 1000.0) * (spots[ispot].x * x + spots[ispot].y * y) + M_PI * spots[ispot].z / (WAVELENGTH * FOCAL_LENGTH * FOCAL_LENGTH * 1e6) * (x * x + y * y);

					total_field += std::exp(IOTA * (p_phase + pists[ispot]));
				}

				phase[j * WIDTH + i] = std::arg(total_field);
			}
		}
	}
}


void SLM::gs_kernel(int n, const Point3D spots[], double pists[], double phase[], const SLMParameters *par, Performance *perf, int iterations) {
	(void) perf;
	const int    &WIDTH        = par->width;
	const int    &HEIGHT       = par->height;
	const double &FOCAL_LENGTH = par->focal_length_mm;
	const double &WAVELENGTH   = par->wavelength_um;
	const std::complex<double> IOTA(0.0, 1.0);


	std::vector<std::complex<double>> spot_fields(n);


	for (int it = 0; it < iterations; ++it) {
		for (int j = 0; j < HEIGHT; ++j) {
			for (int i = 0; i < WIDTH; ++i) {
				double x = linspace(-1.0, 1.0, WIDTH,  i);
				double y = linspace(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					std::complex<double> total_field(0.0, 0.0);
					x = x * FOCAL_LENGTH * static_cast<double>(WIDTH) / 2.0;
					y = y * FOCAL_LENGTH * static_cast<double>(WIDTH) / 2.0;

					for (int ispot = 0; ispot < n; ++ispot) {
						// @OPT: replicating this computation is much better than storing the information? I have to check
						const double p_phase = 2.0 * M_PI / (WAVELENGTH * FOCAL_LENGTH * 1000.0) * (spots[ispot].x * x + spots[ispot].y * y) + M_PI * spots[ispot].z / (WAVELENGTH * FOCAL_LENGTH * FOCAL_LENGTH * 1e6) * (x * x + y * y);

						total_field += std::exp(IOTA * (p_phase + pists[ispot]));
					}

					const double total_phase = std::arg(total_field);
					phase[j * WIDTH + i] = total_phase;

					for (int ispot = 0; ispot < n; ++ispot) {
						// @OPT: we could cache the column of p_phase data
						const double p_phase = 2.0 * M_PI / (WAVELENGTH * FOCAL_LENGTH * 1000.0) * (spots[ispot].x * x + spots[ispot].y * y) + M_PI * spots[ispot].z / (WAVELENGTH * FOCAL_LENGTH * FOCAL_LENGTH * 1e6) * (x * x + y * y);

						spot_fields[ispot] += std::exp(IOTA * (total_phase + p_phase));
					}

					for (int ispot = 0; ispot < n; ++ispot) {
						pists[ispot] = std::arg(spot_fields[ispot]);
					}
				}
			}
		}
	}
}


void SLM::rs(const std::vector<Point3D> &spots, std::vector<double> &pists, bool measure) {
	rs_kernel(spots.size(), spots.data(), pists.data(), phase_buffer.data(), &par, measure ? &perf : NULL);
}


void SLM::gs(const std::vector<Point3D> &spots, std::vector<double> &pists, int iterations, bool measure) {
	gs_kernel(spots.size(), spots.data(), pists.data(), phase_buffer.data(), &par, measure ? &perf : NULL, iterations);
}
