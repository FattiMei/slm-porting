#include "slm.hpp"
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
	out << par.width << " " << par.height << std::endl;
	out.write(reinterpret_cast<const char *>(phase_buffer.data()), par.width * par.height * sizeof(double));
}


void SLM::rs_kernel(int n, const Point3D spots[], double phase[], const SLMParameters *par, Performance *perf, int seed) {
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


	// random pist generation
	std::vector<double> pists(n);

	{
		std::default_random_engine gen(seed);
		std::uniform_real_distribution<double> uniform(0.0, 2.0 * M_PI);

		for (int i = 0; i < n; ++i) {
			pists[i] = uniform(gen);
		}
	}


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


void SLM::rs_kernel_inefficient(int n, const Point3D spots[], double phase[], const SLMParameters *par, Performance *perf, int seed) {
	(void) perf;
	const int    &WIDTH        = par->width;
	const int    &HEIGHT       = par->height;
	const double &FOCAL_LENGTH = par->focal_length_mm;
	const double &WAVELENGTH   = par->wavelength_um;


	// random pist generation
	std::vector<double> pists(n);

	{
		std::default_random_engine gen(seed);
		std::uniform_real_distribution<double> uniform(0.0, 2.0 * M_PI);

		for (int i = 0; i < n; ++i) {
			pists[i] = uniform(gen);
		}
	}

	// pupil coords generation
	std::vector<double> pupil_x;
	std::vector<double> pupil_y;
	std::vector<int> pupil_index;

	// @OPT: make the process of selecting pupil points inside the loop below
	{
		// careful in how you iterate over the two indices
		// @POSSIBLE_BUG: look at the iteration space!
		for (int i = 0; i < WIDTH; ++i) {
			for (int j = 0; j < WIDTH; ++j) {
				const double x_cand = linspace(-1.0, 1.0, WIDTH,  i);
				const double y_cand = linspace(-1.0, 1.0, HEIGHT, j);

				if (x_cand*x_cand + y_cand*y_cand < 1.0) {
					pupil_index.push_back(i * WIDTH + j);

					pupil_x.push_back(x_cand * FOCAL_LENGTH * static_cast<double>(WIDTH) / 2.0);
					pupil_y.push_back(y_cand * FOCAL_LENGTH * static_cast<double>(HEIGHT) / 2.0);
				}
			}
		}
	}


	// thinking in the reverse order seems helpful
	const std::complex<double> IOTA(0.0, 1.0);

	for (size_t i = 0; i < pupil_index.size(); ++i) {
		std::complex<double> total_field(0.0, 0.0);

		for (int j = 0; j < n; ++j) {
			const double p_phase = 2.0 * M_PI / (WAVELENGTH * FOCAL_LENGTH * 1000.0) * (spots[j].x * pupil_x[i] + spots[j].y * pupil_y[i]) + M_PI * spots[j].z / (WAVELENGTH * FOCAL_LENGTH * FOCAL_LENGTH * 1e6) * (pupil_x[i] * pupil_x[i] + pupil_y[i] * pupil_y[i]);

			total_field += std::exp(IOTA * (p_phase + pists[j]));
		}

		phase[pupil_index[i]] = std::arg(total_field / static_cast<double>(pupil_index.size()));
	}
}


void SLM::rs(const std::vector<Point3D> &spots, int seed, bool measure) {
	rs_kernel(spots.size(), spots.data(), phase_buffer.data(), &par, measure ? &perf : NULL, seed);
}
