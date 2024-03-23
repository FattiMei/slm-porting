#include "slm.hpp"
#include <random>
#include <cmath>
#include <complex>
#include <cassert>


// @TODO: see correct linspace implementation
double linspace(double inf, double sup, int n, int i) {
	return inf + static_cast<double>(i) * (sup - inf) / static_cast<double>(n);
}


void generate_random_pistons(std::vector<double> &pists, int seed) {
	std::default_random_engine gen(seed);
	std::uniform_real_distribution<double> uniform(0.0, 2.0 * M_PI);

	for (auto &p : pists) {
		p = uniform(gen);
	}
}


struct Point2D {
	double x;
	double y;

	Point2D(double x_, double y_) : x(x_), y(y_) {};
};


void filter_pupil_points(std::vector<Point2D> &pupil_points, std::vector<int> &pupil_index, double focal_length, int width, int height) {
	for (int j = 0; j < height; ++j) {
		for (int i = 0; i < width; ++i) {
			double x = linspace(-1.0, 1.0, width,  i);
			double y = linspace(-1.0, 1.0, height, j);

			if (x*x + y*y < 1.0) {
				pupil_index.push_back(j * width + i);

				x = x * focal_length * static_cast<double>(width) / 2.0;
				y = y * focal_length * static_cast<double>(height) / 2.0;

				pupil_points.emplace_back(x, y);
			}
		}
	}
}

SLM::SLM(int width_, int height_, double wavelength_um_, double pixel_size_um_, double focal_length_mm_) : par(width_, height_, focal_length_mm_, pixel_size_um_, wavelength_um_), phase_buffer(width_ * height_), texture_buffer(width_ * height_) {
}


void SLM::write_on_file(std::ofstream &out) {
	out << par.width << " " << par.height << std::endl;
	out.write(reinterpret_cast<const char *>(phase_buffer.data()), par.width * par.height * sizeof(double));
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


void SLM::rs_kernel_inefficient(int n, const Point3D spots[], double pists[], double phase[], const SLMParameters *par, Performance *perf) {
	(void) perf;
	const int    &WIDTH        = par->width;
	const int    &HEIGHT       = par->height;
	const double &FOCAL_LENGTH = par->focal_length_mm;
	const double &WAVELENGTH   = par->wavelength_um;
	const std::complex<double> IOTA(0.0, 1.0);


	std::vector<Point2D> pupil_points;
	std::vector<int> pupil_index;
	filter_pupil_points(pupil_points, pupil_index, FOCAL_LENGTH, WIDTH, HEIGHT);


	for (size_t i = 0; i < pupil_index.size(); ++i) {
		std::complex<double> total_field(0.0, 0.0);

		for (int j = 0; j < n; ++j) {
			const double p_phase = 2.0 * M_PI / (WAVELENGTH * FOCAL_LENGTH * 1000.0) * (spots[j].x * pupil_points[i].x + spots[j].y * pupil_points[i].y) + M_PI * spots[j].z / (WAVELENGTH * FOCAL_LENGTH * FOCAL_LENGTH * 1e6) * (pupil_points[i].x * pupil_points[i].x + pupil_points[i].y * pupil_points[i].y);

			total_field += std::exp(IOTA * (p_phase + pists[j]));
		}

		phase[pupil_index[i]] = std::arg(total_field / static_cast<double>(pupil_index.size()));
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


void SLM::rs(const std::vector<Point3D> &spots, int seed, bool measure) {
	std::vector<double> pists(spots.size());
	generate_random_pistons(pists, seed);

	rs_kernel_inefficient(spots.size(), spots.data(), pists.data(), phase_buffer.data(), &par, measure ? &perf : NULL);
}


void SLM::gs(const std::vector<Point3D> &spots, int iterations, int seed, bool measure) {
	std::vector<double> pists(spots.size());
	generate_random_pistons(pists, seed);

	gs_kernel(spots.size(), spots.data(), pists.data(), phase_buffer.data(), &par, measure ? &perf : NULL, iterations);
}
