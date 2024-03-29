#include "slm.hpp"
#include "utils.hpp"
#include <random>
#include <cmath>
#include <complex>
#include <cassert>


// @ASSESS: putting parameters as const helps the compiler?
inline double compute_p_phase(const double wavelength, const double focal_length, const Point3D spot, const double x, const double y) {
	const double c1 = 2.0 * M_PI / (wavelength * focal_length * 1000.0);
	const double c2 = M_PI * spot.z / (wavelength * focal_length * focal_length * 1e6);

	return c1 * (spot.x * x + spot.y * y) + c2 * (x * x + y * y);
}


inline void compute_spot_field_module(int n, const std::complex<double> spot_fields[], int pupil_point_count, double ints[]) {
	for (int i = 0; i < n; ++i) {
		ints[i] = std::abs(spot_fields[i] / static_cast<double>(pupil_point_count));
	}
}


inline void update_weights(int n, const double ints[], double weights[]) {
	// @OPT: I will profile this function and fuse all the operations in one pass
	double total_ints_sum = 0.0;
	double total_weight_sum = 0.0;

	for (int i = 0; i < n; ++i) {
		total_ints_sum += ints[i];
	}

	const double mean = total_ints_sum / static_cast<double>(n);

	for (int i = 0; i < n; ++i) {
		weights[i]       *= (mean / ints[i]);
		total_weight_sum += weights[i];
	}

	for (int i = 0; i < n; ++i) {
		weights[i] /= total_weight_sum;
	}
}


SLM::SLM(int width, int height, const Length &wavelength, const Length &pixel_size, const Length &focal_length) : 
	par(width, height, focal_length, pixel_size, wavelength),
	phase_buffer(width * height),
	texture_buffer(width * height) {
}


void SLM::write_on_file(std::ofstream &out) {
	write_vector_on_file(phase_buffer, par.width, par.height, out);
}


// @TODO, @FORMATTING: align better variable names
void SLM::rs_kernel(
	int                  n,
	const Point3D        spots[],
	const double         pists[],
	double               phase[],
	const SLMParameters* par,
	Performance*         perf
) {
	(void) perf;
	const int    &WIDTH        = par->width;
	const int    &HEIGHT       = par->height;
	const double &FOCAL_LENGTH = par->focal_length_mm;
	const double &PIXEL_SIZE   = par->pixel_size_um;
	const double &WAVELENGTH   = par->wavelength_um;
	const std::complex<double> IOTA(0.0, 1.0);


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

					total_field += std::exp(IOTA * (p_phase + pists[ispot]));
				}

				phase[j * WIDTH + i] = std::arg(total_field);
			}
		}
	}
}


void SLM::gs_kernel(
	int                  n,
	const Point3D        spots[],
	double               pists[],
	double               pists_tmp_buffer[],
	std::complex<double> spot_fields[],
	double               phase[],
	const SLMParameters* par,
	Performance*         perf,
	int                  iterations
) {
	(void) perf;
	const int    &WIDTH        = par->width;
	const int    &HEIGHT       = par->height;
	const double &FOCAL_LENGTH = par->focal_length_mm;
	const double &PIXEL_SIZE   = par->pixel_size_um;
	const double &WAVELENGTH   = par->wavelength_um;
	const std::complex<double> IOTA(0.0, 1.0);


	for (int it = 0; it < iterations; ++it) {
		for (int ispot = 0; ispot < n; ++ispot) {
			spot_fields[ispot] = std::complex<double>(0.0, 0.0);
		}

		for (int j = 0; j < HEIGHT; ++j) {
			for (int i = 0; i < WIDTH; ++i) {
				double x = linspace(-1.0, 1.0, WIDTH,  i);
				double y = linspace(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (int ispot = 0; ispot < n; ++ispot) {
						// @OPT: replicating this computation is much better than storing the information? I have to check
						const double p_phase = compute_p_phase(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += std::exp(IOTA * (p_phase + pists[ispot]));
					}

					const double total_phase = std::arg(total_field);
					phase[j * WIDTH + i] = total_phase;

					for (int ispot = 0; ispot < n; ++ispot) {
						// @OPT: we could cache the column of p_phase data
						const double p_phase = compute_p_phase(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						spot_fields[ispot] += std::exp(IOTA * (total_phase - p_phase));
					}

					for (int ispot = 0; ispot < n; ++ispot) {
						pists_tmp_buffer[ispot] = std::arg(spot_fields[ispot]);
					}
				}
			}
		}

		std::swap(pists, pists_tmp_buffer);
	}
}


void SLM::wgs_kernel(
	int                  n,
	const Point3D        spots[],
	double               pists[],
	double               pists_tmp_buffer[],
	std::complex<double> spot_fields[],
	double               ints[],
	double               weights[],
	double               phase[],
	const SLMParameters* par,
	Performance*         perf,
	int                  iterations
) {
	(void) perf;
	const int    &WIDTH        = par->width;
	const int    &HEIGHT       = par->height;
	const double &FOCAL_LENGTH = par->focal_length_mm;
	const double &PIXEL_SIZE   = par->pixel_size_um;
	const double &WAVELENGTH   = par->wavelength_um;
	const std::complex<double> IOTA(0.0, 1.0);


	int pupil_point_count = 0;


	for (int i = 0; i < n; ++i) {
		weights[i] = 1.0 / static_cast<double>(n);
	}

	// @OPT: transform this for loop to skip the weights update and spot_field at the last iteration
	for (int it = 0; it < iterations; ++it) {
		pupil_point_count = 0;

		for (int ispot = 0; ispot < n; ++ispot) {
			spot_fields[ispot] = std::complex<double>(0.0, 0.0);
		}

		for (int j = 0; j < HEIGHT; ++j) {
			for (int i = 0; i < WIDTH; ++i) {
				double x = linspace(-1.0, 1.0, WIDTH,  i);
				double y = linspace(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					++pupil_point_count;
					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (int ispot = 0; ispot < n; ++ispot) {
						const double p_phase = compute_p_phase(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += weights[ispot] * std::exp(IOTA * (p_phase + pists[ispot]));
					}

					const double total_phase = std::arg(total_field);
					phase[j * WIDTH + i] = total_phase;

					for (int ispot = 0; ispot < n; ++ispot) {
						const double p_phase = compute_p_phase(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						spot_fields[ispot] += std::exp(IOTA * (total_phase - p_phase));
					}

					for (int ispot = 0; ispot < n; ++ispot) {
						pists_tmp_buffer[ispot] = std::arg(spot_fields[ispot]);
					}
				}
			}
		}

		compute_spot_field_module(n, spot_fields, pupil_point_count, ints);
		update_weights(n, ints, weights);
		std::swap(pists, pists_tmp_buffer);
	}
}


// @DESIGN, @TYPE: can we encode in a type that compression has to be in [0,1]?
void SLM::csgs_kernel(
	int                  n,
	const Point3D        spots[],
	double               pists[],
	double               pists_tmp_buffer[],
	std::complex<double> spot_fields[],
	double               phase[],
	const SLMParameters* par,
	Performance*         perf,
	int                  iterations,
	double               compression,
	int                  seed
) {
	(void) perf;
	const int    &WIDTH        = par->width;
	const int    &HEIGHT       = par->height;
	const double &FOCAL_LENGTH = par->focal_length_mm;
	const double &PIXEL_SIZE   = par->pixel_size_um;
	const double &WAVELENGTH   = par->wavelength_um;
	const std::complex<double> IOTA(0.0, 1.0);


	std::default_random_engine gen(seed);
	std::uniform_real_distribution<double> uniform(0.0, 1.0);


	for (int it = 0; it < iterations; ++it) {
		for (int ispot = 0; ispot < n; ++ispot) {
			spot_fields[ispot] = std::complex<double>(0.0, 0.0);
		}

		for (int j = 0; j < HEIGHT; ++j) {
			for (int i = 0; i < WIDTH; ++i) {
				double x = linspace(-1.0, 1.0, WIDTH,  i);
				double y = linspace(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					if (it < (iterations - 1) and uniform(gen) > compression) {
						continue;
					}

					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (int ispot = 0; ispot < n; ++ispot) {
						// @OPT: replicating this computation is much better than storing the information? I have to check
						const double p_phase = compute_p_phase(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += std::exp(IOTA * (p_phase + pists[ispot]));
					}

					const double total_phase = std::arg(total_field);
					phase[j * WIDTH + i] = total_phase;

					for (int ispot = 0; ispot < n; ++ispot) {
						// @OPT: we could cache the column of p_phase data
						const double p_phase = compute_p_phase(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						spot_fields[ispot] += std::exp(IOTA * (total_phase - p_phase));
					}

					for (int ispot = 0; ispot < n; ++ispot) {
						pists_tmp_buffer[ispot] = std::arg(spot_fields[ispot]);
					}
				}
			}
		}

		std::swap(pists, pists_tmp_buffer);
	}
}


void SLM::wcsgs_kernel(
	int                  n,
	const Point3D        spots[],
	double               pists[],
	double               pists_tmp_buffer[],
	std::complex<double> spot_fields[],
	double               ints[],
	double               weights[],
	double               phase[],
	const SLMParameters* par,
	Performance*         perf,
	int                  iterations,
	double               compression,
	int                  seed
) {
	(void) perf;
	const int    &WIDTH        = par->width;
	const int    &HEIGHT       = par->height;
	const double &FOCAL_LENGTH = par->focal_length_mm;
	const double &PIXEL_SIZE   = par->pixel_size_um;
	const double &WAVELENGTH   = par->wavelength_um;
	const std::complex<double> IOTA(0.0, 1.0);


	int pupil_point_count = 0;


	std::default_random_engine gen(seed);
	std::uniform_real_distribution<double> uniform(0.0, 1.0);


	for (int i = 0; i < n; ++i) {
		weights[i] = 1.0 / static_cast<double>(n);
	}


	for (int it = 0; it < iterations; ++it) {
		pupil_point_count = 0;

		for (int ispot = 0; ispot < n; ++ispot) {
			spot_fields[ispot] = std::complex<double>(0.0, 0.0);
		}

		for (int j = 0; j < HEIGHT; ++j) {
			for (int i = 0; i < WIDTH; ++i) {
				double x = linspace(-1.0, 1.0, WIDTH,  i);
				double y = linspace(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					++pupil_point_count;

					if (it < (iterations - 1) and uniform(gen) > compression) {
						continue;
					}

					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (int ispot = 0; ispot < n; ++ispot) {
						// @OPT: replicating this computation is much better than storing the information? I have to check
						const double p_phase = compute_p_phase(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += std::exp(IOTA * (p_phase + pists[ispot]));
					}

					const double total_phase = std::arg(total_field);
					phase[j * WIDTH + i] = total_phase;

					for (int ispot = 0; ispot < n; ++ispot) {
						// @OPT: we could cache the column of p_phase data
						const double p_phase = compute_p_phase(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						spot_fields[ispot] += std::exp(IOTA * (total_phase - p_phase));
					}

					for (int ispot = 0; ispot < n; ++ispot) {
						pists_tmp_buffer[ispot] = std::arg(spot_fields[ispot]);
					}
				}
			}
		}

		std::swap(pists, pists_tmp_buffer);
	}


	compute_spot_field_module(n, spot_fields, pupil_point_count, ints);
	update_weights(n, ints, weights);


	// one last iteration of wgs
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

					total_field += weights[ispot] * std::exp(IOTA * (p_phase + pists[ispot]));
				}

				const double total_phase = std::arg(total_field);
				phase[j * WIDTH + i] = total_phase;
			}
		}
	}
}


void SLM::rs(const std::vector<Point3D> &spots, const std::vector<double> &pists, bool measure) {
	const int N = spots.size();

	rs_kernel(
		N,
		spots.data(),
		pists.data(),
		phase_buffer.data(),
		&par,
		measure ? &perf : NULL
	);
}


void SLM::gs(const std::vector<Point3D> &spots, const std::vector<double> &pists, int iterations, bool measure) {
	const int N = spots.size();

	std::vector<double>               pists_copy(pists);
	std::vector<double>               pists_tmp_buffer(N);
	std::vector<std::complex<double>> spot_fields(N);

	gs_kernel(
		N,
		spots.data(),
		pists_copy.data(),
		pists_tmp_buffer.data(),
		spot_fields.data(),
		phase_buffer.data(),
		&par,
		measure ? &perf : NULL,
		iterations
	);
}


void SLM::wgs(const std::vector<Point3D> &spots, const std::vector<double> &pists, int iterations, bool measure) {
	const int N = spots.size();

	std::vector<double>               pists_copy(pists);
	std::vector<double>               pists_tmp_buffer(N);
	std::vector<double>               ints(N);
	std::vector<double>               weights(N);
	std::vector<std::complex<double>> spot_fields(N);

	wgs_kernel(
		N,
		spots.data(),
		pists_copy.data(),
		pists_tmp_buffer.data(),
		spot_fields.data(),
		ints.data(),
		weights.data(),
		phase_buffer.data(),
		&par,
		measure ? &perf : NULL,
		iterations
	);
}


void SLM::csgs(const std::vector<Point3D> &spots, const std::vector<double> &pists, int iterations, double compression, int seed, bool measure) {
	const int N = spots.size();

	std::vector<double>               pists_copy(pists);
	std::vector<double>               pists_tmp_buffer(N);
	std::vector<std::complex<double>> spot_fields(N);

	csgs_kernel(
		N,
		spots.data(),
		pists_copy.data(),
		pists_tmp_buffer.data(),
		spot_fields.data(),
		phase_buffer.data(),
		&par,
		measure ? &perf : NULL,
		iterations,
		compression,
		seed
	);
}


void SLM::wcsgs(const std::vector<Point3D> &spots, const std::vector<double> &pists, int iterations, double compression, int seed, bool measure) {
	const int N = spots.size();

	std::vector<double>               pists_copy(pists);
	std::vector<double>               pists_tmp_buffer(N);
	std::vector<double>               ints(N);
	std::vector<double>               weights(N);
	std::vector<std::complex<double>> spot_fields(N);

	wcsgs_kernel(
		N,
		spots.data(),
		pists_copy.data(),
		pists_tmp_buffer.data(),
		spot_fields.data(),
		ints.data(),
		weights.data(),
		phase_buffer.data(),
		&par,
		measure ? &perf : NULL,
		iterations,
		compression,
		seed
	);
}
