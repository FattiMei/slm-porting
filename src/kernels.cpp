#include <random>
#include <omp.h>
#include "kernels.hpp"
#include "config.hpp"


// https://stackoverflow.com/questions/2683588/what-is-the-fastest-way-to-compute-sin-and-cos-together
#ifdef REMOVE_EXP
#define CEXP(x) std::complex<double>(std::cos(x), std::sin(x))
#else
#define CEXP(x) std::exp(std::complex<double>(0.0, 1.0) * (x))
#endif


#define LINSPACE(inf, sup, n, i) ((inf) + ((sup) - (inf)) * static_cast<double>(i) / static_cast<double>((n) - 1))
#define COMPUTE_P_PHASE(w, f, spot, pup_x, pup_y) ((2.0 * M_PI / ((w) * (f) * 1000.0)) * ((spot.x) * (pup_x) + (spot.y) * (pup_y)) + (M_PI * (spot.z) / ((w) * (f) * (f) * 1e6)) * ((pup_x) * (pup_x) + (pup_y) * (pup_y)))


#define WIDTH        (par->width)
#define HEIGHT       (par->height)
#define FOCAL_LENGTH (par->focal_length_mm)
#define PIXEL_SIZE   (par->pixel_size_um)
#define WAVELENGTH   (par->wavelength_um)


void compute_spot_field_module(const int n, const std::complex<double> spot_fields[], const int pupil_point_count, double ints[]) {
	for (int i = 0; i < n; ++i) {
		ints[i] = std::abs(spot_fields[i] / static_cast<double>(pupil_point_count));
	}
}


void update_weights(int n, const double ints[], double weights[]) {
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


void rs_kernel_static_scheduling(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	SLM::Parameters*	par
) {
	#pragma omp parallel for num_threads(OMP_NUM_THREADS)
	for (int j = 0; j < HEIGHT; ++j) {
		for (int i = 0; i < WIDTH; ++i) {
			double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
			double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

			if (x*x + y*y < 1.0) {
				std::complex<double> total_field(0.0, 0.0);
				x = x * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;
				y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

				// @ASSESS, @HARD: unroll this loop to give vectorization a chance?
				for (int ispot = 0; ispot < n; ++ispot) {
					const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

					total_field += CEXP(p_phase + pists[ispot]);
				}

				phase[j * WIDTH + i] = std::arg(total_field);
			}
		}
	}
}


void rs_kernel_dynamic_scheduling(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	SLM::Parameters*	par
) {
	// dynamic scheduling compensate the fact that some iterations have more points in the pupil
	#pragma omp parallel for schedule (dynamic) num_threads(OMP_NUM_THREADS)
	for (int j = 0; j < HEIGHT; ++j) {
		for (int i = 0; i < WIDTH; ++i) {
			double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
			double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

			if (x*x + y*y < 1.0) {
				std::complex<double> total_field(0.0, 0.0);
				x = x * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;
				y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

				for (int ispot = 0; ispot < n; ++ispot) {
					const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

					total_field += CEXP(p_phase + pists[ispot]);
				}

				phase[j * WIDTH + i] = std::arg(total_field);
			}
		}
	}
}


// https://stackoverflow.com/questions/49723192/openmp-custom-scheduling
void rs_kernel_custom_scheduling(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	SLM::Parameters*	par
) {
	#pragma omp parallel num_threads(OMP_NUM_THREADS)
	{
		extern const int thread_schedule_bounds[];
		const int thread = omp_get_thread_num();

		for (int j = thread_schedule_bounds[thread]; j < thread_schedule_bounds[thread + 1]; ++j) {
			for (int i = 0; i < WIDTH; ++i) {
				double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
				double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					// @ASSESS, @HARD: unroll this loop to give vectorization a chance?
					for (int ispot = 0; ispot < n; ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += CEXP(p_phase + pists[ispot]);
					}

					phase[j * WIDTH + i] = std::arg(total_field);
				}
			}
		}
	}
}


void rs_kernel_branchless(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	SLM::Parameters*	par
) {
	#pragma omp parallel for num_threads(OMP_NUM_THREADS)
	for (int j = 0; j < HEIGHT; ++j) {
		for (int i = 0; i < WIDTH; ++i) {
			double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
			double y = LINSPACE(-1.0, 1.0, HEIGHT, j);
			std::complex<double> total_field(0.0, 0.0);

			const bool is_inside_pupil = (x*x + y*y < 1.0);

			x = x * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;
			y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

			for (int ispot = 0; ispot < n; ++ispot) {
				const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

				total_field += CEXP(p_phase + pists[ispot]);
			}

			phase[j * WIDTH + i] = is_inside_pupil * std::arg(total_field);
		}
	}
}


// @TODO: compare the assembly code of rs_kernel_branchless and rs_kernel_branch_delay_slot
void rs_kernel_branch_delay_slot(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	SLM::Parameters*	par
) {
	#pragma omp parallel for num_threads(OMP_NUM_THREADS)
	for (int j = 0; j < HEIGHT; ++j) {
		for (int i = 0; i < WIDTH; ++i) {
			double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
			double y = LINSPACE(-1.0, 1.0, HEIGHT, j);
			std::complex<double> total_field(0.0, 0.0);

			const double norm = std::floor(x*x + y*y);

			x = x * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;
			y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

			for (int ispot = 0; ispot < n; ++ispot) {
				const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

				total_field += CEXP(p_phase + pists[ispot]);
			}

			if (norm < 1.0) {
				phase[j * WIDTH + i] = std::arg(total_field);
			}
		}
	}
}


void rs_kernel_cache_constants(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	SLM::Parameters*	par
) {
	const double radius = PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;

	#pragma omp parallel for num_threads(OMP_NUM_THREADS)
	for (int j = 0; j < HEIGHT; ++j) {
		for (int i = 0; i < WIDTH; ++i) {
			double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
			double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

			if (x*x + y*y < 1.0) {
				std::complex<double> total_field(0.0, 0.0);
				x = x * radius;
				y = y * radius;

				// @ASSESS, @HARD: unroll this loop to give vectorization a chance?
				for (int ispot = 0; ispot < n; ++ispot) {
					const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

					total_field += CEXP(p_phase + pists[ispot]);
				}

				phase[j * WIDTH + i] = std::arg(total_field);
			}
		}
	}
}


// compares with rs_kernel_static_scheduling
// @TODO: add aggressive optimizations about linspace calculations
// @TODO: apply this scheme to all kernels
void rs_kernel_math_cache(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	SLM::Parameters*	par
) {
	#pragma omp parallel num_threads(OMP_NUM_THREADS)
	{
		const double C1 = 2.0 * M_PI / (WAVELENGTH * FOCAL_LENGTH * 1000.0);
		const double C2 = M_PI / (WAVELENGTH * FOCAL_LENGTH * FOCAL_LENGTH * 1e6);
		const double radius = PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

		#pragma omp for
		for (int j = 0; j < HEIGHT; ++j) {
			const double y = LINSPACE(-1.0, 1.0, HEIGHT, j) * radius;
			const double cut = radius*radius - y*y;

			for (int i = 0; i < WIDTH; ++i) {
				const double x = LINSPACE(-1.0, 1.0, WIDTH, i) * radius;

				if (x*x < cut) {
					std::complex<double> total_field(0.0, 0.0);

					for (int ispot = 0; ispot < n; ++ispot) {
						const double p_phase = C1 * (spots[ispot].x * x + spots[ispot].y * y) + C2 * spots[ispot].z * (x*x + y*y);

						total_field += CEXP(p_phase + pists[ispot]);
					}

					phase[j * WIDTH + i] = std::arg(total_field);
				}
			}
		}
	}
}


void rs_kernel_pupil_indices_simd(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	int			pupil_count,
	const	int			pupil_indices[],
	const	SLM::Parameters*	par
) {
	const int chunk_size  = SIMD_LANE_SIZE;
	const int chunk_count = pupil_count / chunk_size;
	const int remainder   = pupil_count - chunk_size * chunk_count;

	#pragma omp parallel for schedule(static)
	for (int chunk = 0; chunk < chunk_count; ++chunk) {
		// we hope to put this memory into vector registers
		double x[chunk_size];
		double y[chunk_size];
		std::complex<double> total_field[chunk_size];

		for (int i = 0; i < chunk_size; ++i) {
			const int index = pupil_indices[chunk * chunk_size + i];
			const int row = index % WIDTH;
			const int col = index / WIDTH;

			x[i] = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  row) * static_cast<double>(WIDTH)  / 2.0;
			y[i] = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, col) * static_cast<double>(HEIGHT) / 2.0;

			total_field[i] = std::complex<double>(0.0, 0.0);
		}

		for (int ispot = 0; ispot < n; ++ispot) {
			const Point3D cached_spot = spots[ispot];
			const double  cached_pist = pists[ispot];

			for (int i = 0; i < chunk_size; ++i) {
				const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, cached_spot, x[i], y[i]);

				total_field[i] += CEXP(p_phase + cached_pist);
			}
		}

		for (int i = 0; i < chunk_size; ++i) {
			const int index = pupil_indices[chunk * chunk_size + i];

			phase[index] = std::arg(total_field[i]);
		}
	}

	// deal with the remaining elements left from the last chunk
	double x[chunk_size];
	double y[chunk_size];
	std::complex<double> total_field[chunk_size];

	for (int i = 0; i < remainder; ++i) {
		const int index = pupil_indices[chunk_count * chunk_size + i];
		const int row = index % WIDTH;
		const int col = index / WIDTH;

		x[i] = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  row) * static_cast<double>(WIDTH)  / 2.0;
		y[i] = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, col) * static_cast<double>(HEIGHT) / 2.0;

		total_field[i] = std::complex<double>(0.0, 0.0);
	}

	for (int ispot = 0; ispot < n; ++ispot) {
		const Point3D cached_spot = spots[ispot];
		const double  cached_pist = pists[ispot];

		for (int i = 0; i < remainder; ++i) {
			const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, cached_spot, x[i], y[i]);

			total_field[i] += CEXP(p_phase + cached_pist);
		}
	}

	for (int i = 0; i < remainder; ++i) {
		const int index = pupil_indices[chunk_count * chunk_size + i];

		phase[index] = std::arg(total_field[i]);
	}
}


// replicate the same memory access patterns, but use simple arithmetic operations
// it's the best performance we could expect from this system
void rs_kernel_upper_bound(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	SLM::Parameters*	par
) {
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
	for (int j = 0; j < HEIGHT; ++j) {
		for (int i = 0; i < WIDTH; ++i) {
			double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
			double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

			if (x*x + y*y < 1.0) {
				std::complex<double> total_field(0.0, 0.0);
				x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
				y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

				for (int ispot = 0; ispot < n; ++ispot) {
					const double useless = x * spots[ispot].x + y * spots[ispot].y + spots[ispot].z;

					total_field += std::complex<double>(pists[ispot], useless);
				}

				phase[j * WIDTH + i] = std::abs(total_field);
			}
		}
	}
}


void rs_kernel_pupil_indices(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	int			pupil_count,
	const	int			pupil_indices[],
	const	SLM::Parameters*	par
) {
#pragma omp parallel for num_threads(OMP_NUM_THREADS)
	for (int index = 0; index < pupil_count; ++index) {
		const int i = pupil_indices[index] % WIDTH;
		const int j = pupil_indices[index] / WIDTH;

		const double x = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  i) * static_cast<double>(WIDTH)  / 2.0;
		const double y = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, j) * static_cast<double>(HEIGHT) / 2.0;

		std::complex<double> total_field(0.0, 0.0);

		for (int ispot = 0; ispot < n; ++ispot) {
			const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

			total_field += CEXP(p_phase + pists[ispot]);
		}

		phase[j * WIDTH + i] = std::arg(total_field);
	}
}


void rs_kernel_pupil_indices_dual(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	int			pupil_count,
	const	int			pupil_indices[],
	const	SLM::Parameters*	par
) {
	for (int index = 0; index < pupil_count; ++index) {
		const int i = pupil_indices[index] % WIDTH;
		const int j = pupil_indices[index] / WIDTH;

		const double x = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  i) * static_cast<double>(WIDTH)  / 2.0;
		const double y = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, j) * static_cast<double>(HEIGHT) / 2.0;

		std::complex<double> total_field(0.0, 0.0);

		// https://www.reddit.com/r/cpp_questions/comments/rd12o9/openmp_reduction_not_working_with_complex_vector/
		#pragma omp declare reduction(+: std::complex<double>: omp_out += omp_in) initializer(omp_priv = omp_orig)
		#pragma omp parallel for reduction(+: total_field) num_threads(OMP_NUM_THREADS)
		for (int ispot = 0; ispot < n; ++ispot) {
			const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

			total_field += CEXP(p_phase + pists[ispot]);
		}

		phase[j * WIDTH + i] = std::arg(total_field);
	}
}


void rs_kernel_static_index_bounds(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	std::pair<int,int>	pupil_index_bounds[],
	const	SLM::Parameters*	par
) {
	#pragma omp parallel num_threads(OMP_NUM_THREADS)
	{
		extern const int thread_schedule_bounds[];
		const int thread = omp_get_thread_num();

		for (int j = thread_schedule_bounds[thread]; j < thread_schedule_bounds[thread + 1]; ++j) {
			for (int i = pupil_index_bounds[j].first; i < pupil_index_bounds[j].second; ++i) {
				const double x = LINSPACE(-1.0, 1.0, WIDTH,  i) * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;
				const double y = LINSPACE(-1.0, 1.0, HEIGHT, j) * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

				std::complex<double> total_field(0.0, 0.0);

				for (int ispot = 0; ispot < n; ++ispot) {
					const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

					total_field += CEXP(p_phase + pists[ispot]);
				}

				phase[j * WIDTH + i] = std::arg(total_field);
			}
		}
	}
}


void rs_kernel_computed_index_bounds(
	const	int			n,
	const	Point3D			spots[],
	const	double			pists[],
		double			phase[],
	const	SLM::Parameters*	par
) {
	#pragma omp parallel num_threads(OMP_NUM_THREADS)
	{
		extern const int thread_schedule_bounds[];
		const int thread = omp_get_thread_num();

		for (int j = thread_schedule_bounds[thread]; j < thread_schedule_bounds[thread + 1]; ++j) {
			double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

			const int upper = static_cast<int>(std::ceil(0.5 * (1.0 + std::sqrt(1.0 - y*y)) * static_cast<double>(WIDTH - 1)));
			const int lower = WIDTH - upper;

			y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

			for (int i = lower; i < upper; ++i) {
				const double x = LINSPACE(-1.0, 1.0, WIDTH,  i) * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;

				std::complex<double> total_field(0.0, 0.0);

				for (int ispot = 0; ispot < n; ++ispot) {
					const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

					total_field += CEXP(p_phase + pists[ispot]);
				}

				phase[j * WIDTH + i] = std::arg(total_field);
			}
		}
	}
}


void gs_kernel_naive(
	const	int			n,
	const	Point3D			spots[],
		double			pists[],
		std::complex<double>	spot_fields[],
		double			phase[],
	const	SLM::Parameters*	par,
	const	int			iterations
) {
	for (int it = 0; it < iterations; ++it) {
		for (int ispot = 0; ispot < n; ++ispot) {
			spot_fields[ispot] = std::complex<double>(0.0, 0.0);
		}

		for (int j = 0; j < HEIGHT; ++j) {
			for (int i = 0; i < WIDTH; ++i) {
				double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
				double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (int ispot = 0; ispot < n; ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += CEXP(p_phase + pists[ispot]);
					}

					const double total_phase = std::arg(total_field);
					phase[j * WIDTH + i] = total_phase;

					// @TODO: extract parallelism by separating this computation as GPU implementation
					for (int ispot = 0; ispot < n; ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						spot_fields[ispot] += CEXP(total_phase - p_phase);
					}
				}
			}
		}

		for (int ispot = 0; ispot < n; ++ispot) {
			pists[ispot] = std::arg(spot_fields[ispot]);
		}
	}
}


void gs_kernel_cached(
	const	int			n,
	const	Point3D			spots[],
		double			pists[],
		double			p_phase_cache[],
		std::complex<double>	spot_fields[],
		double			phase[],
	const	SLM::Parameters*	par,
	const	int			iterations
) {
	for (int it = 0; it < iterations; ++it) {
		for (int ispot = 0; ispot < n; ++ispot) {
			spot_fields[ispot] = std::complex<double>(0.0, 0.0);
		}

		for (int j = 0; j < HEIGHT; ++j) {
			for (int i = 0; i < WIDTH; ++i) {
				double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
				double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (int ispot = 0; ispot < n; ++ispot) {
						p_phase_cache[ispot] = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += CEXP(p_phase_cache[ispot] + pists[ispot]);
					}

					const double total_phase = std::arg(total_field);
					phase[j * WIDTH + i] = total_phase;

					for (int ispot = 0; ispot < n; ++ispot) {
						spot_fields[ispot] += CEXP(total_phase - p_phase_cache[ispot]);
					}
				}
			}
		}

		for (int ispot = 0; ispot < n; ++ispot) {
			pists[ispot] = std::arg(spot_fields[ispot]);
		}
	}
}


void gs_kernel_reordered(
	const	int			n,
	const	Point3D			spots[],
		double			pists[],
		std::complex<double>	spot_fields[],
		double			phase[],
	const	SLM::Parameters*	par,
	const	int			iterations
) {
	// this initialization could be remove because the spot_fields is zeroed at the end of this kernel
	for (int ispot = 0; ispot < n; ++ispot) {
		spot_fields[ispot] = std::complex<double>(0.0, 0.0);
	}


	for (int it = 0; it < iterations; ++it) {
		for (int j = 0; j < HEIGHT; ++j) {
			for (int i = 0; i < WIDTH; ++i) {
				double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
				double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (int ispot = 0; ispot < n; ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += CEXP(p_phase + pists[ispot]);
					}

					const double total_phase = std::arg(total_field);
					phase[j * WIDTH + i] = total_phase;

					for (int ispot = 0; ispot < n; ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						spot_fields[ispot] += CEXP(total_phase - p_phase);
					}
				}
			}
		}

		for (int ispot = 0; ispot < n; ++ispot) {
			pists      [ispot] = std::arg(spot_fields[ispot]);
			spot_fields[ispot] = std::complex<double>(0.0, 0.0);
		}
	}
}


void wgs_kernel(
	int                  n,
	const Point3D        spots[],
	double               pists[],
	double               pists_tmp_buffer[],
	std::complex<double> spot_fields[],
	double               ints[],
	double               weights[],
	double               phase[],
	const SLM::Parameters* par,
	Performance*         perf,
	int                  iterations
) {
	int pupil_point_count = 0;
	(void) perf;


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
				double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
				double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					++pupil_point_count;
					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (int ispot = 0; ispot < n; ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += weights[ispot] * CEXP(p_phase + pists[ispot]);
					}

					const double total_phase = std::arg(total_field);
					phase[j * WIDTH + i] = total_phase;

					for (int ispot = 0; ispot < n; ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						spot_fields[ispot] += CEXP(total_phase - p_phase);
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
void csgs_kernel(
	int                  n,
	const Point3D        spots[],
	double               pists[],
	double               pists_tmp_buffer[],
	std::complex<double> spot_fields[],
	double               phase[],
	const SLM::Parameters* par,
	Performance*         perf,
	int                  iterations,
	double               compression,
	int                  seed
) {
	(void) perf;
	std::default_random_engine gen(seed);
	std::uniform_real_distribution<double> uniform(0.0, 1.0);


	for (int it = 0; it < iterations; ++it) {
		for (int ispot = 0; ispot < n; ++ispot) {
			spot_fields[ispot] = std::complex<double>(0.0, 0.0);
		}

		for (int j = 0; j < HEIGHT; ++j) {
			for (int i = 0; i < WIDTH; ++i) {
				double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
				double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					if (it < (iterations - 1) and uniform(gen) > compression) {
						continue;
					}

					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (int ispot = 0; ispot < n; ++ispot) {
						// @OPT: replicating this computation is much better than storing the information? I have to check
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += CEXP(p_phase + pists[ispot]);
					}

					const double total_phase = std::arg(total_field);
					phase[j * WIDTH + i] = total_phase;

					for (int ispot = 0; ispot < n; ++ispot) {
						// @OPT: we could cache the column of p_phase data
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						spot_fields[ispot] += CEXP(total_phase - p_phase);
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


void wcsgs_kernel(
	int                  n,
	const Point3D        spots[],
	double               pists[],
	double               pists_tmp_buffer[],
	std::complex<double> spot_fields[],
	double               ints[],
	double               weights[],
	double               phase[],
	const SLM::Parameters* par,
	Performance*         perf,
	int                  iterations,
	double               compression,
	int                  seed
) {
	(void) perf;
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
				double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
				double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

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
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += CEXP(p_phase + pists[ispot]);
					}

					const double total_phase = std::arg(total_field);
					phase[j * WIDTH + i] = total_phase;

					for (int ispot = 0; ispot < n; ++ispot) {
						// @OPT: we could cache the column of p_phase data
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						spot_fields[ispot] += CEXP(total_phase - p_phase);
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
			double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
			double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

			if (x*x + y*y < 1.0) {
				std::complex<double> total_field(0.0, 0.0);
				x = x * PIXEL_SIZE * static_cast<double>(WIDTH) / 2.0;
				y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

				for (int ispot = 0; ispot < n; ++ispot) {
					const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

					total_field += weights[ispot] * CEXP(p_phase + pists[ispot]);
				}

				const double total_phase = std::arg(total_field);
				phase[j * WIDTH + i] = total_phase;
			}
		}
	}
}
