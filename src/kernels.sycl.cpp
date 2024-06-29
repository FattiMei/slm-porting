#include "kernels.sycl.hpp"


#define CEXP(x) std::complex<double>(std::cos(x), std::sin(x))


#define LINSPACE(inf, sup, n, i) ((inf) + ((sup) - (inf)) * static_cast<double>(i) / static_cast<double>((n) - 1))
#define COMPUTE_P_PHASE(w, f, spot, pup_x, pup_y) ((2.0 * M_PI / ((w) * (f) * 1000.0)) * ((spot.x) * (pup_x) + (spot.y) * (pup_y)) + (M_PI * (spot.z) / ((w) * (f) * (f) * 1e6)) * ((pup_x) * (pup_x) + (pup_y) * (pup_y)))


#define WIDTH        (par.width)
#define HEIGHT       (par.height)
#define FOCAL_LENGTH (par.focal_length_mm)
#define PIXEL_SIZE   (par.pixel_size_um)
#define WAVELENGTH   (par.wavelength_um)


extern const int pupil_count;
extern const int pupil_indices[];
static cl::sycl::buffer<int> buff_pupil(pupil_indices, pupil_count);


void rs_kernel_naive(queue &q, const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters par) {
	cl::sycl::range<2> work_items{static_cast<size_t>(WIDTH), static_cast<size_t>(HEIGHT)};

	q.submit([&](cl::sycl::handler& cgh) {
		cgh.parallel_for<class rs>(
			work_items,
			[=](cl::sycl::id<2> tid) {
				double x = LINSPACE(-1.0, 1.0, WIDTH,  tid[0]);
				double y = LINSPACE(-1.0, 1.0, HEIGHT, tid[1]);

				if (x*x + y*y < 1.0) {
					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (size_t ispot = 0; ispot < n; ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += CEXP(p_phase + pists[ispot]);
					}

					// std::arg is not working!
					phase[tid[1] * WIDTH + tid[0]] = std::atan2(total_field.imag(), total_field.real());
				}
			}
		);
	}).wait();
}


void rs_kernel_pupil(queue &q, const int n, const Point3D spots[], const double pists[], double phase[], const int pupil_count, const int pupil_indices[], const SLM::Parameters par) {
	cl::sycl::range<1> work_items{static_cast<size_t>(pupil_count)};

	q.submit([&](cl::sycl::handler& cgh) {
		cgh.parallel_for<class test>(
			work_items,
			[=](cl::sycl::id<1> tid) {
				const int index = pupil_indices[tid];
				const int i = index % WIDTH;
				const int j = index / WIDTH;

				const double x = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  i) * static_cast<double>(WIDTH)  / 2.0;
				const double y = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, j) * static_cast<double>(HEIGHT) / 2.0;

				std::complex<double> total_field(0.0, 0.0);

				for (size_t ispot = 0; ispot < n; ++ispot) {
					const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

					total_field += CEXP(p_phase + pists[ispot]);
				}

				// std::arg is not working!
				phase[index] = std::atan2(total_field.imag(), total_field.real());
			}
		);
	}).wait();
}


void rs_kernel_local(queue &q, const int n, const Point3D spots[], const double pists[], double phase[], const int pupil_count, const int pupil_indices[], const SLM::Parameters par) {
	cl::sycl::nd_range<1> work_items{static_cast<size_t>(pupil_count) * static_cast<size_t>(n), static_cast<size_t>(n)};

	q.submit([&](cl::sycl::handler& cgh) {
		cgh.parallel_for<class test_group_reductions>(
			work_items,
			[=](cl::sycl::nd_item<1> it) {
				auto g = it.get_group();

				const int index = pupil_indices[g.get_group_id()];
				const int i = index % WIDTH;
				const int j = index / WIDTH;

				const double x = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  i) * static_cast<double>(WIDTH)  / 2.0;
				const double y = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, j) * static_cast<double>(HEIGHT) / 2.0;

				const std::complex<double> partial = CEXP(COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[it.get_local_id()], x, y) + pists[it.get_local_id()]);

				const double total_field_real = cl::sycl::reduce_over_group(g, partial.real(), cl::sycl::plus<double>());
				const double total_field_imag = cl::sycl::reduce_over_group(g, partial.imag(), cl::sycl::plus<double>());

				if (g.leader()) {
					// std::arg is not working!
					phase[index] = std::atan2(total_field_imag, total_field_real);
				}
			}
		);
	}).wait();
}



void gs_kernel_naive(queue &q, const int n, const Point3D spots[], double pists[], double phase[], const SLM::Parameters par, const int iterations) {
	cl::sycl::range<2> work_items{static_cast<size_t>(WIDTH), static_cast<size_t>(HEIGHT)};
	cl::sycl::range<1> spot_items{static_cast<size_t>(n)};

	for (int it = 0; it < iterations; ++it) {
		// first compute all the total phases
		q.submit([&](cl::sycl::handler& cgh) {
			cgh.parallel_for<class rs>(
				work_items,
				[=](cl::sycl::id<2> tid) {
					double x = LINSPACE(-1.0, 1.0, WIDTH,  tid[0]);
					double y = LINSPACE(-1.0, 1.0, HEIGHT, tid[1]);

					if (x*x + y*y < 1.0) {
						std::complex<double> total_field(0.0, 0.0);
						x = x * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;
						y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

						for (size_t ispot = 0; ispot < n; ++ispot) {
							const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

							total_field += CEXP(p_phase + pists[ispot]);
						}

						// std::arg is not working!
						phase[tid[1] * WIDTH + tid[0]] = std::atan2(total_field.imag(), total_field.real());
					}
				}
			);
		}).wait();

		// then use the total phases to update the spot fields, but remember that we have to iterate all over pupil points again!
		// lame implementation for now, could iterate only over pupil points, could do parallel reductions
		q.submit([&](cl::sycl::handler& cgh) {
			cgh.parallel_for<class test_spots>(
				spot_items,
				[=](cl::sycl::id<1> tid) {
					std::complex<double> acc(0.0, 0.0);

					for (int j = 0; j < HEIGHT; ++j) {
						for (int i = 0; i < WIDTH; ++i) {
							double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
							double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

							if (x*x + y*y < 1.0) {
								std::complex<double> total_field(0.0, 0.0);
								x = x * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;
								y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

								const double total_phase = phase[j * WIDTH + i];
								const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[tid], x, y);

								acc += CEXP(total_phase - p_phase);
							}
						}
					}

					pists[tid] = std::atan2(acc.imag(), acc.real());
				}
			);
		}).wait();
	}
}


void gs_kernel_pupil(queue &q, const int n, const Point3D spots[], double pists[], double phase[], const SLM::Parameters par, const int iterations) {
	cl::sycl::range<1> work_items{static_cast<size_t>(pupil_count)};
	cl::sycl::range<1> spot_items{static_cast<size_t>(n)};

	for (int it = 0; it < iterations; ++it) {
		// first compute all the total phases
		q.submit([&](cl::sycl::handler& cgh) {
			cl::sycl::accessor access_pupil{buff_pupil, cgh};

			cgh.parallel_for<class test>(
				work_items,
				[=](cl::sycl::id<1> tid) {
					const int index = access_pupil[tid];
					const int i = index % WIDTH;
					const int j = index / WIDTH;

					const double x = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  i) * static_cast<double>(WIDTH)  / 2.0;
					const double y = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, j) * static_cast<double>(HEIGHT) / 2.0;

					std::complex<double> total_field(0.0, 0.0);

					for (size_t ispot = 0; ispot < n; ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += CEXP(p_phase + pists[ispot]);
					}

					// std::arg is not working!
					// duplication in phase writing!
					phase[index] = std::atan2(total_field.imag(), total_field.real());
				}
			);
		}).wait();

		// then use the total phases to update the spot fields, but remember that we have to iterate all over pupil points again!
		q.submit([&](cl::sycl::handler& cgh) {
			cl::sycl::accessor access_pupil{buff_pupil, cgh};
			const int local_pupil_count = pupil_count;

			cgh.parallel_for<class test_spots>(
				spot_items,
				[=](cl::sycl::id<1> tid) {
					std::complex<double> acc(0.0, 0.0);

					for (int pup = 0; pup < local_pupil_count; ++pup) {
						const int index = access_pupil[pup];
						const int i = index % WIDTH;
						const int j = index / WIDTH;

						const double x = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  i) * static_cast<double>(WIDTH)  / 2.0;
						const double y = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, j) * static_cast<double>(HEIGHT) / 2.0;

						const double total_phase = phase[index];
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[tid], x, y);

						acc += CEXP(total_phase - p_phase);
					}

					pists[tid] = std::atan2(acc.imag(), acc.real());
				}
			);
		}).wait();
	}
}


void gs_kernel_reduction(queue &q, const int n, const Point3D spots[], double pists[], std::complex<double> spot_fields[], double phase[], const SLM::Parameters par, const int iterations) {
	cl::sycl::range<1> work_items{static_cast<size_t>(pupil_count)};
	cl::sycl::range<1> spot_items{static_cast<size_t>(n)};

	std::complex<double> zero(0.0, 0.0);
	cl::sycl::buffer<std::complex<double>> test_accumulator_buffer{&zero, 1};

	for (int it = 0; it < iterations; ++it) {
		// first compute all the total phases
		q.submit([&](cl::sycl::handler& cgh) {
			cl::sycl::accessor access_pupil{buff_pupil, cgh};

			cgh.parallel_for<class test>(
				work_items,
				[=](cl::sycl::id<1> tid) {
					const int index = access_pupil[tid];
					const int i = index % WIDTH;
					const int j = index / WIDTH;

					const double x = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  i) * static_cast<double>(WIDTH)  / 2.0;
					const double y = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, j) * static_cast<double>(HEIGHT) / 2.0;

					std::complex<double> total_field(0.0, 0.0);

					for (size_t ispot = 0; ispot < n; ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						total_field += CEXP(p_phase + pists[ispot]);
					}

					// std::arg is not working!
					// duplication in phase writing!
					phase[index] = std::atan2(total_field.imag(), total_field.real());
				}
			);
		}).wait();

		// then use the total phases to update the spot fields
		// crazy idea: use sycl::reduction for every spot
		for (size_t ispot = 0; ispot < 1; ++ispot) {
#if 0
			q.submit([&](cl::sycl::handler& cgh) {
				cl::sycl::accessor access_pupil{buff_pupil, cgh};

				cgh.parallel_for<class test_reduction>(
					work_items,
					cl::sycl::reduction(spot_fields + ispot, cl::sycl::plus<>()),
					[=](cl::sycl::id<1> tid, auto &acc) {
						const int index = access_pupil[tid];
						const int i = index % WIDTH;
						const int j = index / WIDTH;

						const double x = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  i) * static_cast<double>(WIDTH)  / 2.0;
						const double y = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, j) * static_cast<double>(HEIGHT) / 2.0;

						const double total_phase = phase[index];

						// this line gives segfaults, no idea why
						const double p_phase = 0.0; // COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);

						acc += CEXP(total_phase - p_phase);
					}
				);
			});
#endif
		}

		q.wait();

		// last kernel invocation to manage spot update
		q.submit([&](cl::sycl::handler& cgh) {
			cgh.parallel_for<class spot_update>(
				spot_items,
				[=](cl::sycl::id<1> tid) {
					pists[tid] = std::atan2(spot_fields[tid].imag(), spot_fields[tid].real());
				}
			);
		}).wait();
	}
}


void gs_kernel_block(queue &q, const int n, const Point3D spots[], double pists[], std::complex<double> spot_fields[], double phase[], const SLM::Parameters par, const int iterations) {
	for (int it = 0; it < iterations; ++it) {
		q.submit([&](cl::sycl::handler& cgh) {
			cl::sycl::accessor access_pupil{buff_pupil, cgh};

			cgh.parallel_for(
				cl::sycl::nd_range<1>{pupil_count * n, n},
				[=](cl::sycl::nd_item<1> it) {
					auto g = it.get_group();

					const int index = access_pupil[g.get_group_id()];
					const int ispot = g.get_local_id();

					const int i = index % WIDTH;
					const int j = index / WIDTH;

					const double x = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  i) * static_cast<double>(WIDTH)  / 2.0;
					const double y = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, j) * static_cast<double>(HEIGHT) / 2.0;

					const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);
					const std::complex<double> total_field = cl::sycl::reduce_over_group(g, CEXP(p_phase + pists[ispot]), cl::sycl::plus<>());
					const double total_phase = std::atan2(total_field.imag(), total_field.real());

					const std::complex<double> spot_contribution = CEXP(total_phase - p_phase);

					cl::sycl::atomic_ref<
						double,
						cl::sycl::memory_order::relaxed,
						cl::sycl::memory_scope::system,
						cl::sycl::access::address_space::global_space
					> atomic_spot_fields_real(*(reinterpret_cast<double*>(spot_fields + ispot)));

					cl::sycl::atomic_ref<
						double,
						cl::sycl::memory_order::relaxed,
						cl::sycl::memory_scope::system,
						cl::sycl::access::address_space::global_space
					> atomic_spot_fields_imag(*(reinterpret_cast<double*>(spot_fields + ispot) + 1));


					atomic_spot_fields_real += spot_contribution.real();
					atomic_spot_fields_imag += spot_contribution.imag();

					if (g.leader()) {
						phase[index] = total_phase;
					}
				}
			);
		}).wait();

		q.submit([&](cl::sycl::handler& cgh) {
			cl::sycl::accessor access_pupil{buff_pupil, cgh};

			cgh.parallel_for(
				cl::sycl::range<1>{static_cast<size_t>(n)},
				[=](cl::sycl::id<1> tid) {
					pists[tid] = std::atan2(spot_fields[tid].imag(), spot_fields[tid].real());
					spot_fields[tid] = std::complex<double>(0.0, 0.0);
				}
			);
		}).wait();
	}
}
