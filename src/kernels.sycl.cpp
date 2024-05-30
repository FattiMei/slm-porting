#include "kernels.sycl.hpp"
#include <complex>


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
	});
}


void rs_kernel_pupil(queue &q, const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters par) {
	// with this trick we send once the pupil indices vector
	static cl::sycl::buffer<int>     buff_pupil(pupil_indices, pupil_count);
	cl::sycl::range<1> work_items{static_cast<size_t>(pupil_count)};

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
				phase[index] = std::atan2(total_field.imag(), total_field.real());
			}
		);
	});
}


void rs_kernel_local(queue &q, const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters par) {
	q.submit([&](cl::sycl::handler& cgh) {
		cl::sycl::local_accessor<Point3D> local_spot{1, cgh};

		cgh.parallel_for<class rs>(
			cl::sycl::nd_range<2>{{static_cast<size_t>(WIDTH), static_cast<size_t>(HEIGHT)}, {1, 16}},
			[=](cl::sycl::nd_item<2> item) {
				const int i = item.get_global_id()[0];
				const int j = item.get_global_id()[1];

				double x = LINSPACE(-1.0, 1.0, WIDTH,  i);
				double y = LINSPACE(-1.0, 1.0, HEIGHT, j);

				if (x*x + y*y < 1.0) {
					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (size_t ispot = 0; ispot < n; ++ispot) {
						if (item.get_local_id()[0] == 0) {
							local_spot[0] = spots[ispot];
						}
						item.barrier();

						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, spots[ispot], x, y);
						total_field += CEXP(p_phase + pists[ispot]);

						item.barrier();
					}

					// std::arg is not working!
					phase[j * WIDTH + i] = std::atan2(total_field.imag(), total_field.real());
				}
			}
		);
	});
}
