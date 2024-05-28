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


void rs_kernel_naive(queue &q, const std::vector<Point3D> &spots, const std::vector<double> &pists, std::vector<double> &phase, const SLM::Parameters par) {
	cl::sycl::buffer<Point3D> buff_spots(spots.data(), spots.size());
	cl::sycl::buffer<double>  buff_pists(pists.data(), pists.size());
	cl::sycl::buffer<double>  buff_phase(phase.data(), phase.size());

	cl::sycl::range<2> work_items{static_cast<size_t>(WIDTH), static_cast<size_t>(HEIGHT)};

	q.submit([&](cl::sycl::handler& cgh) {
		auto access_spots = buff_spots.get_access<cl::sycl::access::mode::read>(cgh);
		auto access_pists = buff_pists.get_access<cl::sycl::access::mode::read>(cgh);
		auto access_phase = buff_phase.get_access<cl::sycl::access::mode::write>(cgh);

		cgh.parallel_for<class rs>(
			work_items,
			[=](cl::sycl::id<2> tid) {
				double x = LINSPACE(-1.0, 1.0, WIDTH,  tid[0]);
				double y = LINSPACE(-1.0, 1.0, HEIGHT, tid[1]);

				if (x*x + y*y < 1.0) {
					std::complex<double> total_field(0.0, 0.0);
					x = x * PIXEL_SIZE * static_cast<double>(WIDTH)  / 2.0;
					y = y * PIXEL_SIZE * static_cast<double>(HEIGHT) / 2.0;

					for (size_t ispot = 0; ispot < access_spots.get_count(); ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, access_spots[ispot], x, y);

						total_field += CEXP(p_phase + access_pists[ispot]);
					}

					// std::arg is not working!
					access_phase[tid[1] * WIDTH + tid[0]] = std::atan2(total_field.imag(), total_field.real());
				}
			}
		);
	});
}


void rs_kernel_pupil(queue &q, const std::vector<Point3D> &spots, const std::vector<double> &pists, std::vector<double> &phase, const SLM::Parameters par) {
	cl::sycl::buffer<Point3D> buff_spots(spots.data(), spots.size());
	cl::sycl::buffer<double>  buff_pists(pists.data(), pists.size());
	cl::sycl::buffer<int>     buff_pupil(pupil_indices, pupil_count);
	cl::sycl::buffer<double>  buff_phase(phase.data(), phase.size());

	cl::sycl::range<1> work_items{static_cast<size_t>(pupil_count)};

	q.submit([&](cl::sycl::handler& cgh) {
		auto access_spots = buff_spots.get_access<cl::sycl::access::mode::read>(cgh);
		auto access_pists = buff_pists.get_access<cl::sycl::access::mode::read>(cgh);
		cl::sycl::accessor access_pupil{buff_pupil, cgh};
		auto access_phase = buff_phase.get_access<cl::sycl::access::mode::write>(cgh);

		cgh.parallel_for<class test>(
			work_items,
			[=](cl::sycl::id<1> tid) {
				const int index = access_pupil[tid];
				const int i = index % WIDTH;
				const int j = index / WIDTH;

				const double x = PIXEL_SIZE * LINSPACE(-1.0, 1.0, WIDTH,  i) * static_cast<double>(WIDTH)  / 2.0;
				const double y = PIXEL_SIZE * LINSPACE(-1.0, 1.0, HEIGHT, j) * static_cast<double>(HEIGHT) / 2.0;

				std::complex<double> total_field(0.0, 0.0);

				for (size_t ispot = 0; ispot < access_spots.get_count(); ++ispot) {
					const double p_phase = COMPUTE_P_PHASE(WAVELENGTH, FOCAL_LENGTH, access_spots[ispot], x, y);

					total_field += CEXP(p_phase + access_pists[ispot]);
				}

				// std::arg is not working!
				access_phase[index] = std::atan2(total_field.imag(), total_field.real());
			}
		);
	});
}
