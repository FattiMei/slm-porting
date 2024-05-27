#include <iostream>
#include <complex>
#include "config.hpp"
#include "slm.hpp"
#include "utils.hpp"
#include "units.hpp"
#include <CL/sycl.hpp>


// https://stackoverflow.com/questions/2683588/what-is-the-fastest-way-to-compute-sin-and-cos-together
#define CEXP(x) std::complex<double>(std::cos(x), std::sin(x))


#define LINSPACE(inf, sup, n, i) ((inf) + ((sup) - (inf)) * static_cast<double>(i) / static_cast<double>((n) - 1))
#define COMPUTE_P_PHASE(w, f, spot, pup_x, pup_y) ((2.0 * M_PI / ((w) * (f) * 1000.0)) * ((spot.x) * (pup_x) + (spot.y) * (pup_y)) + (M_PI * (spot.z) / ((w) * (f) * (f) * 1e6)) * ((pup_x) * (pup_x) + (pup_y) * (pup_y)))


int main(int argc, char* argv[]) {
	constexpr int n      = 100;
	constexpr int width  = 512;
	constexpr int height = 512;


	const SLM::Parameters parameters(
		width,
		height,
		Length(20.0, Unit::Millimeters),
		Length(15.0, Unit::Micrometers),
		Length(488.0, Unit::Nanometers)
	);


	extern const int pupil_count;
	extern const int pupil_indices[];


	if (argc != 2) {
		std::cerr << "Error in command line arguments" << std::endl;
		std::cerr << "Usage: porting <output_filename>" << std::endl;

		return 1;
	}


	const std::vector<Point3D> spots = generate_grid_spots(10, 10.0);
	const std::vector<double>  pists = generate_random_vector(spots.size(), 0.0, 2.0 * M_PI, 2);
	      std::vector<double>  phase(parameters.width * parameters.height);
	std::ofstream out(argv[1]);

	// SYCL code here...
	cl::sycl::queue q;

	{
		cl::sycl::buffer<Point3D> buff_spots(spots.data(), spots.size());
		cl::sycl::buffer<double>  buff_pists(pists.data(), pists.size());
		cl::sycl::buffer<int>     buff_pupil(pupil_indices, pupil_count);
		cl::sycl::buffer<double>  buff_phase(phase.data(), phase.size());

		cl::sycl::range<1> work_items{phase.size()};

		q.submit([&](cl::sycl::handler& cgh) {
			auto access_spots = buff_spots.get_access<cl::sycl::access::mode::read>(cgh);
			auto access_pists = buff_pists.get_access<cl::sycl::access::mode::read>(cgh);
			auto access_pupil = buff_pupil.get_access<cl::sycl::access::mode::read>(cgh);
			auto access_phase = buff_phase.get_access<cl::sycl::access::mode::write>(cgh);

			cgh.parallel_for<class test>(
				work_items,
				[=](cl::sycl::id<1> tid) {
					const int i = access_pupil[tid] % width;
					const int j = access_pupil[tid] / width;

					const double x = parameters.pixel_size_um * LINSPACE(-1.0, 1.0, width,  i) * static_cast<double>(width)  / 2.0;
					const double y = parameters.pixel_size_um * LINSPACE(-1.0, 1.0, height, j) * static_cast<double>(height) / 2.0;

					std::complex<double> total_field(0.0, 0.0);

					for (int ispot = 0; ispot < n; ++ispot) {
						const double p_phase = COMPUTE_P_PHASE(parameters.wavelength_um, parameters.focal_length_mm, access_spots[ispot], x, y);

						total_field += CEXP(p_phase + access_pists[ispot]);
					}

					// std::arg is not working!
					// access_phase[access_pupil[tid]] = std::arg(total_field);
					access_phase[access_pupil[tid]] = std::atan2(total_field.imag(), total_field.real());
				}
			);
		});
	}

	q.wait();

	write_vector_on_file(phase, parameters.width, parameters.height, out);
	write_vector_on_file(pists, spots.size(), 1, out);
	write_spots_on_file(spots, out);


	return 0;
}
