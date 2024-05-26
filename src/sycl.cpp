#include <iostream>
#include "config.hpp"
#include "slm.hpp"
#include "utils.hpp"
#include "units.hpp"
#include <CL/sycl.hpp>


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


int main(int argc, char* argv[]) {
	if (argc != 2) {
		std::cerr << "Error in command line arguments" << std::endl;
		std::cerr << "Usage: porting <output_filename>" << std::endl;

		return 1;
	}


	const std::vector<Point3D> spots = generate_grid_spots(10, 10.0);
	const std::vector<double>  pists = generate_random_vector(spots.size(), 0.0, 2.0 * M_PI, 1);
	      std::vector<double>  phase(parameters.width * parameters.height);
	std::ofstream out(argv[1]);

	// SYCL code here...
	cl::sycl::queue q;
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
				access_phase[tid] = 1.0;
			}
		);
	});

	q.wait();

	write_vector_on_file(phase, parameters.width, parameters.height, out);
	write_vector_on_file(pists, spots.size(), 1, out);
	write_spots_on_file(spots, out);


	std::cerr << phase[0] << std::endl;

	return 0;
}
