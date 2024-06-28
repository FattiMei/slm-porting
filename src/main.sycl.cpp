#include <iostream>
#include "config.hpp"
#include "slm.hpp"
#include "utils.hpp"
#include "units.hpp"
#include <CL/sycl.hpp>
#include "kernels.sycl.hpp"


int main(int argc, char* argv[]) {
	const SLM::Parameters parameters(
		512,
		512,
		Length(20.0, Unit::Millimeters),
		Length(15.0, Unit::Micrometers),
		Length(488.0, Unit::Nanometers)
	);


	if (argc != 2) {
		std::cerr << "Error in command line arguments" << std::endl;
		std::cerr << "Usage: porting <output_filename>" << std::endl;

		return 1;
	}


	const std::vector<Point3D> spots = generate_grid_spots(10, 10.0);
	const std::vector<double>  pists = generate_random_vector(spots.size(), 0.0, 2.0 * M_PI, 10);
	      std::vector<double>  phase(parameters.width * parameters.height, 0.0);
	std::ofstream out(argv[1]);

	// SYCL code here...
	cl::sycl::queue q;
	double  *device_pists = cl::sycl::malloc_device<double>(pists.size(), q);
	Point3D *device_spots = cl::sycl::malloc_device<Point3D>(spots.size(), q);
	double  *device_phase = cl::sycl::malloc_device<double>(phase.size(), q);

	q.memcpy(device_spots, spots.data(), spots.size() * sizeof(Point3D));
	q.memcpy(device_pists, pists.data(), pists.size() * sizeof(double));
	q.wait();

	rs_kernel_naive(q, static_cast<int>(spots.size()), device_spots, device_pists, device_phase, parameters);
	// rs_kernel_pupil(q, static_cast<int>(spots.size()), device_spots, device_pists, device_phase, parameters);
	// rs_kernel_local(q, static_cast<int>(spots.size()), device_spots, device_pists, device_phase, parameters);
	// gs_kernel_naive(q, static_cast<int>(spots.size()), device_spots, device_pists, device_phase, parameters, 30);

	q.wait();
	q.memcpy(phase.data(), device_phase, phase.size() * sizeof(double));
	q.wait();


	write_vector_on_file(phase, parameters.width, parameters.height, out);
	write_vector_on_file(pists, spots.size(), 1, out);
	write_spots_on_file(spots, out);


	return 0;
}
