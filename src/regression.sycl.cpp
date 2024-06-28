#include <iostream>
#include "slm.hpp"
#include "utils.hpp"
#include <CL/sycl.hpp>
#include "kernels.sycl.hpp"


constexpr int n      = 100;
constexpr int width  = 512;
constexpr int height = 512;


Point3D spots[n];
double  pists[n];
double  reference[width * height];
double  alternative[width * height];


int main() {
	const SLM::Parameters parameters(
		width,
		height,
		Length(20.0, Unit::Millimeters),
		Length(15.0, Unit::Micrometers),
		Length(488.0, Unit::Nanometers)
	);


	random_fill(3 * n, (double *) spots, -5.0, 5.0, 9);
	random_fill(n, pists, 0.0, 2.0 * M_PI, 8);


	// SYCL setup
	cl::sycl::queue q;
	double  *device_pists = cl::sycl::malloc_device<double> (n, q);
	Point3D *device_spots = cl::sycl::malloc_device<Point3D>(n, q);
	double  *device_phase = cl::sycl::malloc_device<double> (width * height, q);

	q.memcpy(device_spots, spots, n * sizeof(Point3D));
	q.memcpy(device_pists, pists, n * sizeof(double));
	q.wait();


	// setting up the reference for rs kernel
	rs_kernel_naive(q, n, device_spots, device_pists, device_phase, parameters);
	q.wait();
	q.memcpy(reference, device_phase, width * height * sizeof(double));
	q.wait();

	{
		// for the moment I don't test if the kernel actually writes something, but I should
		rs_kernel_local(q, n, device_spots, device_pists, device_phase, parameters);
		q.wait();
		q.memcpy(alternative, device_phase, width * height * sizeof(double));
		q.wait();


		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_local" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		rs_kernel_pupil(q, n, device_spots, device_pists, device_phase, parameters);
		q.wait();
		q.memcpy(alternative, device_phase, width * height * sizeof(double));
		q.wait();


		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_pupil" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		gs_kernel_naive(q, n, device_spots, device_pists, device_phase, parameters, 1);
		q.wait();
		q.memcpy(alternative, device_phase, width * height * sizeof(double));
		q.wait();


		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "gs_kernel_naive" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;


		// Restore the pists
		q.memcpy(device_pists, pists, n * sizeof(double));
		q.wait();
	}
	{
		gs_kernel_pupil(q, n, device_spots, device_pists, device_phase, parameters, 1);
		q.wait();
		q.memcpy(alternative, device_phase, width * height * sizeof(double));
		q.wait();


		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "gs_kernel_pupil" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;


		// Restore the pists
		q.memcpy(device_pists, pists, n * sizeof(double));
		q.wait();
	}


	// setting up the reference for gs kernel
	gs_kernel_naive(q, n, device_spots, device_pists, device_phase, parameters, 30);
	q.wait();
	q.memcpy(reference, device_phase, width * height * sizeof(double));
	q.wait();

	// Restore the pists
	q.memcpy(device_pists, pists, n * sizeof(double));
	q.wait();

	{
		gs_kernel_pupil(q, n, device_spots, device_pists, device_phase, parameters, 30);
		q.wait();
		q.memcpy(alternative, device_phase, width * height * sizeof(double));
		q.wait();


		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "gs_kernel_pupil" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;

		// Restore the pists
		q.memcpy(device_pists, pists, n * sizeof(double));
		q.wait();
	}


	return 0;
}
