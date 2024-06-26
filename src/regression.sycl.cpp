#include <iostream>
#include "slm.hpp"
#include "utils.hpp"
#include <CL/sycl.hpp>
#include "kernels.sycl.hpp"


constexpr int n      = 100;
constexpr int width  = 512;
constexpr int height = 512;


extern int pupil_count;
extern int pupil_indices[];


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
	int     *device_pupil = cl::sycl::malloc_device<int> (pupil_count, q);
	std::complex<double> *device_spot_fields = cl::sycl::malloc_device<std::complex<double>>(n, q);

	q.memcpy(device_spots, spots, n * sizeof(Point3D));
	q.memcpy(device_pists, pists, n * sizeof(double));
	q.memcpy(device_pupil, pupil_indices, pupil_count * sizeof(int));
	q.wait();


	// setting up the reference for rs kernel
	rs_kernel_naive(q, n, device_spots, device_pists, device_phase, parameters);
	q.wait();
	q.memcpy(reference, device_phase, width * height * sizeof(double));
	q.wait();

	{
		// for the moment I don't test if the kernel actually writes something, but I should
		rs_kernel_local(q, n, device_spots, device_pists, device_phase, pupil_count, device_pupil, parameters);
		q.wait();
		q.memcpy(alternative, device_phase, width * height * sizeof(double));
		q.wait();


		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_local" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		rs_kernel_pupil(q, n, device_spots, device_pists, device_phase, pupil_count, device_pupil, parameters);
		q.wait();
		q.memcpy(alternative, device_phase, width * height * sizeof(double));
		q.wait();


		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_pupil" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}


	// setting up the reference for gs kernel
	gs_kernel_naive(q, n, device_spots, device_pists, device_phase, parameters, 30);
	q.wait();
	q.memcpy(reference, device_phase, width * height * sizeof(double));
	q.wait();

	{
		// Restore the pists
		q.memcpy(device_pists, pists, n * sizeof(double));
		q.wait();

		gs_kernel_pupil(q, n, device_spots, device_pists, device_phase, pupil_count, device_pupil, parameters, 30);
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
	{
		// Restore the pists
		q.memcpy(device_pists, pists, n * sizeof(double));
		q.wait();

		gs_kernel_block(q, n, device_spots, device_pists, device_spot_fields, device_phase, pupil_count, device_pupil, parameters, 30);
		q.wait();
		q.memcpy(alternative, device_phase, width * height * sizeof(double));
		q.wait();


		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "gs_kernel_block" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;

		// Restore the pists
		q.memcpy(device_pists, pists, n * sizeof(double));
		q.wait();
	}
	{
		// Restore the pists
		q.memcpy(device_pists, pists, n * sizeof(double));
		q.wait();

		gs_kernel_reduction(q, n, device_spots, device_pists, device_spot_fields, device_phase, pupil_count, device_pupil, parameters, 30);
		q.wait();
		q.memcpy(alternative, device_phase, width * height * sizeof(double));
		q.wait();


		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "gs_kernel_reduction" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;

		// Restore the pists
		q.memcpy(device_pists, pists, n * sizeof(double));
		q.wait();
	}


	return 0;
}
