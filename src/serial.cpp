#include "slm.hpp"
#include "kernels.hpp"
#include "utils.hpp"
#include <complex>


/*
SLM::SLM(int width, int height, const Length &wavelength, const Length &pixel_size, const Length &focal_length) : 
	par(width, height, focal_length, pixel_size, wavelength),
	phase_buffer(width * height) {
}


void SLM::write_on_file(std::ofstream &out) {
	write_vector_on_file(phase_buffer, par.width, par.height, out);
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
*/
