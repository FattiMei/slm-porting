#include "slm.hpp"


SLM::SLM(int width_, int height_, double wavelength_nm_, double pixel_size_um_, double focal_length_mm_) : width(width_), height(height_), wavelength_nm(wavelength_nm_), pixel_size_um(pixel_size_um_), focal_length_mm(focal_length_mm_), phase_buffer(width_ * height_), texture_buffer(width_ * height_) {
}


void SLM::write_on_file(std::ofstream &out) {
	out << width << " " << height << std::endl;
	out.write(reinterpret_cast<const char *>(phase_buffer.data()), width * height * sizeof(double));
}


void SLM::rs_kernel(int n, const double x[], const double y[], const double z[], int width, int height, double phase[], double perf[4], int seed) {
	(void) n;
	(void) x;
	(void) y;
	(void) z;
	(void) width;
	(void) height;
	(void) phase;
	(void) perf;
	(void) seed;
}


void SLM::rs(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &z, int seed, bool measure) {
	double perf[4];

	rs_kernel(x.size(), x.data(), y.data(), z.data(), width, height, phase_buffer.data(), measure ? perf : NULL, seed);
}
