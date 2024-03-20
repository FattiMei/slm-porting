#ifndef __SLM_HPP__
#define __SLM_HPP__


#include <vector>
#include <cstdio>
#include <fstream>


/*
 *  General guidelines:
 *   - kernel signature should be as primitive as possible to maximize performance portability
 *   - this interface is meant to be implemented by multiple backends (CPU, CUDA, HIP, OpenGL4 compute shaders, SYCL)
 *   - GPU implementations of this interface won't be optimal for a couple of reasons:
 *     ~ computing and rendering could be fused in a single kernel call
 *     ~ transfering point information can be costly, it would be better if data is loaded once and transformed on the device (see python/example.py)
 */


class SLM {
	public:
		SLM(int width_, int height_, double wavelength_um_, double pixel_size_um_, double focal_length_mm);

		// @DESIGN: for performance reasons it could be convenient to store point data in AoS form
		void    rs(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &z,                                     int seed, bool measure = false);
		void    gs(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &z, int iterations,                     int seed, bool measure = false);
		void   wgs(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &z, int iterations,                     int seed, bool measure = false);
		void  csgs(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &z, int iterations, double compression, int seed, bool measure = false);
		void wcsgs(const std::vector<double> &x, const std::vector<double> &y, const std::vector<double> &z, int iterations, double compression, int seed, bool measure = false);

		void write_on_texture(int id);
		void write_on_file(FILE *out);
		void write_on_file(std::ofstream &out);

	private:
		const int width;
		const int height;
		const double wavelength_um;
		const double pixel_size_um;
		const double focal_length_mm;

		std::vector<double> phase_buffer;
		std::vector<unsigned char> texture_buffer;

		void    rs_kernel(int n, const double x[], const double y[], const double z[], int width, int height, double phase[], double perf[4], double focal_length, double pitch, double wavelength,                                     int seed);
		void    gs_kernel(int n, const double x[], const double y[], const double z[], int width, int height, double phase[], double perf[4], double focal_length, double pitch, double wavelength, int iterations,                     int seed);
		void   wgs_kernel(int n, const double x[], const double y[], const double z[], int width, int height, double phase[], double perf[4], double focal_length, double pitch, double wavelength, int iterations,                     int seed);
		void  csgs_kernel(int n, const double x[], const double y[], const double z[], int width, int height, double phase[], double perf[4], double focal_length, double pitch, double wavelength, int iterations, double compression, int seed);
		void wcsgs_kernel(int n, const double x[], const double y[], const double z[], int width, int height, double phase[], double perf[4], double focal_length, double pitch, double wavelength, int iterations, double compression, int seed);
};


#endif
