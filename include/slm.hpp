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


struct Point3D {
	double x;
	double y;
	double z;

	Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {};
};


// @TODO: solve the unit of measure problem
struct SLMParameters {
	const int width;
	const int height;
	const double focal_length_mm;
	const double pixel_size_um;
	const double wavelength_um;

	SLMParameters(int width_, int height_, double focal_length_mm_, double pixel_size_um_, double wavelength_um_) :
		width(width_),
		height(height_),
		focal_length_mm(focal_length_mm_),
		pixel_size_um(pixel_size_um_),
		wavelength_um(wavelength_um_) {};
};


struct Performance {
	double efficiency;
	double uniformity;
	double variance;
	double time;
};



class SLM {
	public:
		SLM(int width_, int height_, double wavelength_um_, double pixel_size_um_, double focal_length_mm);

		void    rs(const std::vector<Point3D> &spots, std::vector<double> &pists,                                     bool measure = false);
		void    gs(const std::vector<Point3D> &spots, std::vector<double> &pists, int iterations,                     bool measure = false);
		void   wgs(const std::vector<Point3D> &spots, std::vector<double> &pists, int iterations,                     bool measure = false);
		void  csgs(const std::vector<Point3D> &spots, std::vector<double> &pists, int iterations, double compression, bool measure = false);
		void wcsgs(const std::vector<Point3D> &spots, std::vector<double> &pists, int iterations, double compression, bool measure = false);

		void write_on_texture(int id);
		void write_on_file(std::ofstream &out);

	private:
		const struct SLMParameters par;

		struct Performance perf;
		std::vector<double> phase_buffer;
		std::vector<unsigned char> texture_buffer;

		void    rs_kernel(int n, const Point3D spots[], double pists[], double phase[], const SLMParameters *par, Performance *perf);
		void    gs_kernel(int n, const Point3D spots[], double pists[], double phase[], const SLMParameters *par, Performance *perf, int iterations);
		void   wgs_kernel(int n, const Point3D spots[], double pists[], double phase[], const SLMParameters *par, Performance *perf, int iterations,                     int seed);
		void  csgs_kernel(int n, const Point3D spots[], double pists[], double phase[], const SLMParameters *par, Performance *perf, int iterations, double compression, int seed);
		void wcsgs_kernel(int n, const Point3D spots[], double pists[], double phase[], const SLMParameters *par, Performance *perf, int iterations, double compression, int seed);
};


#endif
