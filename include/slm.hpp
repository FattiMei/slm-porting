#ifndef __SLM_HPP__
#define __SLM_HPP__


#include <vector>
#include <fstream>
#include <complex>
#include "units.hpp"


/*
 *  General guidelines:
 *   - kernel signature should be as primitive as possible to maximize performance portability
 *   - this interface is meant to be implemented by multiple backends (CPU, CUDA, HIP, OpenGL4 compute shaders, SYCL)
 *   - GPU implementations of this interface won't be optimal for a couple of reasons:
 *     ~ computing and rendering could be fused in a single kernel call
 *     ~ transfering point information can be costly, it would be better if data is loaded once and transformed on the device (see python/example.py)
 */


extern int slm_texture_id;


// @ADVICE: can we do something better with this constructor?
struct Point3D {
	double x;
	double y;
	double z;

	Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {};
};


struct SLMParameters {
	const int width;
	const int height;
	const double focal_length_mm;
	const double pixel_size_um;
	const double wavelength_um;

	SLMParameters(int width_, int height_, Length focal_length, Length pixel_size, Length wavelength) :
		width(width_),
		height(height_),
		focal_length_mm(focal_length.as(Unit::Millimeters)),
		pixel_size_um(pixel_size.as(Unit::Micrometers)),
		wavelength_um(wavelength.as(Unit::Micrometers)) {};
};


struct Performance {
	double efficiency;
	double uniformity;
	double variance;
	double time;
};



class SLM {
	public:
		SLM(int width, int height, const Length &wavelength, const Length &pixel_size, const Length &focal_length);

		void    rs(const std::vector<Point3D> &spots, const std::vector<double> &pists,                                                   bool measure = false);
		void    gs(const std::vector<Point3D> &spots, const std::vector<double> &pists, int iterations,                                   bool measure = false);
		void   wgs(const std::vector<Point3D> &spots, const std::vector<double> &pists, int iterations,                                   bool measure = false);
		void  csgs(const std::vector<Point3D> &spots, const std::vector<double> &pists, int iterations, double compression, int seed = 0, bool measure = false);
		void wcsgs(const std::vector<Point3D> &spots, const std::vector<double> &pists, int iterations, double compression, int seed = 0, bool measure = false);

		void write_on_texture();
		void render();
		void write_on_file(std::ofstream &out);

	private:
		const struct SLMParameters par;

		// @DESIGN: perf should be a characteristic of SLM or a measure of the kernel? I think it's better to leave this out
		// in fact all of these members should be outside the class, the have no meaning since their dimension can change due to different inputs to kernels
		// a wrapper to this class might be more friendly by pre allocating the vectors, but the invariants don't hold here
		struct Performance perf;
		std::vector<double> phase_buffer;
		std::vector<unsigned char> texture_buffer;

		void rs_kernel(
			int                  n,
			const Point3D        spots[],
			const double         pists[],
			double               phase[],
			const SLMParameters* par,
			Performance*         perf
		);

		void gs_kernel(
			int                  n,
			const Point3D        spots[],
			double               pists[],
			double               pists_tmp_buffer[],
			std::complex<double> spot_fields[],
			double               phase[],
			const SLMParameters* par,
			Performance*         perf,
			int                  iterations
		);

		void wgs_kernel(
			int                  n,
			const Point3D        spots[],
			double               pists[],
			double               pists_tmp_buffer[],
			std::complex<double> spot_fields[],
			double               ints[],
			double               weights[],
			double               phase[],
			const SLMParameters* par,
			Performance*         perf,
			int                  iterations
		);

		void csgs_kernel(
			int                  n,
			const Point3D        spots[],
			double               pists[],
			double               pists_tmp_buffer[],
			std::complex<double> spot_fields[],
			double               phase[],
			const SLMParameters* par,
			Performance*         perf,
			int                  iterations,
			double               compression,
			int                  seed
		);

		void wcsgs_kernel(
			int                  n,
			const Point3D        spots[],
			double               pists[],
			double               pists_tmp_buffer[],
			std::complex<double> spot_fields[],
			double               ints[],
			double               weights[],
			double               phase[],
			const SLMParameters* par,
			Performance*         perf,
			int                  iterations,
			double               compression,
			int                  seed
		);
};


#endif
