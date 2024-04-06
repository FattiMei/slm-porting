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
 *     ~ transfering point information can be costly, it would be better if data is loaded once and transformed on the device (see python/example.py)
 */


struct Point3D {
	double x;
	double y;
	double z;

	Point3D(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {};
};


struct Performance {
	double efficiency;
	double uniformity;
	double variance;
	double time;
};


namespace SLM {
	struct Parameters {
		const int width;
		const int height;
		const double focal_length_mm;
		const double pixel_size_um;
		const double wavelength_um;

		Parameters(int width_, int height_, Length focal_length, Length pixel_size, Length wavelength) :
			width(width_),
			height(height_),
			focal_length_mm(focal_length.as(Unit::Millimeters)),
			pixel_size_um(pixel_size.as(Unit::Micrometers)),
			wavelength_um(wavelength.as(Unit::Micrometers)) {};
	};


	// this class could be inherited from when we will test CUDA implementation, in the constructor we could allocate memory on the GPU
	// I don't like this approach, for now I will do my testing in an imperative way
	class Wrapper {
		public:
			Wrapper(const SLM::Parameters parameters, const std::vector<Point3D> &spots);

			void    rs();
			void    gs(int iterations);
			void   wgs(int iterations);
			void  csgs(int iterations, double compression, int seed = 0);
			void wcsgs(int iterations, double compression, int seed = 0);


		private:
			const SLM::Parameters parameters;
			const std::vector<Point3D> &spots;

			// @DESIGN: these vectors won't change their dimension, should I declare them as something different that std::vector?
			const int n;
			std::vector<double> pists;
			std::vector<double> phase;
	};


}


std::vector<int>                generate_pupil_indices     (const SLM::Parameters &parameters);
std::vector<std::pair<int,int>> generate_pupil_index_bounds(const SLM::Parameters &parameters);


#endif
