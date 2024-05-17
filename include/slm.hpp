#ifndef __SLM_HPP__
#define __SLM_HPP__


#include <vector>
#include "point.hpp"
#include "units.hpp"



/*
 *  General guidelines:
 *   - kernel signature should be as primitive as possible to maximize performance portability
 *   - this interface is meant to be implemented by multiple backends (CPU, CUDA, HIP, OpenGL4 compute shaders, SYCL)
 *   - GPU implementations of this interface won't be optimal for a couple of reasons:
 *     ~ transfering point information can be costly, it would be better if data is loaded once and transformed on the device (see python/example.py)
 */


struct Performance {
	double efficiency;
	double uniformity;
	double variance;
	double time;
};


namespace SLM {
	struct Parameters {
		const int	width;
		const int	height;
		const double	focal_length_mm;
		const double	pixel_size_um;
		const double	wavelength_um;

		Parameters(int width_, int height_, Length focal_length, Length pixel_size, Length wavelength) :
			width(width_),
			height(height_),
			focal_length_mm(focal_length.as(Unit::Millimeters)),
			pixel_size_um(pixel_size.as(Unit::Micrometers)),
			wavelength_um(wavelength.as(Unit::Micrometers)) {};
	};


	class PupilIterator {
		public:
			PupilIterator(const SLM::Parameters &parameters);

			Point2D        operator*();
			PupilIterator& operator++();
			bool           operator!=(PupilIterator &other);

			bool empty = false;

		private:
			std::vector<Point2D> pupil_coordinates;
			std::vector<Point2D>::iterator current_point;
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

			const int n;
			std::vector<double> pists;
			std::vector<double> phase;
	};


}


std::vector<Point2D>            generate_pupil_coordinates (const SLM::Parameters &parameters);
std::vector<int>                generate_pupil_indices     (const SLM::Parameters &parameters);
std::vector<std::pair<int,int>> generate_pupil_index_bounds(const SLM::Parameters &parameters);
std::vector<std::pair<int,int>>  compute_pupil_index_bounds(const SLM::Parameters &parameters);


#endif
