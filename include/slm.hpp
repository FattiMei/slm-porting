#ifndef __SLM_HPP__
#define __SLM_HPP__


#include <vector>
#include "units.hpp"
#include "utils.hpp"


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
}


std::vector<std::pair<int,int>> generate_pupil_index_bounds(const int resolution);
std::vector<std::pair<int,int>>  compute_pupil_index_bounds(const int resolution);
std::vector<int> generate_pupil_indices(const int resolution);


#endif
