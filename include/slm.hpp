#ifndef __SLM_HPP__
#define __SLM_HPP__


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
		SLM(int width_, int height_, double wavelength_nm_, double pixel_size_um_, double focal_length_mm);

		// @DESIGN: for performance reasons it could be convenient to store point data in AoS form
		rs   (int n, const double x[], const double y[], const double z[], int width, int height, double phase[], double perf[4],                                     int seed);
		gs   (int n, const double x[], const double y[], const double z[], int width, int height, double phase[], double perf[4], int iterations,                     int seed);
		wgs  (int n, const double x[], const double y[], const double z[], int width, int height, double phase[], double perf[4], int iterations,                     int seed);
		csgs (int n, const double x[], const double y[], const double z[], int width, int height, double phase[], double perf[4], int iterations, double compression, int seed);
		wcsgs(int n, const double x[], const double y[], const double z[], int width, int height, double phase[], double perf[4], int iterations, double compression, int seed);


	private:
		const int width;
		const int height;
		const double wavelength_nm;
		const double pixel_size_um;
		const double focal_length_mm;

		double        *phase_buffer   = NULL;
		unsigned char *texture_buffer = NULL;

		void render_on_texture(int id);
};


#endif
