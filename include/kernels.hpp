#ifndef __KERNELS_HPP__
#define __KERNELS_HPP__


#include "slm.hpp"
#include <complex>


// optimizations related to iterating over pupil points (should be exact)
void rs_kernel_naive              (const int n, const Point3D spots[], const double pists[], double phase[],                                                                       const SLM::Parameters *par);
void rs_kernel_pupil_indices      (const int n, const Point3D spots[], const double pists[], double phase[], const int pupil_count, const int                pupil_indices[],      const SLM::Parameters *par);
void rs_kernel_pupil_index_bounds (const int n, const Point3D spots[], const double pists[], double phase[],                        const std::pair<int,int> pupil_index_bounds[], const SLM::Parameters *par);
void rs_kernel_static_index_bounds(const int n, const Point3D spots[], const double pists[], double phase[],                                                                       const SLM::Parameters *par);


// optimizations related to rescheduling mathematical operations (might be different due to finite precision)
void gs_kernel_naive      (const int n, const Point3D spots[], double pists[], double pists_copy_buffer[], std::complex<double> spot_fields[], double phase[], const SLM::Parameters *par, const int iterations);
void gs_kernel_loop_fusion(const int n, const Point3D spots[], double pists[], double pists_copy_buffer[], std::complex<double> spot_fields[], double phase[], const SLM::Parameters *par, const int iterations);


void wgs_kernel(
		int                  n,
		const Point3D        spots[],
		double               pists[],
		double               pists_tmp_buffer[],
		std::complex<double> spot_fields[],
		double               ints[],
		double               weights[],
		double               phase[],
		const SLM::Parameters* par,
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
		const SLM::Parameters* par,
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
		const SLM::Parameters* par,
		Performance*         perf,
		int                  iterations,
		double               compression,
		int                  seed
		);


#endif
