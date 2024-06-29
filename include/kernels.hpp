#ifndef __KERNELS_HPP__
#define __KERNELS_HPP__


#include "slm.hpp"
#include <complex>


// thread scheduling variants
void rs_kernel_static_scheduling    (const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters *par);
void rs_kernel_dynamic_scheduling   (const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters *par);
void rs_kernel_custom_scheduling    (const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters *par);


// reordering variants
void rs_kernel_branchless           (const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters *par);
void rs_kernel_branch_delay_slot    (const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters *par);


// math operations variants
void rs_kernel_cache_constants      (const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters *par);
void rs_kernel_math_cache           (const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters *par);


// iteration on pupil points variants + scheduling
void rs_kernel_pupil_indices        (const int n, const Point3D spots[], const double pists[], double phase[], const int pupil_count, const int                pupil_indices[],      const SLM::Parameters *par);
void rs_kernel_pupil_indices_simd   (const int n, const Point3D spots[], const double pists[], double phase[], const int pupil_count, const int                pupil_indices[],      const SLM::Parameters *par);
void rs_kernel_pupil_indices_dual   (const int n, const Point3D spots[], const double pists[], double phase[], const int pupil_count, const int                pupil_indices[],      const SLM::Parameters *par);
void rs_kernel_static_index_bounds  (const int n, const Point3D spots[], const double pists[], double phase[],                        const std::pair<int,int> pupil_index_bounds[], const SLM::Parameters *par);
void rs_kernel_computed_index_bounds(const int n, const Point3D spots[], const double pists[], double phase[],                                                                       const SLM::Parameters *par);


// useful for estimation of performance across multiple architectures
void rs_kernel_upper_bound               (const int n, const Point3D spots[], const double pists[], double phase[],                                                                       const SLM::Parameters *par);


// optimizations related to caching results and reordering operations
void gs_kernel_naive    (const int n, const Point3D spots[], double pists[],                         std::complex<double> spot_fields[], double phase[], const SLM::Parameters *par, const int iterations);
void gs_kernel_pupil    (const int n, const Point3D spots[], double pists[],                         std::complex<double> spot_fields[], double phase[], const int pupil_count, const int pupil_indices[], const SLM::Parameters *par, const int iterations);
void gs_kernel_cached   (const int n, const Point3D spots[], double pists[], double p_phase_cache[], std::complex<double> spot_fields[], double phase[], const SLM::Parameters *par, const int iterations);
void gs_kernel_reordered(const int n, const Point3D spots[], double pists[],                         std::complex<double> spot_fields[], double phase[], const SLM::Parameters *par, const int iterations);


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
