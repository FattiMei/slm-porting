#ifndef __RS_H__
#define __RS_H__


#include <cmath>
#include <cassert>
#include <complex>
#include <format>
#include <iostream>
#include <experimental/simd>
#include "types.h"


template <SpotContainer Vector>
void rs(const Vector& spots,
        const std::vector<double>& pists,
        const std::vector<double>& xx,
        const std::vector<double>& yy,
        const double C1,
        const double C2,
        std::vector<double>& phase) {

#pragma omp parallel for
	for (std::size_t i = 0; i < xx.size(); ++i) {
		std::complex<double> acc(0.0, 0.0);

		for (int spot_idx = 0; spot_idx < spots.size(); ++spot_idx) {
			const auto spot = spots[spot_idx];
			const auto& x = spot.x;
			const auto& y = spot.y;
			const auto& z = spot.z;

			const double p_phase = C1 * (x*xx[i] + y*yy[i]) +
			                       C2 * z * (xx[i]*xx[i] + yy[i]*yy[i]) +
			                       2.0 * M_PI * pists[spot_idx];

			acc += std::complex<double>(std::cos(p_phase), std::sin(p_phase));
		}

		// no normalization is required here
		phase[i] = std::arg(acc);
	}
}


template <SpotContainer Vector>
void rs_simd(const Vector& spots,
             const std::vector<double>& pists,
             const std::vector<double>& xx,
             const std::vector<double>& yy,
             const double C1,
             const double C2,
             std::vector<double>& phase) {

	using simd_t = std::experimental::native_simd<double>;
	constexpr auto tile_size = simd_t::size();
	const auto tile_count = xx.size() / tile_size;
	const auto remainder = xx.size() % tile_size;

#pragma omp parallel for
	for (int tile = 0; tile < tile_count; ++tile) {
		simd_t vxx;
		simd_t vyy;
		simd_t acc_real(0.0);
		simd_t acc_imag(0.0);

		vxx.copy_from(xx.data() + tile * tile_size, std::experimental::element_aligned);
		vyy.copy_from(yy.data() + tile * tile_size, std::experimental::element_aligned);

		for (int spot_idx = 0; spot_idx < spots.size(); ++spot_idx) {
			const auto spot = spots[spot_idx];
			const auto& x = spot.x;
			const auto& y = spot.y;
			const auto& z = spot.z;

			const simd_t p_phase = C1 * (x*vxx + y*vyy)
			                     + C2 * z * (vxx*vxx + vyy*vyy)
			                     + 2.0 * M_PI * pists[spot_idx];

			acc_real += std::experimental::cos(p_phase);
			acc_imag += std::experimental::sin(p_phase);
		}

		for (int i = 0; i < tile_size; ++i) {
			phase[tile * tile_size + i] = std::arg(
				std::complex<double>(
					acc_real[i],
					acc_imag[i]
				)
			);
		}
	}

	if (remainder > 0) {
		for (std::size_t i = xx.size() - remainder; i < xx.size(); ++i) {
			std::complex<double> acc(0.0, 0.0);

			for (int spot_idx = 0; spot_idx < spots.size(); ++spot_idx) {
				const auto spot = spots[spot_idx];
				const auto& x = spot.x;
				const auto& y = spot.y;
				const auto& z = spot.z;

				const double p_phase = C1 * (x*xx[i] + y*yy[i]) +
				                       C2 * z * (xx[i]*xx[i] + yy[i]*yy[i]) +
				                       2.0 * M_PI * pists[spot_idx];

				acc += std::complex<double>(std::cos(p_phase), std::sin(p_phase));
			}

			// no normalization is required here
			phase[i] = std::arg(acc);
		}
	}
}


template <SpotContainer Vector, auto tile_size = 4>
void rs_simulated_simd(const Vector& spots,
                       const std::vector<double>& pists,
                       const std::vector<double>& xx,
                       const std::vector<double>& yy,
                       const double C1,
                       const double C2,
                       std::vector<double>& phase) {

	const auto tile_count = xx.size() / tile_size;
	const auto remainder = xx.size() % tile_size;

#pragma omp parallel for
	for (int tile = 0; tile < tile_count; ++tile) {
		double vxx[tile_size];
		double vyy[tile_size];
		std::complex<double> acc[tile_size];

		for (int i = 0; i < tile_size; ++i) {
			vxx[i] = xx[tile * tile_size + i];
			vyy[i] = yy[tile * tile_size + i];
			acc[i] = std::complex<double>(0.0, 0.0);
		}

		for (int spot_idx = 0; spot_idx < spots.size(); ++spot_idx) {
			const auto spot = spots[spot_idx];
			const auto& x = spot.x;
			const auto& y = spot.y;
			const auto& z = spot.z;

			for (int i = 0; i < tile_size; ++i) {
				const double p_phase = C1 * (x*vxx[i] + y*vyy[i]) +
				                       C2 * z * (vxx[i]*vxx[i] + vyy[i]*vyy[i]) +
				                       2.0 * M_PI * pists[spot_idx];

				acc[i] += std::complex<double>(std::cos(p_phase), std::sin(p_phase));
			}
		}

		for (int i = 0; i < tile_size; ++i) {
			phase[tile * tile_size + i] = std::arg(acc[i]);
		}
	}

	if (remainder > 0) {
		for (std::size_t i = xx.size() - remainder; i < xx.size(); ++i) {
			std::complex<double> acc(0.0, 0.0);

			for (int spot_idx = 0; spot_idx < spots.size(); ++spot_idx) {
				const auto spot = spots[spot_idx];
				const auto& x = spot.x;
				const auto& y = spot.y;
				const auto& z = spot.z;

				const double p_phase = C1 * (x*xx[i] + y*yy[i]) +
				                       C2 * z * (xx[i]*xx[i] + yy[i]*yy[i]) +
				                       2.0 * M_PI * pists[spot_idx];

				acc += std::complex<double>(std::cos(p_phase), std::sin(p_phase));
			}

			// no normalization is required here
			phase[i] = std::arg(acc);
		}
	}
}


#endif // __RS_H__
