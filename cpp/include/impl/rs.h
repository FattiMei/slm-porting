#ifndef __RS_H__
#define __RS_H__


#include <cmath>
#include <complex>
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
			const Spot spot = spots[spot_idx];
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
	constexpr auto lane_size = simd_t::size();

#pragma omp parallel for
	for (std::size_t i = 0; i < xx.size() - lane_size; i += lane_size) {
		simd_t vxx;
		simd_t vyy;
		vxx.copy_from(xx.data() + i, std::experimental::element_aligned);
		vyy.copy_from(yy.data() + i, std::experimental::element_aligned);

		simd_t acc_real(0.0);
		simd_t acc_imag(0.0);

		for (int spot_idx = 0; spot_idx < spots.size(); ++spot_idx) {
			const Spot spot = spots[spot_idx];
			const auto& x = spot.x;
			const auto& y = spot.y;
			const auto& z = spot.z;

			const simd_t p_phase = C1 * (x*vxx + y*vyy) +
			                       C2 * z * (vxx*vxx + vyy*vyy) +
			                       2.0 * M_PI * pists[spot_idx];

			acc_real += std::experimental::cos(p_phase);
			acc_imag += std::experimental::sin(p_phase);
		}

		// I need to do the arg
		for (int j = 0; j < lane_size; ++j) {
			phase[i+j] = std::atan2(
				acc_real[j],
				acc_imag[j]
			);
		}
	}

	// process the remaining elements
	const auto remainder = xx.size() % lane_size;
	if (remainder > 0) {
		for (std::size_t i = xx.size() - remainder; i < xx.size(); ++i) {
			std::complex<double> acc(0.0, 0.0);

			for (int spot_idx = 0; spot_idx < spots.size(); ++spot_idx) {
				const Spot spot = spots[spot_idx];
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
			vxx[i] = 0;
			vyy[i] = 0;
			acc[i] = std::complex<double>(0.0, 0.0);
		}

		for (int spot_idx = 0; spot_idx < spots.size(); ++spot_idx) {
			const Spot spot = spots[spot_idx];
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
				const Spot spot = spots[spot_idx];
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
