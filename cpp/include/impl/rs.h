#ifndef __RS_H__
#define __RS_H__


#include <cmath>
#include <complex>
#include "types.h"


template <SpotContainer Vector>
void rs(const Vector& spots,
        const std::vector<double>& pists,
        const std::vector<double>& xx,
        const std::vector<double>& yy,
        const double C1,
        const double C2,
        std::vector<double>& phase) {

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


#endif // __RS_H__
