#include <chrono>
#include <iostream>
#include "slm.h"
#include "rand.h"
#include "types.h"
#include "impl/rs.h"


int main() {
	constexpr int n = 100;

	SLM slm = get_standard_slm();
	Rng rng(42);

	std::vector<double> x(n);
	std::vector<double> y(n);
	std::vector<double> z(n);
	std::vector<double> pists(n);

	rng.fill(x, -50.0, 50.0);
	rng.fill(y, -50.0, 50.0);
	rng.fill(z, -5.0, 5.0);
	rng.fill(pists, 0.0, 2*M_PI);

	std::vector<double> phase(slm.pupil_idx.size());
	SpotSoaContainer<Spot> spots(std::move(x),
	                             std::move(y),
	                             std::move(z));

	rs(spots, pists, slm.xx, slm.yy, slm.C1, slm.C2, phase);

	return 0;
}
