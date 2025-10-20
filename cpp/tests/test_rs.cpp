#include <chrono>
#include <iostream>
#include "slm.h"
#include "rand.h"
#include "types.h"
#include "impl/rs.h"


#define TIC() std::chrono::high_resolution_clock::now()
#define ELAPSED_S(start_time, end_time) (std::chrono::duration_cast<std::chrono::duration<double>>((end_time) - (start_time)))


int main() {
	constexpr int n = 100;

	SLM slm = get_standard_slm();
	Rng rng(42);

	std::vector<double> x(n);
	std::vector<double> y(n);
	std::vector<double> z(n);
	std::vector<double> pists(n);
	std::vector<double> phase(slm.pupil_idx.size());
	SpotSoaContainer<Spot> spots(std::move(x),
	                             std::move(y),
	                             std::move(z));

	rng.fill(spots.m_x, -50.0, 50.0);
	rng.fill(spots.m_y, -50.0, 50.0);
	rng.fill(spots.m_z, -5.0, 5.0);
	rng.fill(pists, 0.0, 2*M_PI);

	const auto start_time = TIC();
	// rs(spots, pists, slm.xx, slm.yy, slm.C1, slm.C2, phase);
	// rs_simd(spots, pists, slm.xx, slm.yy, slm.C1, slm.C2, phase);
	rs_simulated_simd(spots, pists, slm.xx, slm.yy, slm.C1, slm.C2, phase);
	const auto end_time = TIC();

	std::cout << std::format("Compute time: {} s\n",
	                         ELAPSED_S(start_time, end_time));
	std::cout << phase[phase.size()-1] << std::endl;

	return 0;
}
