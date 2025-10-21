#include <chrono>
#include <iostream>
#include "slm.h"
#include "rand.h"
#include "types.h"
#include "impl/rs.h"


#define TIC() std::chrono::high_resolution_clock::now()
#define ELAPSED_S(start_time, end_time) (std::chrono::duration_cast<std::chrono::duration<double>>((end_time) - (start_time)))


double error_function(const std::vector<double>& x,
                      const std::vector<double>& y) {
	double max = 0;

	if (x.size() != y.size()) {
		return NAN;
	}

	for (auto i = 0; i < x.size(); ++i) {
		const auto diff = std::abs(x[i] - y[i]);

		if (diff > max) {
			max = diff;
		}
	}

	return max;
}


int main() {
	constexpr int n = 100;
	constexpr int nruns = 5;

	SLM slm = get_standard_slm();
	Rng rng(42);

	std::vector<double> x(n);
	std::vector<double> y(n);
	std::vector<double> z(n);
	SpotSoaContainer<Spot> spots(std::move(x),
	                             std::move(y),
	                             std::move(z));
	std::vector<double> pists(n);
	std::vector<double> reference_phase(slm.pupil_idx.size());
	std::vector<double> alternative_phase(slm.pupil_idx.size());

	std::cout << "algorithm,name,backend,device,dtype,key,abs_err,total_time\n";

	for (auto i = 0; i < nruns; ++i) {
		rng.fill(spots.m_x, -50.0, 50.0);
		rng.fill(spots.m_y, -50.0, 50.0);
		rng.fill(spots.m_z, -5.0, 5.0);
		rng.fill(pists, 0.0, 2*M_PI);

		rs(spots, pists, slm.xx, slm.yy, slm.C1, slm.C2, reference_phase);

		#define VERIFY_IMPL(impl) do {                                                    \
			for (auto& p : alternative_phase) p = 0.0;                                \
			const auto start_time = TIC();                                            \
			impl(spots, pists, slm.xx, slm.yy, slm.C1, slm.C2, alternative_phase);    \
			const auto end_time = TIC();                                              \
			const auto elapsed_seconds = ELAPSED_S(start_time, end_time);             \
			const auto err = error_function(reference_phase, alternative_phase);      \
			std::cout << std::format(                                                 \
				"Algorithm.RS,{},Backend.CPP,Device.CPU,DType.fp64,NAN,{},{}\n",  \
				#impl, err, elapsed_seconds                                       \
			);                                                                        \
		} while (0);

		VERIFY_IMPL(rs);
		VERIFY_IMPL(rs_simd);
		VERIFY_IMPL(rs_simulated_simd);
	}

	return 0;
}

