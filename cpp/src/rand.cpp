#include <random>
#include "rand.h"


void Rng::fill(std::vector<double>& dest, double inf, double sup) {
	std::uniform_real_distribution<double> uniform(inf, sup);

	for (auto& x : dest) {
		x = uniform(m_gen);
	}
}
