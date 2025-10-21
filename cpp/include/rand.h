#ifndef __RAND_H__
#define __RAND_H__


#include <random>
#include <vector>
#include "types.h"


class Rng {
	public:
		Rng(int seed) : m_gen(seed) {}
		Rng(const Rng& other)  = delete;
		Rng(const Rng&& other) = delete;

		void fill(std::vector<double>& dest, double inf, double sup);

		// hardcoded functioni for testing AoS data layout
		void fill(std::vector<SpotLike auto>& dest) {
			std::uniform_real_distribution<double> uniform_xy(-50.0, 50.0);
			std::uniform_real_distribution<double> uniform_z(-5.0, 5.0);

			for (auto& spot : dest) {
				spot.x = uniform_xy(m_gen);
				spot.y = uniform_xy(m_gen);
				spot.z = uniform_z(m_gen);
			}
		}

	private:
		std::default_random_engine m_gen;
};


#endif // __RAND_H__
