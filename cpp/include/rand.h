#ifndef __RAND_H__
#define __RAND_H__


#include <random>
#include <vector>


class Rng {
	public:
		Rng(int seed) : m_gen(seed) {}
		Rng(const Rng& other)  = delete;
		Rng(const Rng&& other) = delete;

		void fill(std::vector<double>& dest, double inf, double sup);

	private:
		std::default_random_engine m_gen;
};


#endif // __RAND_H__
