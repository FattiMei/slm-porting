#include "utils.hpp"
#include <cassert>


void generate_random_vector(std::vector<double> &x, double inf, double sup, int seed) {
	std::default_random_engine gen(seed);
	std::uniform_real_distribution<double> uniform(inf, sup);

	for (auto &p : x) {
		p = uniform(gen);
	}
}


void write_vector_on_file(const std::vector<double> &x, size_t width, size_t height, std::ofstream &out) {
	assert(width * height == x.size());

	out << width << " " << height << std::endl;
	out.write(reinterpret_cast<const char *>(x.data()), width * height * sizeof(double));
}
