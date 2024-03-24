#include "utils.hpp"
#include <cassert>


// @TODO: see correct linspace implementation
double linspace(double inf, double sup, int n, int i) {
	return inf + static_cast<double>(i) * (sup - inf) / static_cast<double>(n);
}


void generate_random_vector(std::vector<double> &x, double inf, double sup, int seed) {
	std::default_random_engine gen(seed);
	std::uniform_real_distribution<double> uniform(inf, sup);

	for (auto &p : x) {
		p = uniform(gen);
	}
}


void generate_grid_spots(int n, double size, std::vector<Point3D> &spots) {
	spots.reserve(n);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			const double x = linspace(-0.5 * size, 0.5 * size, size, i);
			const double y = linspace(-0.5 * size, 0.5 * size, size, j);

			spots.emplace_back(x, y, 0.0);
		}
	}

	assert(spots.size() == n*n);
}


void write_vector_on_file(const std::vector<double> &x, size_t width, size_t height, std::ofstream &out) {
	assert(width * height == x.size());

	out << width << " " << height << std::endl;
	out.write(reinterpret_cast<const char *>(x.data()), width * height * sizeof(double));
}
