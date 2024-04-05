#include "utils.hpp"
#include <random>
#include <cassert>


static double linspace(double inf, double sup, int n, int i) {
	return inf + static_cast<double>(i) * (sup - inf) / static_cast<double>(n - 1);
}


std::vector<double> generate_random_vector(int n, double inf, double sup, int seed) {
	std::default_random_engine gen(seed);
	std::uniform_real_distribution<double> uniform(inf, sup);
	std::vector<double> result(n);

	for (auto &x : result) {
		x = uniform(gen);
	}

	return result;
}


std::vector<Point3D> generate_grid_spots(int n, double size) {
	std::vector<Point3D> result;
	result.reserve(n * n);

	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < n; ++j) {
			const double x = linspace(-0.5 * size, 0.5 * size, size, i);
			const double y = linspace(-0.5 * size, 0.5 * size, size, j);

			result.push_back(Point3D(x, y, 0.0));
		}
	}

	return result;
}


void write_vector_on_file(const std::vector<double> &x, size_t width, size_t height, std::ofstream &out) {
	assert(width * height == x.size());

	out << width << " " << height << std::endl;
	out.write(reinterpret_cast<const char *>(x.data()), width * height * sizeof(double));
}


// @BAD: at the moment I accept this horrible code, might refactor when my C++ skills grow
void write_spots_on_file(const std::vector<Point3D> &spots, std::ofstream &out) {
	// magic number 3 is the number of elements in the Point3D struct
	out << spots.size() << " " << 3 << std::endl;
	out.write(reinterpret_cast<const char *>(spots.data()), spots.size() * 3 * sizeof(double));
}
