#ifndef __UTILS_HPP__
#define __UTILS_HPP__


#include <vector>
#include <fstream>
#include "point.hpp"


double linspace(double inf, double sup, int n, int i);
std::vector<double>  generate_random_vector(int n, double inf, double sup, int seed);
std::vector<Point3D> generate_grid_spots(int n, double size);
void write_vector_on_file(const std::vector<double> &x, size_t width, size_t height, std::ofstream &out);
void write_spots_on_file(const std::vector<Point3D> &spots, std::ofstream &out);


struct Difference {
	const double linf_norm;
	const double l2_norm;

	Difference(double linf_norm_, double l2_norm_) : linf_norm(linf_norm_), l2_norm(l2_norm_) {};
};


struct Difference compare_outputs(const std::vector<double> &reference, const std::vector<double> &alternative);


#endif
