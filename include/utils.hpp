#ifndef __UTILS_HPP__
#define __UTILS_HPP__


#include <vector>
#include <fstream>
#include "slm.hpp"


double linspace(double inf, double sup, int n, int i);
std::vector<double>  generate_random_vector(int n, double inf, double sup, int seed);
std::vector<Point3D> generate_grid_spots(int n, double size);
void write_vector_on_file(const std::vector<double> &x, size_t width, size_t height, std::ofstream &out);
void write_spots_on_file(const std::vector<Point3D> &spots, std::ofstream &out);


#endif
