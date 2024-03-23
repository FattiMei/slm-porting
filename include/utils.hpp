#ifndef __UTILS_HPP__
#define __UTILS_HPP__


#include <vector>
#include <random>
#include <fstream>


void generate_random_vector(std::vector<double> &x, double inf, double sup, int seed);
void write_vector_on_file(const std::vector<double> &x, size_t width, size_t height, std::ofstream &out);


#endif
