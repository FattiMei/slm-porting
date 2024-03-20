#include <iostream>
#include <fstream>
#include <random>
#include "slm.hpp"


const double focal_length  = 20.0;
const int width            = 512;
const int height           = 512;
const double pitch         = 15.0;
const double wavelength_nm = 488.0;
const int npoints          = 100;
const int iterations       = 30;
const double compression   = 0.05;
const int seed             = 42;


int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Error in command line arguments" << std::endl;
		std::cerr << "Usage: porting <output_filename>" << std::endl;

		return 1;
	}

	SLM slm(width, height, wavelength_nm, pitch, focal_length);
	std::ofstream out(argv[1]);


	// manually filling the random vectors
	std::default_random_engine gen(seed);
	std::uniform_real_distribution<double> uniform(0.0, 1.0);

	std::vector<double> x(npoints);
	std::vector<double> y(npoints);
	std::vector<double> z(npoints);

	for (int i = 0; i < npoints; ++i) {
		// i know it's not compatible with the python version
		x[i] = 100.0 * (uniform(gen) - 0.5);
		y[i] = 100.0 * (uniform(gen) - 0.5);
		z[i] =  10.0 * (uniform(gen) - 0.5);
	}

	slm.rs(x, y, z, 1);
	slm.write_on_file(out);

	return 0;
}
