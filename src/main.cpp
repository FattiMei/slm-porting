#include <iostream>
#include <fstream>
#include <random>
#include "slm.hpp"
#include "utils.hpp"
#include "units.hpp"


const int width            = 512;
const int height           = 512;
const int npoints          = 100;
const int iterations       = 30;
const double compression   = 0.05;
const int seed             = 42;


const Length wavelength  (0.488, Unit::micrometers);
const Length pitch       ( 15.0, Unit::micrometers);
const Length focal_length( 20.0, Unit::millimeters);


int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Error in command line arguments" << std::endl;
		std::cerr << "Usage: porting <output_filename>" << std::endl;

		return 1;
	}

	SLM slm(width, height, wavelength, pitch, focal_length);
	std::ofstream out(argv[1]);


	std::vector<double> pists(npoints);
	generate_random_vector(pists, 0.0, 2.0 * M_PI, 1);


	std::vector<Point3D> spots;
	generate_grid_spots(10, 10.0, spots);


	// slm.rs(spots, pists);
	// slm.gs(spots, pists, iterations);
	// slm.wgs(spots, pists, iterations);
	slm.csgs(spots, pists, iterations, compression);

	slm.write_on_file(out);
	write_vector_on_file(pists, npoints, 1, out);
	write_spots_on_file(spots, out);


	return 0;
}
