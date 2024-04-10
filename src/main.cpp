#include <iostream>
#include <fstream>
#include <random>
#include <complex>
#include "slm.hpp"
#include "utils.hpp"
#include "units.hpp"
#include "kernels.hpp"


const int width            = 512;
const int height           = 512;
const int npoints          = 100;
const int iterations       = 30;
const double compression   = 0.05;
const int seed             = 42;


const Length wavelength  (0.488, Unit::Micrometers);
const Length pitch       ( 15.0, Unit::Micrometers);
const Length focal_length( 20.0, Unit::Millimeters);


int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Error in command line arguments" << std::endl;
		std::cerr << "Usage: porting <output_filename>" << std::endl;

		return 1;
	}


	const SLM::Parameters parameters(width, height, focal_length, pitch, wavelength);

	const std::vector<Point3D> spots = generate_grid_spots(10, 10.0);
	const int nspots = spots.size();

	const std::vector<double> pists = generate_random_vector(nspots, 0.0, 2.0 * M_PI, 1);
	std::vector<double> pists_tmp(nspots);
	std::vector<double> weights(nspots);
	std::vector<double> phase(parameters.width * parameters.height);
	std::vector<double> ints(nspots);
	std::vector<std::complex<double>> spot_fields(nspots);
	std::vector<double> p_phase_cache(nspots);


	std::ofstream out(argv[1]);


	std::vector<double> pists_copy = pists;


	// rs_kernel_naive(nspots, spots.data(), pists.data(), phase.data(), &parameters);
	// gs_kernel_naive(nspots, spots.data(), pists_copy.data(), pists_tmp.data(), spot_fields.data(), phase.data(), &parameters, iterations);
	gs_kernel_cached(nspots, spots.data(), pists_copy.data(), p_phase_cache.data(), spot_fields.data(), phase.data(), &parameters, iterations);
	// wgs_kernel(nspots, spots.data(), pists.data(), pists_tmp.data(), spot_fields.data(), ints.data(), weights.data(), phase.data(), &parameters, NULL, iterations);
	// csgs_kernel(nspots, spots.data(), pists.data(), pists_tmp.data(), spot_fields.data(), phase.data(), &parameters, NULL, iterations, compression, seed);
	// wcsgs_kernel(nspots, spots.data(), pists.data(), pists_tmp.data(), spot_fields.data(), ints.data(), weights.data(), phase.data(), &parameters, NULL, iterations, compression, seed);

	write_vector_on_file(phase, parameters.width, parameters.height, out);
	write_vector_on_file(pists, nspots, 1, out);
	write_spots_on_file(spots, out);


	return 0;
}
