#include <iostream>
#include <chrono>
#include "slm.hpp"
#include "utils.hpp"
#include "kernels.hpp"


#define NSAMPLES 1000
#define SEED     1


// we don't care about correctness in this experiment
int main() {
	const int    width            = 512;
	const int    height           = 512;
	const Length focal_length     ( 20.0, Unit::Millimeters);
	const Length pitch            ( 15.0, Unit::Micrometers);
	const Length wavelength       (0.488, Unit::Micrometers);


	SLM::Parameters parameters(width, height, focal_length, pitch, wavelength);
	std::vector<Point3D> spots = generate_grid_spots(10, 10.0);
	SLM::Wrapper wrapper(parameters, spots);

	for (int i = 0; i < NSAMPLES; ++i) {
		const auto start_time = std::chrono::high_resolution_clock::now();

		wrapper.rs();

		const auto end_time = std::chrono::high_resolution_clock::now();
		const auto delta = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();

		std::cout << delta << " ms" << std::endl;
	}


	return 0;
}
