#include <iostream>
#include <fstream>
#include "slm.hpp"
#include "utils.hpp"
#include "units.hpp"
#include "kernels.hpp"


constexpr int n      = 100;
constexpr int width  = 512;
constexpr int height = 512;


const SLM::Parameters parameters(
	width,
	height,
	Length(20.0, Unit::Millimeters),
	Length(15.0, Unit::Micrometers),
	Length(488.0, Unit::Nanometers)
);


extern const int pupil_count;
extern const int pupil_indices[];


int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Error in command line arguments" << std::endl;
		std::cerr << "Usage: porting <output_filename>" << std::endl;

		return 1;
	}


	const std::vector<Point3D> spots = generate_grid_spots(10, 10.0);
	const std::vector<double>  pists = generate_random_vector(spots.size(), 0.0, 2.0 * M_PI, 1);
	      std::vector<double>  phase(parameters.width * parameters.height);
	std::ofstream out(argv[1]);

	rs_kernel_pupil_indices_simd(spots.size(), spots.data(), pists.data(), phase.data(), pupil_count, pupil_indices, &parameters);

	write_vector_on_file(phase, parameters.width, parameters.height, out);
	write_vector_on_file(pists, spots.size(), 1, out);
	write_spots_on_file(spots, out);


	return 0;
}
