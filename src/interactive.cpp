#include "slm.hpp"
#include "utils.hpp"
#include "units.hpp"
#include "window.hpp"
#include "render.hpp"


#define DEFAULT_WINDOW_WIDTH  800
#define DEFAULT_WINDOW_HEIGHT 600


const int width            = 512;
const int height           = 512;
const int npoints          = 100;
const int iterations       = 30;
const double compression   = 0.05;
const int seed             = 42;


const Length wavelength  (0.488, Unit::Micrometers);
const Length pitch       ( 15.0, Unit::Micrometers);
const Length focal_length( 20.0, Unit::Millimeters);


int main() {
	int frames = 0;
	SLM slm(width, height, wavelength, pitch, focal_length);


	std::vector<double> pists(npoints);
	generate_random_vector(pists, 0.0, 2.0 * M_PI, 1);


	std::vector<Point3D> spots;
	generate_grid_spots(10, 10.0, spots);


	if (window_init("slm application", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT) != 0) {
		window_close();
		exit(EXIT_FAILURE);
	}


	render_init();
	window_set_callbacks();


	double last_time = glfwGetTime();

	while (!window_should_close()) {
		double current_time = glfwGetTime();
		++frames;

		if (current_time - last_time >= 1.0) {
			printf("%f ms/frame\n", 1000.0 * (current_time - last_time) / ((double) frames));

			frames = 0;
			last_time += 1.0;
		}

		slm.rs(spots, pists);
		render_present();

		window_swap_buffers();
		window_poll_events();
	}

	window_close();
	return 0;
}
