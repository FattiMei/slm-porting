#include "slm.hpp"


SLM::SLM(int width_, int height_, double wavelength_nm_, double pixel_size_um_, double focal_length_mm_) : width(width_), height(height_), wavelength_nm(wavelength_nm_), pixel_size_um(pixel_size_um_), focal_length_mm(focal_length_mm_) {
	phase_buffer   = new double[width * height];
	texture_buffer = new unsigned char[width * height];
}


SLM::~SLM() {
	delete[] phase_buffer;
	delete[] texture_buffer;
}
