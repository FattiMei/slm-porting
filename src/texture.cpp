#include "texture.hpp"
#include <GLFW/glfw3.h>


unsigned int texture_create(int width, int height) {
	(void) width;
	(void) height;
	unsigned int id;

	glGenTextures(1, &id);
	// glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, id);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);

	return id;
}


void texture_destroy(unsigned int texture_id) {
	glDeleteTextures(1, &texture_id);
}
