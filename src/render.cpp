#include "render.hpp"
#include "shader.hpp"
#include "texture.hpp"
#include "slm.hpp"


static unsigned int render_program;


const float vertices[] = {
	-1.0f, -1.0f, 0.0f,
	-1.0f,  1.0f, 0.0f,
	 1.0f, -1.0f, 0.0f,
	-1.0f,  1.0f, 0.0f,
	 1.0f,  1.0f, 0.0f,
	 1.0f, -1.0f, 0.0f
};


const float texture_vertices[] = {
	0.0f, 0.0f,
	0.0f, 1.0f,
	1.0f, 0.0f,
	0.0f, 1.0f,
	1.0f, 1.0f,
	1.0f, 0.0f,
};


// to be called after initializing lbm
void render_init() {
	// @TODO, @DESIGN: maybe declare the buffers as static and move them to opengl
	render_program = program_load_from_file("shaders/legacy_vs_quad.glsl", "shaders/legacy_fs_texture_apply.glsl");

	glBindAttribLocation(render_program, 0, "position");
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, vertices);

	glBindAttribLocation(render_program, 1, "atexCoord");
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 0, texture_vertices);
}


void render_resize(int width, int height) {
	glViewport(0, 0, width, height);
}


void render_present() {
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, slm_texture_id);

	glUseProgram(render_program);
	glDrawArrays(GL_TRIANGLES, 0, 6);
}
