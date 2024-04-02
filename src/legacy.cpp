#include "window.hpp"


#define DEFAULT_WINDOW_WIDTH  800
#define DEFAULT_WINDOW_HEIGHT 600


int main() {
	if (window_init("legacy opengl", DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT) != 0) {
		window_close();
		return 1;
	}

	window_set_callbacks();


	glMatrixMode( GL_PROJECTION );
	glLoadIdentity();
	// abbiamo bisogno di questa cosa?
	// glOrtho( 0.0, SCREEN_WIDTH, SCREEN_HEIGHT, 0.0, 1.0, -1.0 );

	glMatrixMode( GL_MODELVIEW );
	glLoadIdentity();

	glClearColor(0.0f, 1.0f, 0.2f, 1.0f);

	glEnable( GL_TEXTURE_2D );

	unsigned char tex[64][64][3];

	for (int i = 0; i < 64; ++i) {
		for (int j = 0; j < 64; ++j) {
			if (i < 30) {
				tex[i][j][0] = 100;
				tex[i][j][1] = 100;
				tex[i][j][2] = 100;
			}
			else {
				tex[i][j][0] = 200;
				tex[i][j][1] = 200;
				tex[i][j][2] = 200;
			}
		}
	}

	unsigned int id;
	glGenTextures(1, &id);
	glBindTexture(GL_TEXTURE_2D, id);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 64, 64, 0, GL_RGB, GL_UNSIGNED_BYTE, tex);

	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );

	// glBindTexture( GL_TEXTURE_2D, NULL );

	while (!window_should_close()) {
		glClear(GL_COLOR_BUFFER_BIT);
		
		glBindTexture(GL_TEXTURE_2D, id);
		glBegin( GL_QUADS );

		glVertex2f( -0.5f, -0.5f );
		glTexCoord2f(0.0f, 0.0f);

		glVertex2f(  0.5f, -0.5f );
		glTexCoord2f(1.0f, 0.0f);

		glVertex2f(  0.5f,  0.5f );
		glTexCoord2f(1.0f, 1.0f);

		glVertex2f( -0.5f,  0.5f );
		glTexCoord2f(0.0f, 1.0f);
		glEnd();


		window_swap_buffers();
		window_poll_events();
	}

	window_close();
	return 0;
}
