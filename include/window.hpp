#ifndef __WINDOW_H__
#define __WINDOW_H__


#include <GLFW/glfw3.h>


// there should only be one instance of this class
class Window {
	public:
		Window(const char *title, int width, int height);
		~Window();

		bool should_close();
		void poll_events();
		void swap_buffers();


	private:
		void set_hints(const int hints[][2], int n);
		void set_callbacks();

		GLFWwindow *window = NULL;
};


#endif
