#ifndef __WINDOW_H__
#define __WINDOW_H__


#include <GLFW/glfw3.h>
#include <vector>


// there should only be one instance of this class
class Window {
	public:
		Window(const char *title, int width, int height);
		~Window();

		bool should_close();
		void poll_events();
		void swap_buffers();


	private:
		void set_hints(const std::vector<std::pair<int, int>> &hints);
		void set_callbacks();

		GLFWwindow *window = NULL;
};


#endif
