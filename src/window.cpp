#include "window.hpp"
#include <GLFW/glfw3.h>
#include <stdio.h>
#include <vector>


// @DESIGN: is there something better than vector?
static const std::vector<std::pair<int, int>> default_window_hints = {
	{GLFW_RESIZABLE			, GLFW_TRUE			},
	{GLFW_VISIBLE			, GLFW_TRUE			},
	{GLFW_DECORATED			, GLFW_TRUE			},
	{GLFW_FOCUSED			, GLFW_TRUE			},
	{GLFW_AUTO_ICONIFY		, GLFW_TRUE			},
	{GLFW_FLOATING			, GLFW_FALSE			},
	{GLFW_MAXIMIZED			, GLFW_FALSE			},
	{GLFW_CENTER_CURSOR		, GLFW_TRUE			},
	{GLFW_TRANSPARENT_FRAMEBUFFER	, GLFW_FALSE			},
	{GLFW_FOCUS_ON_SHOW		, GLFW_TRUE			},
	{GLFW_SCALE_TO_MONITOR		, GLFW_FALSE			},
	{GLFW_RED_BITS			, 8				},
	{GLFW_GREEN_BITS		, 8				},
	{GLFW_BLUE_BITS			, 8				},
	{GLFW_ALPHA_BITS		, 8				},
	{GLFW_DEPTH_BITS		, 24				},
	{GLFW_STENCIL_BITS		, 8				},
	{GLFW_ACCUM_RED_BITS		, 0				},
	{GLFW_ACCUM_GREEN_BITS		, 0				},
	{GLFW_ACCUM_BLUE_BITS		, 0				},
	{GLFW_ACCUM_ALPHA_BITS		, 0				},
	{GLFW_AUX_BUFFERS		, 0				},
	{GLFW_SAMPLES			, 0				},
	{GLFW_REFRESH_RATE		, GLFW_DONT_CARE		},
	{GLFW_STEREO			, GLFW_FALSE			},
	{GLFW_SRGB_CAPABLE		, GLFW_FALSE			},
	{GLFW_DOUBLEBUFFER		, GLFW_TRUE			},
	{GLFW_CLIENT_API		, GLFW_OPENGL_API		},
	{GLFW_CONTEXT_CREATION_API	, GLFW_NATIVE_CONTEXT_API	},
	{GLFW_CONTEXT_VERSION_MAJOR	, 2				},
	{GLFW_CONTEXT_VERSION_MINOR	, 0				},
	{GLFW_CONTEXT_ROBUSTNESS	, GLFW_NO_ROBUSTNESS		},
	{GLFW_CONTEXT_RELEASE_BEHAVIOR	, GLFW_ANY_RELEASE_BEHAVIOR	},
	{GLFW_OPENGL_FORWARD_COMPAT	, GLFW_FALSE			},
	{GLFW_OPENGL_DEBUG_CONTEXT	, GLFW_FALSE			},
	{GLFW_OPENGL_PROFILE		, GLFW_OPENGL_ANY_PROFILE	}
};


static void error_callback(int error, const char* description) {
	fprintf(stderr, "GLFW error (code %d): %s\n", error, description);
}


static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	(void) scancode;
	(void) mods;

	switch (key) {
		case GLFW_KEY_W: break;
		case GLFW_KEY_A: break;
		case GLFW_KEY_S: break;
		case GLFW_KEY_D: break;
	}

	switch (action) {
		case GLFW_PRESS  : break;
		case GLFW_REPEAT : break;
		case GLFW_RELEASE: break;
	}

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}
	else if (key == GLFW_KEY_SPACE && action == GLFW_PRESS) {
		// toggle pause state
	}
	else if (key == GLFW_KEY_R && action == GLFW_PRESS) {
		// reload
		// write on texture
	}
}


static void mouse_button_callback(GLFWwindow *window, int button, int action, int mods) {
	(void) window;
	(void) mods;

	switch (button) {
		case GLFW_MOUSE_BUTTON_LEFT  : break;
		case GLFW_MOUSE_BUTTON_RIGHT : break;
	}

	switch (action) {
		case GLFW_PRESS  : break;
		case GLFW_RELEASE: break;
	}
}


static void resize_callback(GLFWwindow *window, int width, int height) {
	(void) window;

	glViewport(0, 0, width, height);
}


Window::Window(const char *title, int width, int height) {
	glfwInit();

	glfwSetErrorCallback(error_callback);
	set_hints(default_window_hints);

	window = glfwCreateWindow(width, height, title, NULL, NULL);
	if (window == NULL) {
		// @DESIGN: do we need to throw an exception?
		glfwTerminate();
	}

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	// @FUTURE: this might hurt in the future when we will integrate ImGUI
	set_callbacks();
	glViewport(0, 0, width, height);
}


void Window::set_hints(const std::vector<std::pair<int, int>> &hints) {
	for (auto h : hints) {
		glfwWindowHint(h.first, h.second);
	}
}


void Window::set_callbacks() {
	glfwSetKeyCallback            (window, key_callback);
	glfwSetMouseButtonCallback    (window, mouse_button_callback);
	glfwSetFramebufferSizeCallback(window, resize_callback);
}


bool Window::should_close() {
	return glfwWindowShouldClose(window) != 0;
}


void Window::swap_buffers() {
	glfwSwapBuffers(window);
}


void Window::poll_events() {
	glfwPollEvents();
}


Window::~Window() {
	glfwTerminate();
}
