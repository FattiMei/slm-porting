#include <iostream>
#include <fstream>
#include "slm.hpp"


int main(int argc, char *argv[]) {
	if (argc != 2) {
		std::cerr << "Error in command line arguments" << std::endl;
		std::cerr << "Usage: porting <output_filename>" << std::endl;

		return 1;
	}

	SLM slm(512, 512, 488.0, 15.0, 30.0);

	std::ofstream out(argv[1]);
	slm.write_on_file(out);

	return 0;
}
