#include <vector>
#include <iostream>
#include "cnpy.h"


// Put the integers from 1 to 10
// and the doubles from 1.0 to 10.0
// in a .npz file


int main(int argc, char* argv[]) {
	if (argc == 1) {
		std::cout << "[ERROR]: I need the output filename\n";
		return 1;
	}

	const char* output_filename = argv[1];

	const std::vector<int> integers{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
	const std::vector<double> doubles{1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

	cnpy::npz_save(
		output_filename,
		"integers",
		integers.data(),
		{integers.size()},
		"a"
	);

	cnpy::npz_save(
		output_filename,
		"doubles",
		doubles.data(),
		{doubles.size()},
		"a"
	);

	return 0;
}
