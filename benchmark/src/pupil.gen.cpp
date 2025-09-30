#include <cstdio>
#include "config.hpp"
#include "slm.hpp"


int main() {
	printf("#include <utility>\n\n");


	// Exporting array of pupil indices -------------------------------------------

	const auto pupil_indices = generate_pupil_indices(RESOLUTION);

	printf("extern const int pupil_count = %ld;\n",     pupil_indices.size());
	printf("extern const int pupil_indices[%ld] = {\n", pupil_indices.size());

	for (const int i : pupil_indices) {
		printf("\t%d,\n", i);
	}

	printf("};\n\n");

	// Exporting array of loop bounds for pupil points ----------------------------

	const auto pupil_index_bounds = generate_pupil_index_bounds(RESOLUTION);

	printf("extern const std::pair<int,int> pupil_index_bounds[%d] = {\n", RESOLUTION);

	for (const auto &pair : pupil_index_bounds) {
		printf("\t{%d, %d},\n", pair.first, pair.second);
	}

	printf("};\n\n");


	return 0;
}
