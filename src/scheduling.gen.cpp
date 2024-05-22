#include <cstdio>
#include "config.hpp"
#include "slm.hpp"


int main() {
	const auto index_bounds = generate_pupil_index_bounds(RESOLUTION);

	std::vector<int> cost_per_row(index_bounds.size());
	int total_cost = 0;

	// in this simple model, the cost of a row is the number of pupil points in that row
	// this is accurate for kernels that don't do any filtering like rs_kernel_static_index_bounds
	for (int i = 0; i < RESOLUTION; ++i) {
		cost_per_row[i] = index_bounds[i].second - index_bounds[i].first;
		total_cost += cost_per_row[i];
	}

	// since the load distribution is inherently imprecise, this is a nice operations research problem
	std::vector<int> schedule{0};
	const int cost_per_thread = total_cost / OMP_NUM_THREADS;

	// this is the lamest solution I could think of (and implement)
	int acc = 0;
	for (int i = 0, acc = 0; i < RESOLUTION; ++i) {
		acc += cost_per_row[i];

		if (acc >= cost_per_thread) {
			acc = 0;
			schedule.push_back(i + 1);
		}
	}
	schedule.push_back(RESOLUTION);

	// Exporting array of thread schedule chunks ----------------------------------
	printf("extern const int thread_schedule_bounds[] = {\n");

	for (const int i : schedule) {
		printf("\t%i, ", i);
	}

	printf("};\n");


	// @TODO: print uniformity metrics, this will give estimations about the supposed speedup over static scheduling

	
	return 0;
}
