#include <iostream>
#include "slm.hpp"
#include "utils.hpp"
#include "kernels.hpp"


constexpr int n      = 100;
constexpr int width  = 512;
constexpr int height = 512;


Point3D spots[n];
double  pists[n];
double  reference[width * height];
double  alternative[width * height];


extern const int pupil_count;
extern const int pupil_indices[];
extern const std::pair<int, int> pupil_index_bounds[];


int main() {
	const SLM::Parameters parameters(
		width,
		height,
		Length(20.0, Unit::Millimeters),
		Length(15.0, Unit::Micrometers),
		Length(488.0, Unit::Nanometers)
	);


	random_fill(3 * n, (double *) spots, -5.0, 5.0, 0);
	random_fill(n, pists, 0.0, 2.0 * M_PI, 0);


	for (int i = 0; i < width * height; ++i) {
		reference[i] = 0.0;
	}


	// reference implementation
	rs_kernel_static_scheduling(n, spots, pists, reference, &parameters);
	{
		for (int i = 0; i < width * height; ++i) alternative[i] = 0.0;

		rs_kernel_dynamic_scheduling(n, spots, pists, alternative, &parameters);
		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_dynamic_scheduling" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		for (int i = 0; i < width * height; ++i) alternative[i] = 0.0;

		rs_kernel_custom_scheduling(n, spots, pists, alternative, &parameters);
		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_custom_scheduling" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		for (int i = 0; i < width * height; ++i) alternative[i] = 0.0;

		rs_kernel_branchless(n, spots, pists, alternative, &parameters);
		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_branchless" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		for (int i = 0; i < width * height; ++i) alternative[i] = 0.0;

		rs_kernel_branch_delay_slot(n, spots, pists, alternative, &parameters);
		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_branch_delay_slot" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		for (int i = 0; i < width * height; ++i) alternative[i] = 0.0;

		rs_kernel_pupil_indices(n, spots, pists, alternative, pupil_count, pupil_indices, &parameters);
		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_pupil_indices" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		for (int i = 0; i < width * height; ++i) alternative[i] = 0.0;

		rs_kernel_static_index_bounds(n, spots, pists, alternative, pupil_index_bounds, &parameters);
		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_static_index_bounds" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		for (int i = 0; i < width * height; ++i) alternative[i] = 0.0;

		rs_kernel_computed_index_bounds(n, spots, pists, alternative, &parameters);
		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_computed_index_bounds" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		for (int i = 0; i < width * height; ++i) alternative[i] = 0.0;

		rs_kernel_math_cache(n, spots, pists, alternative, &parameters);
		const Difference diff = compare_outputs(width, height, reference, alternative);

		std::cout << "rs_kernel_math_cache" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}


	return 0;
}
