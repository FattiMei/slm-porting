#include <iostream>
#include "slm.hpp"
#include "utils.hpp"
#include "kernels.hpp"


int main() {
	SLM::Parameters parameters(
		512,
		512,
		Length(20.0, Unit::Millimeters),
		Length(15.0, Unit::Micrometers),
		Length(488.0, Unit::Nanometers)
	);

	const auto spots              = generate_grid_spots(10, 10.0);
	const auto pists              = generate_random_vector(spots.size(), 0.0, 2.0 * M_PI, 1);
	const auto pupil_indices      = generate_pupil_indices(parameters);
	const auto pupil_index_bounds = generate_pupil_index_bounds(parameters);

	std::vector<double> reference  (parameters.width * parameters.height);
	std::vector<double> alternative(parameters.width * parameters.height);


	rs_kernel_naive(spots.size(), spots.data(), pists.data(), reference.data(), &parameters);

	{
		rs_kernel_pupil_indices(spots.size(), spots.data(), pists.data(), alternative.data(), pupil_indices.size(), pupil_indices.data(), &parameters);

		const Difference diff = compare_outputs(reference, alternative);

		std::cout << "rs_kernel_pupil_indices" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		rs_kernel_pupil_index_bounds(spots.size(), spots.data(), pists.data(), alternative.data(), pupil_index_bounds.data(), &parameters);

		const Difference diff = compare_outputs(reference, alternative);

		std::cout << "rs_kernel_pupil_indices" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		rs_kernel_pupil_indices(spots.size(), spots.data(), pists.data(), alternative.data(), pupil_indices.size(), pupil_indices.data(), &parameters);

		const Difference diff = compare_outputs(reference, alternative);

		std::cout << "rs_kernel_pupil_indices" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		rs_kernel_pupil_index_bounds(spots.size(), spots.data(), pists.data(), alternative.data(), pupil_index_bounds.data(), &parameters);

		const Difference diff = compare_outputs(reference, alternative);

		std::cout << "rs_kernel_pupil_index_bounds" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}
	{
		rs_kernel_static_index_bounds(spots.size(), spots.data(), pists.data(), alternative.data(), &parameters);

		const Difference diff = compare_outputs(reference, alternative);

		std::cout << "rs_kernel_static_index_bounds" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}


	std::vector<double> pists_mutable = pists;
	std::vector<double> pists_copy_buffer(spots.size());
	std::vector<std::complex<double>> spot_fields(spots.size());

	gs_kernel_naive(spots.size(), spots.data(), pists_mutable.data(), pists_copy_buffer.data(), spot_fields.data(), reference.data(), &parameters, 30);
	{
		pists_mutable = pists;
		gs_kernel_loop_fusion(spots.size(), spots.data(), pists_mutable.data(), pists_copy_buffer.data(), spot_fields.data(), alternative.data(), &parameters, 30);

		const Difference diff = compare_outputs(reference, alternative);

		std::cout << "gs_kernel_loop_fusion" << std::endl;
		std::cout << "max abs err: " << diff.linf_norm << std::endl;
	}


	return 0;
}

