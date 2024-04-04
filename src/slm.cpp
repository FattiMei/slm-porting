#include "slm.hpp"
#include "kernels.hpp"
#include "utils.hpp"


// @DESIGN: I should copy the parameters structure or store the reference?
// @DESIGN: what can I do for allocating std::vectors? resize?
SLMWrapper::SLMWrapper(const SLMParameters &parameters_, const std::vector<Point3D> &spots_) :
	parameters(parameters_),
	spots(spots_),
	n(spots_.size()),
	pists(spots_.size()),
	phase(parameters_.width * parameters_.height)
{
	// @DESIGN: I would like to write
	// 	pists = generate_random_vector(spots.size(), 0.0, 2.0 * M_PI, 1);

	generate_random_vector(pists, 0.0, 2.0 * M_PI, 1);
}


void SLMWrapper::rs() {
	rs_kernel(n, spots.data(), pists.data(), phase.data(), &parameters, NULL);
}