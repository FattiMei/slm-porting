#include "slm.hpp"
#include "kernels.hpp"
#include "utils.hpp"


// @DESIGN: I should copy the parameters structure or store the reference?
// @DESIGN: what can I do for allocating std::vectors? resize?
SLM::Wrapper::Wrapper(const SLM::Parameters parameters_, const std::vector<Point3D> &spots_) :
	parameters(parameters_),
	spots     (spots_),
	n         (spots_.size()),
	phase     (parameters_.width * parameters_.height)
{
	pists = generate_random_vector(n, 0.0, 2.0 * M_PI, 1);
}


void SLM::Wrapper::rs() {
	rs_kernel(n, spots.data(), pists.data(), phase.data(), &parameters, NULL);
}
