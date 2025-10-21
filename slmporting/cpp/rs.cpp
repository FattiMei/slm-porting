#include <torch/extension.h>
#include <cmath>
#include <complex>
#include <experimental/simd>


void primitive_rs(const int64_t nspots,
                  const double* x,
                  const double* y,
                  const double* z,
                  const double* pists,
                  const int64_t npupils,
                  const double* xx,
                  const double* yy,
                  double C1,
                  double C2,
                  double* phase) {

#pragma omp parallel for
	for (int64_t i = 0; i < npupils; ++i) {
		std::complex<double> acc(0.0, 0.0);

		for (int64_t spot_idx = 0; spot_idx < nspots; ++spot_idx) {
			const double p_phase = C1 * (x[spot_idx]*xx[i] + y[spot_idx]*yy[i]) +
			                       C2 * z[spot_idx] * (xx[i]*xx[i] + yy[i]*yy[i]) +
			                       2.0 * M_PI * pists[spot_idx];

			acc += std::complex<double>(std::cos(p_phase), std::sin(p_phase));
		}

		// no normalization is required here
		phase[i] = std::arg(acc);
	}
}


// we can't really use metaprogramming here so we'll just write a lot of wrappers
torch::Tensor rs(torch::Tensor x,
                 torch::Tensor y,
                 torch::Tensor z,
                 torch::Tensor pists,
                 torch::Tensor xx,
                 torch::Tensor yy,
                 double C1,
                 double C2) {
	// I don't think I'm going to be checking every assumption about
	// this data. I'm confident that the caller pays attention.
	// Also this could be a performance malus
#if TORCH_CHECK_THINGS_BEFORE_COMPUTING
	TORCH_CHECK(x.device().is_cpu());
	TORCH_CHECK(y.device().is_cpu());
	TORCH_CHECK(z.device().is_cpu());
	TORCH_CHECK(pists.device().is_cpu());
	TORCH_CHECK(xx.device().is_cpu());
	TORCH_CHECK(yy.device().is_cpu());

	TORCH_CHECK(x.sizes() == y.sizes() == z.sizes() == pists.sizes());
	TORCH_CHECK(xx.sizes() == yy.sizes());

	// assume also that the tensors are contiguous
	TORCH_CHECK(x.is_contiguous());
	TORCH_CHECK(y.is_contiguous());
	TORCH_CHECK(z.is_contiguous());
	TORCH_CHECK(pists.is_contiguous());
	TORCH_CHECK(xx.is_contiguous());
	TORCH_CHECK(yy.is_contiguous());
#endif

	torch::Tensor phase = torch::empty_like(xx);
	primitive_rs(
		x.numel(),
		x.data_ptr<double>(),
		y.data_ptr<double>(),
		z.data_ptr<double>(),
		pists.data_ptr<double>(),
		xx.numel(),
		xx.data_ptr<double>(),
		yy.data_ptr<double>(),
		C1,
		C2,
		phase.data_ptr<double>()
	);

	return phase;
}


TORCH_LIBRARY_FRAGMENT(meilib, m) {
	m.def("rs", &rs);
}
