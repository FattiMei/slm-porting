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


void primitive_rs_simd(const int64_t nspots,
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
	using simd_t = std::experimental::native_simd<double>;
	constexpr auto tile_size = simd_t::size();
	const auto tile_count = npupils / tile_size;
	const auto remainder = npupils % tile_size;

#pragma omp parallel for
	for (int tile = 0; tile < tile_count; ++tile) {
		simd_t vxx;
		simd_t vyy;
		simd_t acc_real(0.0);
		simd_t acc_imag(0.0);

		vxx.copy_from(xx + tile * tile_size, std::experimental::element_aligned);
		vyy.copy_from(yy + tile * tile_size, std::experimental::element_aligned);

		for (int spot_idx = 0; spot_idx < nspots; ++spot_idx) {
			const simd_t p_phase = C1 * (x[spot_idx]*vxx + y[spot_idx]*vyy)
			                     + C2 * z[spot_idx] * (vxx*vxx + vyy*vyy)
			                     + 2.0 * M_PI * pists[spot_idx];

			acc_real += std::experimental::cos(p_phase);
			acc_imag += std::experimental::sin(p_phase);
		}

		for (int i = 0; i < tile_size; ++i) {
			phase[tile * tile_size + i] = std::arg(
				std::complex<double>(
					acc_real[i],
					acc_imag[i]
				)
			);
		}
	}

	if (remainder > 0) {
		for (std::size_t i = npupils - remainder; i < npupils; ++i) {
			std::complex<double> acc(0.0, 0.0);

			for (int spot_idx = 0; spot_idx < nspots; ++spot_idx) {
				const double p_phase = C1 * (x[spot_idx]*xx[i] + y[spot_idx]*yy[i]) +
				                       C2 * z[spot_idx] * (xx[i]*xx[i] + yy[i]*yy[i]) +
				                       2.0 * M_PI * pists[spot_idx];

				acc += std::complex<double>(std::cos(p_phase), std::sin(p_phase));
			}

			// no normalization is required here
			phase[i] = std::arg(acc);
		}
	}
}


void primitive_rs_simulated_simd(const int64_t nspots,
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

	const auto tile_size = 4;
	const auto tile_count = npupils / tile_size;
	const auto remainder = npupils % tile_size;

#pragma omp parallel for
	for (int tile = 0; tile < tile_count; ++tile) {
		double vxx[tile_size];
		double vyy[tile_size];
		std::complex<double> acc[tile_size];

		for (int i = 0; i < tile_size; ++i) {
			vxx[i] = xx[tile * tile_size + i];
			vyy[i] = yy[tile * tile_size + i];
			acc[i] = std::complex<double>(0.0, 0.0);
		}

		for (int spot_idx = 0; spot_idx < nspots; ++spot_idx) {
			for (int i = 0; i < tile_size; ++i) {
				const double p_phase = C1 * (x[spot_idx]*vxx[i] + y[spot_idx]*vyy[i]) +
				                       C2 * z[spot_idx] * (vxx[i]*vxx[i] + vyy[i]*vyy[i]) +
				                       2.0 * M_PI * pists[spot_idx];

				acc[i] += std::complex<double>(std::cos(p_phase), std::sin(p_phase));
			}
		}

		for (int i = 0; i < tile_size; ++i) {
			phase[tile * tile_size + i] = std::arg(acc[i]);
		}
	}

	if (remainder > 0) {
		for (std::size_t i = npupils - remainder; i < npupils; ++i) {
			std::complex<double> acc(0.0, 0.0);

			for (int spot_idx = 0; spot_idx < nspots; ++spot_idx) {
				const double p_phase = C1 * (x[spot_idx]*xx[i] + y[spot_idx]*yy[i]) +
				                       C2 * z[spot_idx] * (xx[i]*xx[i] + yy[i]*yy[i]) +
				                       2.0 * M_PI * pists[spot_idx];

				acc += std::complex<double>(std::cos(p_phase), std::sin(p_phase));
			}

			// no normalization is required here
			phase[i] = std::arg(acc);
		}
	}
}


#define TORCH_IMPL_WRAPPER(fn)                         \
torch::Tensor fn(torch::Tensor x,                      \
                 torch::Tensor y,                      \
                 torch::Tensor z,                      \
                 torch::Tensor pists,                  \
                 torch::Tensor xx,                     \
                 torch::Tensor yy,                     \
                 double C1,                            \
                 double C2) {                          \
	torch::Tensor phase = torch::empty_like(xx);   \
	primitive_##fn(                                \
		x.numel(),                             \
		x.data_ptr<double>(),                  \
		y.data_ptr<double>(),                  \
		z.data_ptr<double>(),                  \
		pists.data_ptr<double>(),              \
		xx.numel(),                            \
		xx.data_ptr<double>(),                 \
		yy.data_ptr<double>(),                 \
		C1,                                    \
		C2,                                    \
		phase.data_ptr<double>()               \
	);                                             \
	return phase;                                  \
}


TORCH_IMPL_WRAPPER(rs);
TORCH_IMPL_WRAPPER(rs_simd);
TORCH_IMPL_WRAPPER(rs_simulated_simd);


TORCH_LIBRARY_FRAGMENT(meilib, m) {
	m.def("rs", &rs);
	m.def("rs_simd", &rs_simd);
	m.def("rs_simulated_simd", &rs_simulated_simd);
}
