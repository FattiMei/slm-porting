#include <torch/extension.h>


void scale_float(float* data, unsigned long n, float scale) {
	for (unsigned long i = 0; i < n; ++i) {
		data[i] *= scale;
	}
}


void scale_inplace(torch::Tensor tensor, float scale) {
	TORCH_CHECK(tensor.device().is_cpu(), "CPU tensor expected");
	TORCH_CHECK(tensor.dtype() == torch::kFloat32, "float 32 expected");

	tensor = tensor.contiguous();
	auto n = tensor.numel();
	scale_float(tensor.data_ptr<float>(), n, scale);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
	m.def("scale_inplace", &scale_inplace, "Scale tensor in-place");
}
