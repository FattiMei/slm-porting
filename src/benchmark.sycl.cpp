#include <benchmark/benchmark.h>
#include <vector>
#include "config.hpp"
#include "slm.hpp"
#include "utils.hpp"
#include "units.hpp"
#include <CL/sycl.hpp>
#include "kernels.sycl.hpp"


constexpr int n      = 100;
constexpr int width  = 512;
constexpr int height = 512;


const SLM::Parameters parameters(width, height, Length(20.0, Unit::Millimeters), Length(15.0, Unit::Micrometers), Length(488.0, Unit::Nanometers));
std::vector<Point3D> spots(n);
std::vector<double>  pists(n);
std::vector<double>  phase(width * height);


extern int pupil_count;
extern int pupil_indices[];


cl::sycl::queue q{cl::sycl::gpu_selector_v};
double  *device_pists = cl::sycl::malloc_device<double>(pists.size(), q);
Point3D *device_spots = cl::sycl::malloc_device<Point3D>(spots.size(), q);
double  *device_phase = cl::sycl::malloc_device<double>(phase.size(), q);
int     *device_pupil = cl::sycl::malloc_device<int>(pupil_count, q);
std::complex<double> *device_spot_fields = cl::sycl::malloc_device<std::complex<double>>(spots.size(), q);


// @IMPORTANT: this kernel has to be executed first because it loads all the necessary constant data for all kernels
static void rs_sycl_naive(benchmark::State &state) {
	q.memcpy(device_spots, spots.data(), spots.size() * sizeof(Point3D));
	q.memcpy(device_pupil, pupil_indices, pupil_count * sizeof(int));

	for (auto _ : state) {
		random_fill(n, pists.data(), 0.0, 2.0 * M_PI, 1);
		q.memcpy(device_pists, pists.data(), pists.size() * sizeof(double));
		q.wait();

		rs_kernel_naive(q, n, device_spots, device_pists, device_phase, parameters);
		q.memcpy(phase.data(), device_phase, phase.size() * sizeof(double));
		q.wait();
	}
}


static void rs_sycl_pupil(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists.data(), 0.0, 2.0 * M_PI, 1);
		q.memcpy(device_pists, pists.data(), pists.size() * sizeof(double));
		q.wait();

		rs_kernel_pupil(q, n, device_spots, device_pists, device_phase, pupil_count, device_pupil, parameters);
		q.memcpy(phase.data(), device_phase, phase.size() * sizeof(double));
		q.wait();
	}
}


static void rs_sycl_local(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists.data(), 0.0, 2.0 * M_PI, 1);
		q.memcpy(device_pists, pists.data(), pists.size() * sizeof(double));
		q.wait();

		rs_kernel_local(q, n, device_spots, device_pists, device_phase, pupil_count, device_pupil, parameters);
		q.memcpy(phase.data(), device_phase, phase.size() * sizeof(double));
		q.wait();
	}
}


static void gs_sycl_naive(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists.data(), 0.0, 2.0 * M_PI, 1);
		q.memcpy(device_pists, pists.data(), pists.size() * sizeof(double));
		q.wait();

		gs_kernel_naive(q, n, device_spots, device_pists, device_phase, parameters, 30);
		q.memcpy(phase.data(), device_phase, phase.size() * sizeof(double));
		q.wait();
	}
}


static void gs_sycl_pupil(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists.data(), 0.0, 2.0 * M_PI, 1);
		q.memcpy(device_pists, pists.data(), pists.size() * sizeof(double));
		q.wait();

		gs_kernel_pupil(q, n, device_spots, device_pists, device_phase, pupil_count, device_pupil, parameters, 30);
		q.memcpy(phase.data(), device_phase, phase.size() * sizeof(double));
		q.wait();
	}
}


static void gs_sycl_reduction(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists.data(), 0.0, 2.0 * M_PI, 1);
		q.memcpy(device_pists, pists.data(), pists.size() * sizeof(double));
		q.wait();

		gs_kernel_reduction(q, n, device_spots, device_pists, device_spot_fields, device_phase, pupil_count, device_pupil, parameters, 30);
		q.memcpy(phase.data(), device_phase, phase.size() * sizeof(double));
		q.wait();
	}
}


static void gs_sycl_block(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists.data(), 0.0, 2.0 * M_PI, 1);
		q.memcpy(device_pists, pists.data(), pists.size() * sizeof(double));
		q.wait();

		gs_kernel_block(q, n, device_spots, device_pists, device_spot_fields, device_phase, pupil_count, device_pupil, parameters, 30);
		q.memcpy(phase.data(), device_phase, phase.size() * sizeof(double));
		q.wait();
	}
}


// @TODO: set the time unit in the command invocation
BENCHMARK(rs_sycl_naive)->Unit(benchmark::kMillisecond);
BENCHMARK(rs_sycl_pupil)->Unit(benchmark::kMillisecond);
BENCHMARK(rs_sycl_local)->Unit(benchmark::kMillisecond);
BENCHMARK(gs_sycl_naive)->Unit(benchmark::kMillisecond);
BENCHMARK(gs_sycl_pupil)->Unit(benchmark::kMillisecond);
// BENCHMARK(gs_sycl_reduction)->Unit(benchmark::kMillisecond);
BENCHMARK(gs_sycl_block)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
