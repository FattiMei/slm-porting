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


cl::sycl::queue q{cl::sycl::gpu_selector_v};
double  *device_pists = cl::sycl::malloc_device<double>(pists.size(), q);
Point3D *device_spots = cl::sycl::malloc_device<Point3D>(spots.size(), q);
double  *device_phase = cl::sycl::malloc_device<double>(phase.size(), q);


static void rs_sycl_naive(benchmark::State &state) {
	q.memcpy(device_spots, spots.data(), spots.size() * sizeof(Point3D));

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

		rs_kernel_pupil(q, n, device_spots, device_pists, device_phase, parameters);
		q.memcpy(phase.data(), device_phase, phase.size() * sizeof(double));
		q.wait();
	}
}


static void rs_sycl_local(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists.data(), 0.0, 2.0 * M_PI, 1);
		q.memcpy(device_pists, pists.data(), pists.size() * sizeof(double));
		q.wait();

		rs_kernel_local(q, n, device_spots, device_pists, device_phase, parameters);
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


// @TODO: set the time unit in the command invocation
BENCHMARK(rs_sycl_naive)->Unit(benchmark::kMillisecond);
BENCHMARK(rs_sycl_pupil)->Unit(benchmark::kMillisecond);
BENCHMARK(rs_sycl_local)->Unit(benchmark::kMillisecond);
BENCHMARK(gs_sycl_naive)->Unit(benchmark::kMillisecond);
BENCHMARK_MAIN();
