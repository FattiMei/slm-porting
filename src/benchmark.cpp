#include <benchmark/benchmark.h>
#include "slm.hpp"
#include "utils.hpp"
#include "kernels.hpp"


constexpr int n      = 100;
constexpr int width  = 512;
constexpr int height = 512;


const SLM::Parameters parameters(width, height, Length(20.0, Unit::Millimeters), Length(15.0, Unit::Micrometers), Length(488.0, Unit::Nanometers));
Point3D spots[n];
double  pists[n];
double  phase[width * height];
std::complex<double> spot_fields[n];
double  p_phase_cache[n];

extern const int pupil_count;
extern const int pupil_indices[];
extern const std::pair<int, int> pupil_index_bounds[];


static void rs_upper_bound(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_upper_bound(n, spots, pists, phase, &parameters);
	}
}


static void rs_static_scheduling(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_static_scheduling(n, spots, pists, phase, &parameters);
	}
}


static void rs_dynamic_scheduling(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_dynamic_scheduling(n, spots, pists, phase, &parameters);
	}
}


static void rs_custom_scheduling(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_custom_scheduling(n, spots, pists, phase, &parameters);
	}
}


static void rs_branchless(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_branchless(n, spots, pists, phase, &parameters);
	}
}


static void rs_branch_delay_slot(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_branchless(n, spots, pists, phase, &parameters);
	}
}


static void rs_pupil_indices(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_pupil_indices(n, spots, pists, phase, pupil_count, pupil_indices, &parameters);
	}
}


static void rs_simd(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_pupil_indices_simd(n, spots, pists, phase, pupil_count, pupil_indices, &parameters);
	}
}


static void rs_static_index_bounds(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_static_index_bounds(n, spots, pists, phase, pupil_index_bounds, &parameters);
	}
}


static void rs_computed_index_bounds(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_computed_index_bounds(n, spots, pists, phase, &parameters);
	}
}


static void rs_cache_constants(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_cache_constants(n, spots, pists, phase, &parameters);
	}
}


static void rs_math_cache(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		rs_kernel_math_cache(n, spots, pists, phase, &parameters);
	}
}


static void gs_naive(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		gs_kernel_naive(n, spots, pists, spot_fields, phase, &parameters, 30);
	}
}


static void gs_pupil(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		gs_kernel_pupil(n, spots, pists, spot_fields, phase, pupil_count, pupil_indices, &parameters, 30);
	}
}


static void gs_openmp(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		gs_kernel_openmp(n, spots, pists, spot_fields, phase, pupil_count, pupil_indices, &parameters, 30);
	}
}


static void gs_atomic(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		gs_kernel_atomic(n, spots, pists, spot_fields, phase, pupil_count, pupil_indices, &parameters, 30);
	}
}


static void gs_cached(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		gs_kernel_cached(n, spots, pists, p_phase_cache, spot_fields, phase, &parameters, 30);
	}
}


static void gs_reordered(benchmark::State &state) {
	for (auto _ : state) {
		random_fill(n, pists, 0.0, 2.0 * M_PI, 1);
		gs_kernel_naive(n, spots, pists, spot_fields, phase, &parameters, 30);
	}
}


BENCHMARK(rs_upper_bound);
BENCHMARK(rs_static_scheduling);
BENCHMARK(rs_dynamic_scheduling);
BENCHMARK(rs_custom_scheduling);
BENCHMARK(rs_pupil_indices);
BENCHMARK(rs_simd);
BENCHMARK(rs_static_index_bounds);
BENCHMARK(rs_computed_index_bounds);
BENCHMARK(rs_branchless);
BENCHMARK(rs_branch_delay_slot);
BENCHMARK(rs_cache_constants);
BENCHMARK(rs_math_cache);
BENCHMARK(gs_naive);
BENCHMARK(gs_pupil);
BENCHMARK(gs_openmp);
BENCHMARK(gs_atomic);
BENCHMARK(gs_cached);
BENCHMARK(gs_reordered);
BENCHMARK_MAIN();
