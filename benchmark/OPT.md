# Optimizations


## Loop invariants
From the profiling I see too many calls to linspace, in the double nested for loops that iterates over the pupil points the computation of the y coordinate is at the deepest level, but it's useless
*UPDATE*: the compiler is smart enough to optimize that code, therefore I won't do any transformations to keep the baseline algorithm easy to read


## Filtering of pupil points
Approximately 21% of iterations don't compute any data because the corresponding pixel point is outside the pupil. The check for norm <= 1 is called `WIDTH * HEIGHT` times. I propose a solution to statically compute the indexes of pupil points:
 1. Compute pupil points indices and store them in memory
 2. Store only the ranges for every row
 3. At every kernel call statically compute the range


These variants are implemented in:
 1. `rs_kernel_naive` (baseline)
 2. `rs_kernel_pupil_indices`
 3. `rs_kernel_pupil_index_bounds`
 4. `rs_kernel_static_index_bounds`


This optimization is important because pupil point filtering is common to all kernel calls, and it's also orthogonal to other optimizations (like caching). However, I'm not sure that GPUs will benefit from this approach, given that it's best to have balanced iterations, the benchmarks will give the answer.
For the analysis of benchmarks see https://github.com/FattiMei/benchmark/analysis.py


## Caching
Caching a column of `p_phase` 2D-array to maximize reuse (not on it yet)
