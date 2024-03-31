# Optimizations

## Loop invariants
From the profiling I see too many calls to linspace, in the double nested for loops that iterates over the pupil points the computation of the y coordinate is at the deepest level, but it's useless


## Filtering of pupil points
Approximately 21% of iterations don't compute any data because the corresponding pixel point is outside the pupil. The check for norm <= 1 is quite expensive (called WIDTH * HEIGHT times). I propose a solution to statically compute the indexes of pupil points:
 1. Compute pupil points indeces in the constructor of SLM class and store them in memory (BAD, but I still have to assess it)
 2. Compute in a clever way the range of x indeces for any given y index (CLEVER, but there could be numerical instability)


## Caching
Caching a column of `p_phase` 2D-array to maximize reuse (not on it yet)
