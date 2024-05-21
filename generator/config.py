#ifndef __CONFIG__
#define __CONFIG__


#define OMP_NUM_THREADS constexpr int omp_num_threads
OMP_NUM_THREADS = 8;
#undef OMP_NUM_THREADS


#define RESOLUTION constexpr int resolution
RESOLUTION = 512;
#undef RESOLUTION


#endif
