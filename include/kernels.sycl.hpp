#ifndef __KERNELS_SYCL_HPP__
#define __KERNELS_SYCL_HPP__


#include <CL/sycl.hpp>
#include <vector>
#include "slm.hpp"


using namespace cl::sycl;


void rs_kernel_naive(queue &q, const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters par);
void rs_kernel_pupil(queue &q, const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters par);
void rs_kernel_local(queue &q, const int n, const Point3D spots[], const double pists[], double phase[], const SLM::Parameters par);


#endif
