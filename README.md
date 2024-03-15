# slm-porting

## DISCLAIMER
I do not own the rights of:
 * csgs.py
 * example.py
 * slm_3dpointcloud.py


but I will modify them to suit my needs. I don't take any responsability for the use or misuse of the programs in this repository.


## Key people
Project developed under the supervision of:
 * Gianluca Palermo
 * Gianmarco Accordi
 * Davide Gadioli


## Proposed goals
 * Understand the kernels in `csgs.py` and produce an equivalent serial implementation in C/C++
 * Measure the correctness of the new version
 * Port the `slm_3dpointcloud.py` + `example.py` application to C/C++ while still using CUDA and openGL
 * Squeeze some performance out of the kernels
 * Get rid of CUDA and port the kernels to SYCL
 * Build a pipeline for automatically evaluate kernel performance and correctness


## Agreed goals
...
