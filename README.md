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


## Development steps
`csgs.py`
 * Remove the indeterminism in the kernel invocation (fixed seed)
 * Annotate shapes in all data used
 * Produce regression tests with adequate makefile rules
 * (HARD) setup remote pipeline for testing and reporting results

`slm_3dpointcloud.py`
 * Build a virtual environment that takes care of the many dependencies (virtual environment or maybe Docker??)
 * Search on the internet the way to properly install pycuda with openGL support
 * Run on an NVIDIA machine the script once
 * Separate CUDA code in its own file
 * Solve kernel parameter hell


## Critical path
Understand the relation between `csgs.py` and `slm_3dpointcloud.py`. It seems that the latter builds on top of the basic kernels, but implements them in CUDA and does obfuscated parameter passing (we'll figure it out).
