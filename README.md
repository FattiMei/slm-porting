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
 * Measure the correctness of the new version (but keep in mind that C++ and numpy random generators are different)
 * Port the `slm_3dpointcloud.py` + `example.py` application to C/C++ while still using CUDA and openGL
 * Squeeze some performance out of the kernels
 * Get rid of CUDA and port the kernels to SYCL
 * Build a pipeline for automatically evaluate kernel performance and correctness


## Agreed goals
...


## Development steps
`csgs.py`
 * Remove the indeterminism in the kernel invocation (fixed seed)
 * Annotate shapes in all data used (done for rs, others still in progress)
 * Produce regression tests with adequate makefile rules (partially solved by `regression.py`)
 * (HARD) setup remote pipeline for testing and reporting results

`slm_3dpointcloud.py`
 * Build a virtual environment that takes care of the many dependencies (virtual environment or maybe Docker??)
 * Search on the internet the way to properly install pycuda with openGL support
 * Run on an NVIDIA machine the script once
 * Separate CUDA code in its own file
 * Solve kernel parameter hell (see Critical path section)

The oldest file also gave me the most detailed information about the kernels, and producing the serial version makes sense of the algorithm.


## Critical path
Understand the relation between `csgs.py` and `slm_3dpointcloud.py`. It seems that the latter builds on top of the basic kernels, but implements them in CUDA and does obfuscated parameter passing (we'll figure it out).
The parameter passing was an emergent behaviour, every kernel needs pretty much the same data and it's often run-time constants (e.g wavelength, pitch, focal length...), it is advisable to pass this information in a pointer to a struct. I will still need some benchmark to assess the performance difference for this new parameter passing.


## Testing procedure
 1. Strip away the rng from the python kernels
 2. Output the random data from the cpp application
 3. Load such data into python kernels and compare the two implementations

Unfortunately this approach falls short for compressive kernels (csgs and wcsgs) because the process of sampling pupil points is hard to port:
 * python computes all the pupil points, shuffles them and then take chunks for computation
 * cpp decides to accept or decline every pupil point based on a Bernoulli outcome

Since I completely removed the computation of all pupil points for performance reasons (big memory footprint), I won't port the python kernels as-is. To make tests like all the other kernels I would have to modify the python implementation but such changes couldn't be tested against the original implementation (I'm talking about deterministic tests here).
For these reasons the testing for the csgs and wcsgs kernels will be only visual, until I find some weaker tests.


## Profiling procedure
I propose an hybrid strategy:
 1. `gprof` profiling for high level information, to know where the kernel spend time
 2. google benchmark for assessing the effects of optimizations in kernels
