# slm-porting
This project extends the work of [Paolo Pozzi](https://github.com/ppozzi) for *"generating 3D point cloud holograms, with phase only spatial light modulators, in real time through a GPU implementation of 5 algorithms (random superposition, Gerchberg-Saxton, weighted Gerchberg-Saxton, compressed sensing Gerchberg-Saxton, compressed sensing weighted Gerchberg-Saxton)"*

The original repositories are available at:
  * [SLM-3dPointCloud](https://github.com/ppozzi/SLM-3dPointCloud) for a real time NVidia GPU implementation
  * [compressive-sensing-Gerchberg-Saxton](https://github.com/csi-dcsc/compressive-sensing-Gerchberg-Saxton) for a CPU only implementation (if you have NVidia hardware, prefer the first one)

### Goals
I want to explore the vector programming paradigm, which is traditionally implemented in the `numpy` library. Describing the hologram algorithms as tensor operations allows the user to:
  * reduce the number of for loops written
  * keep writing in a high level language like python
  * enable **performance portability** across different backends (openmp threads, GPU acceleration with CUDA or OpenCL) with minimal modifications to the code

Since tensor operations are very well understood patterns, it's likely that off-the-shelf implementations will beat hand written kernels, this project will test this hypothesis.

### Tech stack
I propose to map the same algorithm into the following python libraries:
  * [numpy](https://numpy.org/)
  * [jax](https://docs.jax.dev/en/latest/) - should have support for GPU execution
  * [pyTorch](https://pytorch.org/) - should support a wide variety of backends, including non-NVidia GPUs
  * [Triton](https://triton-lang.org/main/index.html)

the performance will be compared against heavily hand optimized solutions in C++ and SYCL.

Hand optimized solutions have required considerable effort and specialized low-level knowledge and are not even guaranteed to be optimal. From the point of view of the original algorithm designers, such effort should be dedicated only to the most promising algorithms.
This project explores the use of high-level libraries to produce with reasonable effort performant implementations without extensive handmade optimizations. The goal is aligned with the design and marketing of the proposed libraries.
