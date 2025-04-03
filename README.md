# slm-porting
This project extends the work of [Paolo Pozzi](https://github.com/ppozzi) for *"generating 3D point cloud holograms, with phase only spatial light modulators, in real time through a GPU implementation of 5 algorithms (random superposition, Gerchberg-Saxton, weighted Gerchberg-Saxton, compressed sensing Gerchberg-Saxton, compressed sensing weighted Gerchberg-Saxton)"*

The original repositories are available at:
  * [SLM-3dPointCloud](https://github.com/ppozzi/SLM-3dPointCloud) for a real time NVidia GPU implementation
  * [compressive-sensing-Gerchberg-Saxton](https://github.com/csi-dcsc/compressive-sensing-Gerchberg-Saxton) for a CPU only implementation (if you have NVidia hardware, prefer the first one)

### Goals
I want to explore the vector programming paradigm, which is implemented by the `numpy` library. Describing the hologram algorithms as tensor operations allows the user to:
  * reduce the number of for loops written
  * keep writing in a high level language like python
  * **performance portability:** access different backends (openmp, cuda, opencl) with minimal modifications to the code

Since tensor operations are very well understood patterns, it's likely that off-the-shelf implementations will beat hand written kernels, this project will test this hypothesis.

### Tech stack
I propose to map the same algorithm into the following python libraries:
  * [numpy](https://numpy.org/)
  * [jax](https://docs.jax.dev/en/latest/) - should have support for GPU execution
  * [pyTorch](https://pytorch.org/) - should support a wide variety of backends, including non-NVidia GPUs
  * [Triton](https://triton-lang.org/main/index.html)

the performance will be compared against heavily hand optimized solutions in C++ and SYCL. Keep in mind that even in case of a tie the **hand optimized solutions have required engineering effort and specialized low-level knowledge**. The original algorithm designers may not be willing to invest time and money in optimizing a suboptimal solution, rather they'd want to explore the design space of algorithms given a rough estimation of the performance. This is something for which the chosen high-level libraries proposed have been designed and marketed.

## Architecture
All the algorithm proposed, namely:
  * rs
  * gs
  * wgs
  * csgs
  * wcsgs

share a common signature. They take as input a vector of points, the SLM configuration, and produce a 2D vector representing the input signals of the spatial light modulator to generate the desired hologram. Since every algorithm perform the same preprocessing on the inputs, I decided to encapsulate this logic in the base class `BaseSLM`.

Every possible algorithm would be tied to a particular derived class of `BaseSLM` an would expose the method `compute(points)`. Additional performance metric will be evaluated for every implementation.
