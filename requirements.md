# Requirements for the C++ part
  * make the SLM class like the python one. Use a general Length class
  * try to incorporate `std::milli`, `std::micro`, ... in the Length constructor
  * make the RNG classes, with an interface to generate numbers, but also save the generator state

  * the `rs` algorithm must be generic on the data layout of the Point vector (use C++20 concepts)
    and on the rng strategy (use compile time polymorphism)
