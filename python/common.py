from typing import NamedTuple


'''
Those metrics are useful to domain experts to characterize
the quality of the proposed algorithms.

Each backend provides its own implementation even if I could transfer
all the data back to CPU and then use only the numpy implementation.

The redundancy is justified in the case there are far better backends: it
will be more convenient to profile the quality of algorithms in the best
backend, assuming an in-depth data collection on quality metrics is required.
From the paper I have been reading, it seems that some form of distribution
analysis has been used.
'''
class QualityMetrics(NamedTuple):
    efficiency: float
    uniformity: float
    variance:   float
