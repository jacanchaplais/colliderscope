"""
``colliderscope.data``
======================

Data structures for formatting and pre-processing data in preparation
for plotting.
"""
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np


@dataclass
class Histogram:
    """Data structure for maintaining a constant memory, pre-binned
    histogram of particle multiplicities.
    
    Parameters
    ----------
    num_bins : int
        Number of bins to store multiplicities within.
    expected : float
        Expected value from theory for a peak.
    window_width : float
        The range around the expected value to bin.
        ie. x_range = expected +/- window_width.
    """
    num_bins: int
    expected: float
    window_width: float
    x_range: Tuple[float, float] = field(init=False)
    counts: np.ndarray = field(init=False)
    bin_edges: np.ndarray = field(init=False)
    
    def __post_init__(self):
        self.x_range = (self.expected - self.window_width,
                        self.expected + self.window_width)
        self.counts = np.zeros(self.num_bins, dtype='<i')
        self.bin_edges = np.linspace(*self.x_range, self.num_bins + 1)
    
    def update(self, val: float) -> None:
        """Records a new value to the binned counts."""
        idx = np.digitize(val, self.bin_edges) - 1
        if idx == -1:
            return
        if idx == self.num_bins:
            return
        self.counts[idx] += 1
