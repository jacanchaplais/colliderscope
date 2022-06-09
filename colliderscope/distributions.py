"""
``colliderscope.data``
======================

Data structures for formatting and pre-processing data in preparation
for plotting.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt
from scipy.stats import cauchy


class Align(Enum):
    CENTER = "center"
    LEFT = "left"


@dataclass
class Histogram:
    """Data structure for maintaining a constant memory, pre-binned
    histogram of particle multiplicities.
    
    Parameters
    ----------
    num_bins : int
        Number of bins to store multiplicities within.
    window : tuple[float, float]
        Range of x-axis to use for binning.
    _align : str
        Alignment of x-axis. Valid options are "left" or "center".
        If "left", `window` will be relative to `x=0`. If "center",
        `window` will be relative to `expected`.
        Default is left.
    expected : float, optional
        Expected value from theory for a peak.

    Attributes
    ----------
    num_bins : int
    expected : float
    window : tuple[float, float]
    align : Align
    x_range : tuple
        Low and high boundaries of the binned range.
    counts : ndarray
        Integer counts of values which entered each bin.
    bin_edges : ndarray
        Boundaries defining all bins.
    pdf : tuple[ndarray, ndarray]
        Bin centers and count density. May be used for bar chart plots.
    """
    num_bins: int
    window: Tuple[float, float]
    _align: str = "left"
    expected: Optional[float] = None
    counts: np.ndarray = field(init=False)
    bin_edges: np.ndarray = field(init=False)
    _total: int = field(init=False, repr=False, default=0)
    
    def __post_init__(self):
        self.align = Align(self._align)
        self.x_range = self.window
        if self.align is Align.CENTER:
            if self.expected is None:
                raise ValueError(
                        "You must specify `expected` in order to use "
                        "center alignment."
                        )
            self.x_range = (self.expected + self.window[0],
                            self.expected + self.window[1])
        self.counts = np.zeros(self.num_bins, dtype='<i')
        self.bin_edges = np.linspace(*self.x_range, self.num_bins + 1)
        self.bin_edges = self.bin_edges.squeeze()
    
    def update(self, val: float) -> None:
        """Records a new value to the binned counts."""
        self._total += 1
        idx = np.digitize(val, self.bin_edges) - 1
        if idx <= -1:
            return
        if idx >= self.num_bins:
            return
        self.counts[idx] += 1

    @property
    def pdf(self) -> Tuple[np.ndarray, np.ndarray]:
        """Provides tuple of bin centers and count density."""
        return ((self.bin_edges[1:] + self.bin_edges[:-1]) / 2.0,
                self.counts / self._total)


def breit_wigner_pdf(
        energy: npt.ArrayLike, mass_centre: float, width: float) -> np.ndarray:
    """Returns a Breit-Wigner probability density function.
    
    Parameters
    ----------
    energy : ndarray
        Energy domain over which the density is calculated.
    mass_center : float
        Central position of the mass peak.
    width : float
        The half-maximum-height width of the distribution.
    """
    bw: np.ndarray = cauchy.pdf(x=energy, loc=mass_centre, scale=(width / 2.0))
    return bw
