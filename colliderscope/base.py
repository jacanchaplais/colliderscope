"""
``colliderscope.base``
======================

Package-wide base classes, interfaces, and type definitions.
"""
import typing as ty

import numpy as np
import numpy.typing as npt
import typing_extensions as tyx

__all__ = [
    "BoolVector",
    "IntVector",
    "DoubleVector",
    "VoidVector",
    "AnyVector",
    "ColorNames",
]


BoolVector: tyx.TypeAlias = npt.NDArray[np.bool_]
IntVector: tyx.TypeAlias = npt.NDArray[np.int32]
DoubleVector: tyx.TypeAlias = npt.NDArray[np.float64]
VoidVector: tyx.TypeAlias = npt.NDArray[np.void]
AnyVector: tyx.TypeAlias = npt.NDArray[ty.Any]

HistValue: tyx.TypeAlias = ty.Union[DoubleVector, float, ty.Sequence[float]]


class HistogramLike(ty.Protocol):
    """Interface for 1D histogram data.

    Attributes
    ----------
    window : tuple[float, float]
        Domain of the histogram.
    num_bins : int
        Number of bins the domain is split into.
    counts : NDArray[int32]
        Number of values observed in each bin.
    total : int
        Total number of values which were attempted to bin. This is, in
        general, larger than the sum of the counts, as failed binning
        attempts are also tracked.
    """

    @property
    def window(self) -> ty.Tuple[float, float]:
        ...

    @property
    def num_bins(self) -> int:
        ...

    @property
    def counts(self) -> IntVector:
        ...

    @property
    def total(self) -> int:
        ...


ColorNames = ty.Literal[
    "aliceblue",
    "antiquewhite",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgrey",
    "darkgreen",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategrey",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "ghostwhite",
    "gold",
    "goldenrod",
    "grey",
    "green",
    "greenyellow",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgrey",
    "lightgreen",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategrey",
    "lightsteelblue",
    "lightyellow",
    "lime",
    "limegreen",
    "linen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orange",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "red",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategrey",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "whitesmoke",
    "yellow",
    "yellowgreen",
]
