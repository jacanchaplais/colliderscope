"""
``colliderscope``
=================

Visualise your high energy physics (HEP) data with colliderscope!
"""
import collections as cl
import collections.abc as cla
import dataclasses as dc
import gzip as gz
import itertools as it
import json
import operator as op
import textwrap as tw
import typing as ty
import warnings
from pathlib import Path

import colour
import graphicle as gcl
import more_itertools as mit
import numpy as np
import pandas as pd
import plotly.express as px
import webcolors
from pyvis.network import Network
from scipy.stats import cauchy

from . import base
from ._version import __version__, __version_tuple__

if ty.TYPE_CHECKING:
    from plotly.graph_objs._figure import Figure as PlotlyFigure
    from pyvis.network import IFrame


__all__ = [
    "__version__",
    "__version_tuple__",
    "Color",
    "NodePdgs",
    "color_range",
    "shower_dag",
    "eta_phi_scatter",
    "Histogram",
    "breit_wigner_pdf",
    "hist_to_bw_params",
    "histogram_barchart",
]


def _clip(val: float, min_: float = 0.0, max_: float = 1.0) -> float:
    """Clips ``val`` within the range given by ``max_`` and ``min_``."""
    return max(min(val, max_), min_)


@dc.dataclass
class Color:
    """Data type for representing colors.

    .. versionadded:: 0.2.0

    :group: data

    Parameters
    ----------
    value : tuple[float, float, float]
        The RGB values, in the interval [0, 1], representing the color.

    Attributes
    ----------
    value : tuple[float, float, float]
        The RGB values, in the interval [0, 1], representing the color.

    Notes
    -----
    ``Color`` instances may be mixed by simply adding them together,
    using the ``+`` operator.

    Raises
    ------
    TypeError
        If ``value`` is not passed a sequence type.
    ValueError
        If the length of ``value`` is not 3.
    """

    value: ty.Tuple[float, float, float] = (0.0, 0.0, 0.0)
    _data: colour.Color = dc.field(repr=False, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.value, cla.Sequence):
            raise TypeError(
                "Must pass length-3 tuple of floats, "
                f"{type(self.value)} is incompatible."
            )
        elif len(self.value) != 3:
            num = len(self.value)
            raise ValueError(
                "Size mismatch: must pass length-3 tuple of "
                f"floats, received sequence of length {num}."
            )
        self.value = tuple(self.value)
        self._data = colour.Color(rgb=self.value)

    @classmethod
    def from_name(cls, val: base.ColorNames) -> "Color":
        """Instantiate ``Color`` from CSS3 valid name.

        Parameters
        ----------
        val : str
            CSS3 valid name for color.

        Returns
        -------
        Color
            Instance from the color name.

        Raises
        ------
        ValueError
            If passed a string which is not a CSS3 valid color name.
        """
        hex_val = webcolors.name_to_hex(val)
        return cls(colour.Color(hex_val).rgb)

    @classmethod
    def from_hex(cls, val: str) -> "Color":
        """Instantiate ``Color`` from its hexadecimal code.

        Parameters
        ----------
        val : str
            Hexadecimal code for the color.

        Returns
        -------
        Color
            Instance from the color hexadecimal code.

        Raises
        ------
        ValueError
            If passed an invalid color hexadecimal code.
        """
        return cls(colour.Color(val).rgb)

    @property
    def rgb(self) -> ty.Tuple[float, float, float]:
        """Alias of ``Color.value``."""
        return self.value

    @property
    def hex(self) -> str:
        """Hexadecimal representation of the color."""
        return self._data.hex

    @property
    def web(self) -> str:
        """Browser-friendly representation of the color."""
        return self._data.get_web()

    @property
    def name(self) -> ty.Optional[base.ColorNames]:
        """CSS3 color name, if it exists. Otherwise, value is ``None``."""
        try:
            return webcolors.hex_to_name(self.hex)  # type: ignore
        except ValueError:
            return None

    def __iter__(self) -> ty.Iterator[float]:
        """Returns an iterator over the red-green-blue values."""
        return iter(self.rgb)

    def __add__(self, other: "Color") -> "Color":
        """Provides intrinsic way of mixing colors via ``+`` operator."""
        lum_average = (self._data.luminance + other._data.luminance) / 2.0
        add_rgb = map(op.add, self, other)
        add_rgb = map(_clip, add_rgb)
        add_color = colour.Color(rgb=tuple(add_rgb))
        add_color.set_luminance(_clip(lum_average, 0.4, 0.6))
        new_rgb = map(_clip, add_color.rgb)
        return self.__class__(tuple(new_rgb))


def color_range(start: Color, stop: Color, size: int) -> ty.Tuple[Color, ...]:
    """Construct a ``tuple`` of ``Color`` objects which interpolate
    smoothly between ``start`` and ``stop`` in ``size`` steps.

    .. versionadded:: 0.2.0

    :group: helpers

    Parameters
    ----------
    start, stop : Color
        Interval boundaries.
    size : int
        Number of ``Color`` instances to produce.

    Returns
    -------
    tuple[Color, ...]
        Colors interpolating from ``start`` to ``stop``.

    Examples
    --------
    Generating a range of ``Color`` objects between red and blue.

        >>> import colliderscope as csp
        ...
        >>> red = csp.Color.from_name("red")
        ... blue = csp.Color.from_name("blue")
        >>> csp.color_range(red, blue, 3)
        (Color(value=(1.0, 0.0, 0.0)),
         Color(value=(0.0, 1.0, 0.0)),
         Color(value=(0.0, 0.0, 1.0)))
        >>> csp.color_range(red, blue, 6)
        (Color(value=(1.0, 0.0, 0.0)),
         Color(value=(1.0, 0.8, 0.0)),
         Color(value=(0.3999999999999999, 1.0, 0.0)),
         Color(value=(0.0, 1.0, 0.40000000000000024)),
         Color(value=(0.0, 0.7999999999999998, 1.0)),
         Color(value=(0.0, 0.0, 1.0)))
    """
    colors = start._data.range_to(stop._data, size)
    rgbs = map(op.attrgetter("rgb"), colors)
    return tuple(map(Color, rgbs))


class NodePdgs(ty.NamedTuple):
    """``NamedTuple`` for holding the PDG names of the incoming and
    outgoing particles at a given interaction vertex. These are stored
    in ``list`` objects, and so may be mutated.

    :group: data

    .. versionadded:: 0.2.0
    """

    incoming: ty.List[str]
    outgoing: ty.List[str]


def shower_dag(
    output: ty.Union[str, Path],
    edges: ty.Iterable[ty.Tuple[int, int]],
    pdgs: ty.Iterable[int],
    masks: ty.Optional[
        ty.Union[
            ty.Sequence[ty.Iterable[bool]], ty.Mapping[str, ty.Iterable[bool]]
        ]
    ] = None,
    width: ty.Union[int, str] = "100%",
    height: ty.Union[int, str] = 750,
    notebook: bool = False,
    hover_width: int = 80,
) -> ty.Optional["IFrame"]:
    """HTML visualisation of a full event history as a directed acyclic
    graph (DAG).

    .. versionadded:: 0.2.0

    :group: figs

    Parameters
    ----------
    output : str or Path
        Location to which the HTML file should be written.
    edges : iterable[tuple[int, int]]
        COO format adjacency, referring to edges by the pairs of source
        and destination vertices, respectively.
    pdgs : iterable[int]
        PDG codes for each particle in the DAG.
    masks : sequence[iterable[bool]] or mapping[str, iterable[bool]]
        Optional masks grouping particles. Usually refers to
        the particle ancestries. Each group will be represented with
        a different color in the visualisation. Where ``masks`` overlap,
        their colors will be blended. Default is ``None``.
    width : str or int
        Width of the IFrame containing the visualisation. Default is
        ``"100%"``.
    height : str or int
        height of the IFrame containing the visualisation. Default is
        ``750``.
    notebook : bool
        If running in a Jupyter Notebook, passing ``True`` will render
        the visualisation as the output of the cell. Default is
        ``False``.
    hover_width : int
        Number of characters in node hover labels to allow per line,
        before text-wrapping. Default is ``80``.

    Returns
    -------
    IFrame or None
        If ``notebook`` parameter is passed ``True``, this function
        returns an IFrame containing the HTML visualisation.

    Raises
    ------
    ValueError
        If ``masks`` is passed nested mapping, *ie.* a dict-of-dicts,
        or a non-flat ``graphicle.MaskGroup`` instance.

    Notes
    -----
    Particles are represented by edges on this plot. Therefore,
    ``edges``, ``pdgs``, and the elements of ``masks`` must all have
    the same length.

    Dimensions of the IFrame given by ``width`` and ``height`` are
    inferred as pixels if passed integers. Any common browser unit of
    length may be used if passed as strings.

    ``masks`` must be "flat", *ie.* all the iterables of booleans must
    be at the top level of the passed data structure. If using the
    ``graphicle`` package, you may use ``MaskGroup.flatten()`` to ensure
    this is properly handled.

    Examples
    --------
    Generating and plotting a top pair production using ``showerpipe``
    and ``graphicle``, highlighting the descendants of the top quarks.

        >>> import showerpipe as shp
        ... import graphicle as gcl
        ... import colliderscope as csp
        ...
        >>> gen = shp.generator.PythiaGenerator(
        ...     "pythia-settings.cmnd", "top-events.lhe.gz", None
        ... )
        ... graph = gcl.Graphicle.from_event(next(gen))
        ... hier = gcl.select.hierarchy(graph)
        ... print(hier)
        MaskGroup(agg_op=OR)
        ├── t
        │   ├── b
        │   ├── W+
        │   │   ├── u
        │   │   ├── d~
        │   │   └── latent
        │   └── latent
        └── t~
            ├── b~
            ├── W-
            │   ├── d
            │   ├── u~
            │   └── latent
            └── latent
        >>> masks = hier.flatten("agg")  # flatten nested levels
        ... print(masks)
        MaskGroup(agg_op=OR)
        ├── t
        └── t~
        >>> csp.shower_dag("top_dag.html", graph.adj, graph.pdg, masks)
    """
    wrapper = tw.TextWrapper(
        width=hover_width, break_long_words=False, break_on_hyphens=False
    )
    if isinstance(width, int):
        width = f"{width}px"
    if isinstance(height, int):
        height = f"{height}px"
    if isinstance(output, str):
        output = Path(output)
    if not isinstance(pdgs, gcl.PdgArray):
        pdgs = gcl.PdgArray(pdgs)  # type: ignore
    if isinstance(edges, gcl.AdjacencyList):
        leaves = edges.leaves
        roots = edges.roots
        edge_tup = tuple(iter(edges))
    else:
        edge_tup = tuple(it.starmap(gcl.VertexPair, edges))
        _adj = gcl.AdjacencyList(edge_tup)  # type: ignore
        leaves = _adj.leaves
        roots = _adj.roots
    if masks is None:
        masks_iter = it.repeat((False,), len(edge_tup))
        pallette = (Color(),)
    else:
        pallette = color_range(
            Color.from_name("red"), Color.from_name("blue"), len(masks)
        )
        packed_masks = masks
        if isinstance(masks, cla.Mapping):
            if any(map(lambda v: isinstance(v, cla.Mapping), masks.values())):
                raise ValueError(
                    "Masks must be flat, ie. have no hierarchical nesting. "
                    "If you are using graphicle MaskGroups, try calling the "
                    "MaskGroup.flatten() method."
                )
            packed_masks = masks.values()
        masks_iter = zip(*packed_masks)
    nodes = list(mit.unique_everseen(it.chain.from_iterable(edge_tup)))
    list_factory = map(op.methodcaller("__call__"), it.repeat(list))
    list_pairs = zip(*(list_factory,) * 2, strict=True)
    node_pdgs = dict(zip(nodes, it.starmap(NodePdgs, list_pairs)))
    NUM_NODES = len(nodes)
    net = Network(height, width, directed=True, notebook=notebook)
    net.add_nodes(
        nodes,
        label=[" "] * NUM_NODES,
        size=[10] * NUM_NODES,
        color=["black"] * NUM_NODES,
    )
    particle_iter = zip(edge_tup, roots, leaves, pdgs.name, masks_iter)
    for edge, root, leaf, name, mask in particle_iter:
        node_pdgs[edge.src].outgoing.append(name)
        node_pdgs[edge.dst].incoming.append(name)
        out_vtx = net.node_map[edge.dst]
        in_vtx = net.node_map[edge.src]
        if root:
            in_vtx["label"] = "start"
            in_vtx["shape"] = "square"
            in_vtx["size"] = 20
        if leaf:
            out_vtx["label"] = name
            out_vtx["shape"] = "star"
        if any(mask):
            node_color = Color()
            for shade in it.compress(pallette, mask):
                node_color = node_color + shade
            in_vtx["color"] = out_vtx["color"] = node_color.web
        net.add_edge(edge.src, edge.dst)
    for node_id, node_dict in net.node_map.items():
        if node_dict["shape"] == "star":
            continue
        node_pdg = node_pdgs[node_id]
        pcls_in = sorted(node_pdg.incoming)
        pcls_out = sorted(node_pdg.outgoing)
        in_str = "<br />".join(wrapper.wrap(", ".join(pcls_in)))
        out_str = "<br />".join(wrapper.wrap(", ".join(pcls_out)))
        node_dict["title"] = (
            f"<b>Particles in ({len(pcls_in)}):</b><br />{in_str}"
            "<br /><br />"
            f"<b>Particles out ({len(pcls_out)}):</b><br />{out_str}"
        )
    if notebook is True:
        return net.show(str(output))
    return net.write_html(str(output), notebook=notebook)


@dc.dataclass
class Histogram:
    """Constant memory histogram data structure.

    .. versionchanged:: 0.2.0
       Migrated from ``data`` module.
       Renamed ``_align`` parameter to ``align``.

    :group: data

    Parameters
    ----------
    num_bins : int
        Number of bins to store multiplicities within.
    window : tuple[float, float]
        Range of x-axis to use for binning.
    align : {"left", "center"}
        Alignment of x-axis. Valid options are "left" or "center".
        If ``"left"``, ``window`` will be relative to ``x=0``. If
        ``"center"``, ``window`` will be centered relative to
        ``x=expected``. Default is ``"left"``.
    expected : float, optional
        Expected value for distribution's measure of location.

    Attributes
    ----------
    num_bins : int
        Number of bins to store multiplicities within.
    window : tuple[float, float]
        Range of x-axis values over which the histogram should track.
    align : {"left", "center"}
        Alignment of x-axis. Valid options are "left" or "center".
        If ``"left"``, ``window`` will be relative to ``x=0``. If
        ``"center"``, ``window`` will be centered relative to
        ``x=expected``. Default is ``"left"``.
    expected : float, optional
        Expected value for distribution's measure of location.
    x_range : tuple[float, float]
        Low and high boundaries of the binned range.
    counts : ndarray[int32]
        Integer counts of values which entered each bin.
    bin_edges : ndarray[float64]
        Sequence of values defining the boundaries of the bins.
    """

    num_bins: int
    window: ty.Tuple[float, float]
    align: ty.Literal["left", "center"] = "left"
    expected: ty.Optional[float] = None
    counts: base.IntVector = dc.field(init=False)
    bin_edges: base.DoubleVector = dc.field(init=False)
    _total: int = dc.field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self.x_range = self.window
        if self.align == "center":
            if self.expected is None:
                raise ValueError(
                    "Must set expected param to use align center."
                )
            self.x_range = (
                self.expected + self.window[0],
                self.expected + self.window[1],
            )
        self.counts = np.zeros(self.num_bins, dtype="<i4")
        self.bin_edges = np.linspace(*self.x_range, self.num_bins + 1)
        self.bin_edges = self.bin_edges.squeeze()

    def __eq__(self, other: "Histogram") -> bool:
        """Determines if two ``Histogram`` instances contain the same
        ``bin_edges``, ``counts``, and normalisation for ``pdf``.
        """
        eq = np.array_equal(self.bin_edges, other.bin_edges)
        eq = eq and np.array_equal(self.counts, other.counts)
        eq = eq and (self._total == other._total)
        return bool(eq)

    def __add__(self, other: "Histogram") -> "Histogram":
        """Merges two ``Histogram`` instances, by adding their
        ``counts`` and ``_total`` normalisation variables. Must have the
        same bins.
        """
        if not np.array_equal(self.bin_edges, other.bin_edges):
            raise ValueError("Histograms must have the same bins.")
        hist = self.__class__(
            self.num_bins, self.window, self.align, self.expected
        )
        hist.counts[...] = self.counts + other.counts
        hist._total = self._total + other._total
        return hist

    def update(self, val: float) -> None:
        """Records a new value to the binned counts."""
        idx = np.digitize(val, self.bin_edges) - 1
        if -1 < idx < self.num_bins:
            self.counts[idx] += 1
        self._total += 1

    def to_json(
        self,
        fname: ty.Union[str, Path],
        encoding: str = "utf-8",
    ) -> None:
        """Serialise and store ``Histogram`` object as a JSON file.

        Parameters
        ----------
        fname : str or Path
            Location on disk to save the data. If ``gzip`` compression
            is desired, use the ``".gz"`` extension.
        encoding : str
            Standard used to encode the text into binary formats.
            Default is ``"utf-8"``.
        """
        fname = Path(fname)
        if fname.suffix == ".gz":
            f = gz.open(fname, mode="wt", encoding=encoding)
        else:
            f = open(fname, mode="w", encoding=encoding)
        data = dc.asdict(self)
        for key, val in data.items():
            if isinstance(val, np.ndarray):
                data[key] = val.tolist()
        with f:
            json.dump(data, f)

    @classmethod
    def from_json(
        cls,
        fname: ty.Union[str, Path],
        encoding: str = "utf-8",
    ) -> "Histogram":
        """Instantiate a histogram from JSON file, encoded using the
        ``Histogram.to_json()`` method.

        Parameters
        ----------
        fname : str or Path
            Location on disk from which to load the data.
        encoding : str
            Standard used to encode the text into binary formats.
            Default is ``"utf-8"``.

        Returns
        -------
        Histogram
            Instance loaded from JSON stored data.
        """
        fname = Path(fname)
        if fname.suffix == ".gz":
            f = gz.open(fname, mode="rt", encoding=encoding)
        else:
            f = open(fname, encoding=encoding)
        with f:
            data = json.load(f)
        hist = cls(
            data.pop("num_bins"),
            tuple(data.pop("window")),
            data.pop("align"),
            data.pop("expected"),
        )
        hist._total = data.pop("_total")
        for key, val in data.items():  # remaining numpy arrays
            getattr(hist, key)[...] = val
        return hist

    @property
    def bin_width(self) -> float:
        """The width of all bins along the x-axis."""
        return abs((self.window[1] - self.window[0]) / self.num_bins)

    @property
    def pdf(self) -> ty.Tuple[base.DoubleVector, base.DoubleVector]:
        """Bin centers and count density. May be used for bar chart
        plots.
        """
        return (
            (self.bin_edges[1:] + self.bin_edges[:-1]) / 2.0,
            self.counts / (self._total * self.bin_width),
        )


def breit_wigner_pdf(
    energy: ty.Sequence[float], mass_centre: float, width: float
) -> base.DoubleVector:
    """Produces the non-relativistic *Breit-Wigner* probability density
    function for a particle of given ``width`` and ``mass_centre``.

    .. versionchanged:: 0.2.0
       Migrated from ``data`` module.

    :group: helpers

    Parameters
    ----------
    energy : sequence[float]
        Energy domain over which the density is calculated.
    mass_centre : float
        Expected value of the mass peak for the particle.
    width : float
        The half-maximum-height width of the distribution, or simply
        the 'width of the particle'.

    Returns
    -------
    ndarray[float64]
        Densities corresponding to passed sequence of ``energy`` values.

    Warns
    -----
    UserWarning
        If the resolution of the passed ``energy`` data is lower than
        the width of the distribution.
    """
    half_width = width / 2.0
    if (energy[1] - energy[0]) > half_width:
        warnings.warn(
            "Width of distribution is smaller than the energy bin width. "
            "This will result in unexpected behaviour. It is recommended "
            "to increase the resolution of the passed energy values."
        )
    return cauchy.pdf(x=energy, loc=mass_centre, scale=half_width)


def hist_to_bw_params(hist: Histogram) -> ty.Tuple[float, float]:
    """Parameters which fit a *Breit-Wigner* distribution to the passed
    ``Histogram``.

    :group: helpers

    .. versionadded:: 0.2.2

    Parameters
    ----------
    hist : Histogram
        Mass histogram for a reconstructed particle.

    Returns
    -------
    tuple[float, float]
        Mass centre and width of the Breit-Wigner peak, respectively.
    """
    e_iter_nested = it.starmap(
        lambda e, count: [e] * count.item(), zip(hist.pdf[0], hist.counts)
    )
    e_iter = it.chain.from_iterable(e_iter_nested)
    fit_kwargs = dict()
    if hist.expected is not None:
        fit_kwargs["loc"] = hist.expected
    mass_centre, half_width = cauchy.fit(list(e_iter), **fit_kwargs)
    return mass_centre, half_width * 2.0


def _pt_size(pt: base.DoubleVector) -> float:
    log_pt = np.log(pt)
    size = (log_pt - np.min(log_pt)) / (np.max(log_pt) - np.min(log_pt))
    size = size + 0.1
    return size


def _iterable_to_momentum(
    pmu: ty.Iterable[ty.Tuple[float, float, float, float]],
    count: ty.Optional[int] = None,
) -> gcl.MomentumArray:
    if isinstance(pmu, gcl.MomentumArray):
        return pmu
    kwargs = dict()
    if count is not None:
        kwargs["count"] = count
    if not isinstance(pmu, cla.Sized):
        pmu = np.fromiter(pmu, dtype=("<f8", 4), **kwargs)
    return gcl.MomentumArray(pmu)  # type: ignore


def _iterable_to_pdg(
    pdg: ty.Iterable[int],
    count: ty.Optional[int] = None,
) -> gcl.PdgArray:
    if isinstance(pdg, gcl.PdgArray):
        return pdg
    kwargs = dict()
    if count is not None:
        kwargs["count"] = count
    if not isinstance(pdg, cla.Sized):
        pdg = np.fromiter(pdg, dtype="<i4", **kwargs)
    return gcl.PdgArray(pdg)  # type: ignore


def eta_phi_scatter(
    pmu: ty.Iterable[ty.Tuple[float, float, float, float]],
    pdg: ty.Iterable[int],
    masks: ty.Optional[ty.Mapping[str, ty.Iterable[bool]]] = None,
    eta_max: ty.Optional[float] = 2.5,
    pt_min: ty.Optional[float] = 0.5,
    indices: ty.Optional[ty.Iterable[int]] = None,
) -> "PlotlyFigure":
    """Produces a scatter plot of particles over the
    :math:`\\eta-\\phi`, *ie.* pseudorapidity-azimuth, plane.

    :group: figs

    .. versionadded:: 0.2.0

    Parameters
    ----------
    pmu : iterable[tuple[float, float, float, float]]
        Four momenta of particles in basis :math:`(p_x, p_y, p_z, E)`.
    pdg : iterable[int]
        PDG particle identification codes.
    masks : mapping[str, iterable[bool]], optional
        Groups particles using key-value pairs, where the values are
        boolean masks identifying members of a group, and the keys are
        used in the plot legend. Default is ``None``.
    eta_max : float, optional
        Maximum cut-off for the pseudorapidity, :math:`\\eta`. Default
        is ``2.5``.
    pt_min : float, optional
        Minimum cut-off for the transverse momentum, :math:`p_T`.
        Default is ``0.5``.
    indices : iterable[int], optional
        Adds custom data indices to points on scatter plot. Mostly
        useful for keeping track of particles in ``Dash`` callbacks.
        Default is ``None``.

    Returns
    -------
    PlotlyFigure
        Interactive scatter plot over the :math:`\\eta-\\phi` plane.
    """
    pdg_ = _iterable_to_pdg(pdg)
    NUM_PCLS = len(pdg_)
    pmu_ = _iterable_to_momentum(pmu, count=NUM_PCLS)
    vis_mask = gcl.MaskGroup(agg_op="and")  # type: ignore
    if eta_max is not None:
        vis_mask["eta"] = np.abs(pmu_.eta) < eta_max
    if pt_min is not None:
        vis_mask["pt"] = pmu_.pt > pt_min
    if len(vis_mask) == 0:
        vis_mask["none"] = np.ones(NUM_PCLS, dtype="<?")
    NUM_VISIBLE = np.sum(vis_mask.data, dtype="<i4")
    vis_pmu = pmu_[vis_mask]
    vis_pdg = pdg_[vis_mask]
    df = pd.DataFrame(
        {
            "pt": vis_pmu.pt,
            "eta": vis_pmu.eta,
            "phi": vis_pmu.phi / np.pi,
            "name": vis_pdg.name,
            "size": _pt_size(vis_pmu.pt),
            "group": np.full(NUM_VISIBLE, "background", dtype=object),
        }
    )
    if masks is not None:
        for name, mask_ in masks.items():
            group_mask = np.fromiter(mask_, dtype="<?", count=NUM_PCLS)
            vis_group_mask = group_mask[vis_mask]
            df.loc[(vis_group_mask, "group")] = name
    custom_data = []
    if indices is not None:
        indices = np.fromiter(indices, dtype="<i4", count=NUM_PCLS)
        df["indices"] = indices[vis_mask]
        custom_data.append("indices")
    fig = px.scatter(
        df,
        x="eta",
        y="phi",
        color="group",
        symbol="group",
        size="size",
        hover_data=["size", "pt", "name"],
        custom_data=custom_data,
    )
    fig.update_yaxes(title_text=r"$\phi\;\; (\pi \text{ rad})$")
    fig.update_xaxes(title_text=r"$\eta$")
    return fig


def histogram_barchart(
    hist: Histogram,
    hist_label: str,
    title: str = "",
    x_label: str = "x",
    y_label: str = "Probability density",
    overlays: ty.Optional[ty.Dict[str, base.DoubleVector]] = None,
    opacity: float = 0.6,
) -> "PlotlyFigure":
    """Automatically convert a ``Histogram`` object, and optionally a
    number of ``overlays``, into a ``plotly`` bar chart / histogram.

    :group: figs

    Parameters
    ----------
    hist : Histogram
        Histogram data to render.
    hist_label : str
        Label for the histogram in the plot legend.
    title: str
        Heading for the plot. Default is ``""``.
    x_label, y_label : str
        Axis labels.
    overlays : dict[str, ndarray[float64]], optional
        Additional PDFs to overlay on the same plot. Keys are the labels
        displayed in the plot legend, and values are densities
        corresponding to the same x-bins of ``hist``. Default is
        ``None``.
    opacity : float
        Value in range [0, 1] setting how opaque bars are. If using
        many overlays, lower values may improve visual clarity. Default
        is ``0.6``.

    Returns
    -------
    PlotlyFigure
        Interactive ``plotly`` bar chart figure.
    """
    data_map = {x_label: hist.pdf[0], hist_label: hist.pdf[1]}
    if overlays is not None:
        overlays = cl.OrderedDict(overlays)
        overlays.update(data_map)
        data_map = overlays
    data = pd.DataFrame(data_map)
    data_map.pop(x_label)
    legend_labels = list(data_map.keys())
    fig = px.bar(
        data,
        x=x_label,
        y=legend_labels,
        labels={"x": x_label, "value": y_label},
        title=title,
        barmode="overlay",
        opacity=opacity,
    )
    return fig
