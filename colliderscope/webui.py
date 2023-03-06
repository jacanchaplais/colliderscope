"""
``colliderscope.webui``
=======================

Functions providing callbacks and parsing utilities to the web user
interface.
"""
import gzip as gz
import itertools as it
import logging as lg
import operator as op
import tempfile as tf
import typing as ty

import graphicle as gcl
import numpy as np
import numpy.typing as npt
import pandas as pd
import plotly.graph_objs as go
from dash import dcc
from dash import exceptions as dash_exceptions
from plotly.graph_objs._figure import Figure

from . import eta_phi_scatter, shower_dag


def gen_masks(
    graph_json: str,
    parton_exp: float,
) -> ty.Tuple[str, ty.Dict[str, npt.NDArray[np.bool_]]]:
    graph = json_to_gcl(graph_json)
    hier_ = gcl.select.hierarchy(graph)
    hier = gcl.select.partition_descendants(graph, hier_, parton_exp)
    lg.info(hier)
    for parton in hier.values():
        parton.pop("latent")
    root_masks = {name: mask.data for name, mask in hier.items()}
    masks = gcl.select.leaf_masks(hier)
    lg.info("Created partonic masks over the event")
    return maskgroup_to_json(masks), root_masks


def filter_data(
    graph_json: str,
    masks_json: str,
    eta_max: ty.Optional[float],
    pt_min: ty.Optional[float],
) -> Figure:
    graph = json_to_gcl(graph_json)
    masks = json_to_maskgroup(masks_json)
    hard_mask = graph.hard_mask
    del hard_mask["incoming"]
    hard_pcls = graph[hard_mask]
    leaf_filter = np.in1d(hard_pcls.pdg.name, list(masks.keys()))
    hard_pcls = hard_pcls[leaf_filter]
    cut = graph.final.data
    if eta_max is not None:
        cut = cut & (abs(graph.pmu.eta) < eta_max)
    if pt_min is not None:
        cut = cut & (graph.pmu.pt > pt_min)
    graph = graph[cut]
    masks = masks[cut]
    indices = np.flatnonzero(cut)
    lg.info("Updating figure.")
    lg.debug(f"{indices=}")
    fig = eta_phi_scatter(
        graph.pmu,
        graph.pdg,
        masks,  # type: ignore
        eta_max=None,
        pt_min=None,
        indices=indices,
    )
    overlay = go.Scatter(
        x=hard_pcls.pmu.eta,
        y=hard_pcls.pmu.phi / np.pi,
        text=hard_pcls.pdg.name,
        mode="markers",
        name="parents",
        marker=dict(size=10),
    )
    fig.add_trace(overlay)
    return fig


def gcl_to_json(graph: gcl.Graphicle) -> str:
    df = pd.DataFrame()
    df["final"] = graph.final.data
    df["pdg"] = graph.pdg.data
    df["status"] = graph.status.data
    df["in"] = graph.edges["in"]
    df["out"] = graph.edges["out"]
    df["color"] = graph.color.data["color"]
    df["anticolor"] = graph.color.data["anticolor"]
    for coord in "xyze":
        df[coord] = graph.pmu.data[coord]
    json_str = df.to_json(date_format="iso", orient="split")
    if json_str is None:
        raise RuntimeError("JSON serialisation failed.")
    lg.info("Serialising Graphicle object as JSON.")
    return json_str


def maskgroup_to_json(masks: gcl.MaskGroup[gcl.MaskArray]) -> str:
    json_str = pd.DataFrame(masks.dict).to_json(
        date_format="iso", orient="split"
    )
    if json_str is None:
        raise RuntimeError("JSON serialisation failed.")
    lg.info("Serialising MaskGroup object as JSON.")
    return json_str


def json_to_gcl(json_str: str) -> gcl.Graphicle:
    df = pd.read_json(json_str, orient="split")
    graph = gcl.Graphicle.from_numpy(
        pdg=df["pdg"].values,  # type: ignore
        final=df["final"].values,  # type: ignore
        status=df["status"].values,  # type: ignore
        pmu=df[list("xyze")].values,  # type: ignore
        color=df[["color", "anticolor"]].values,  # type: ignore
        edges=df[["in", "out"]].values,  # type: ignore
    )
    lg.info("Reading JSON string into Graphicle object.")
    return graph


def json_to_maskgroup(json_str: str) -> gcl.MaskGroup[gcl.MaskArray]:
    df = pd.read_json(json_str, orient="split")
    assert isinstance(df, pd.DataFrame), "JsonReader instead of DataFrame."
    masks = gcl.MaskGroup.from_numpy_structured(df.to_records(index=False))
    lg.info("Reading JSON string into MaskGroup object.")
    return masks


def select_mass(
    select: ty.Optional[ty.Dict[str, ty.Any]],
    graph_json: str,
    root_masks: ty.Dict[str, ty.List[np.bool_]],
    eta_max: ty.Optional[float],
    pt_min: ty.Optional[float],
) -> str:
    graph = json_to_gcl(graph_json)
    root_masks_ = {
        name: gcl.MaskArray(mask) for name, mask in root_masks.items()
    }
    select_mask = np.ones_like(graph.final.data, dtype="<?")
    if select is not None:
        select_mask[...] = False
        points: ty.List[ty.Dict[str, ty.Any]] = select["points"]
        has_custom = map(op.contains, points, it.repeat("customdata"))
        custom_points = it.compress(points, has_custom)
        custom = map(op.getitem, custom_points, it.repeat("customdata"))
        indices = map(op.getitem, custom, it.repeat(0))
        select_mask[list(indices)] = True
    vis_mask: gcl.MaskGroup[gcl.MaskArray] = gcl.MaskGroup(agg_op="and")
    vis_mask["select"] = select_mask
    vis_mask["final"] = graph.final
    root_pmu = {
        name: graph.pmu[vis_mask & mask] for name, mask in root_masks_.items()
    }
    if eta_max is not None:
        for name, pmu in root_pmu.items():
            root_pmu[name] = pmu[np.abs(pmu.eta) < eta_max]
    if pt_min is not None:
        for name, pmu in root_pmu.items():
            root_pmu[name] = pmu[pmu.pt > pt_min]
    root_mass = (
        (name, gcl.calculate.combined_mass(pmu))
        for name, pmu in root_pmu.items()
    )
    mass_strs = it.starmap("{}: {:.10}".format, root_mass)
    mass_descr_str = ", ".join(mass_strs)
    return mass_descr_str


def download_dag(
    n_clicks: ty.Optional[int],
    graph_json: str,
    masks_json: str,
    event_num: int,
    seed: int,
) -> ty.Tuple[ty.Dict[str, ty.Any], None]:
    if n_clicks is None:
        raise dash_exceptions.PreventUpdate
    masks = json_to_maskgroup(masks_json)
    graph = json_to_gcl(graph_json)
    with tf.TemporaryDirectory() as temp_dir:
        shower_dag(f"{temp_dir}/dag.html", graph.adj, graph.pdg, masks)
        return (
            dcc.send_file(
                f"{temp_dir}/dag.html", f"dag-e{event_num:02}-s{seed}.html"
            ),
            None,
        )


def download_event(
    n_clicks: ty.Optional[int],
    graph_json: str,
    event_num: int,
    seed: int,
) -> ty.Tuple[ty.Dict[str, ty.Any], None]:
    if n_clicks is None:
        raise dash_exceptions.PreventUpdate
    with tf.NamedTemporaryFile() as f:
        with gz.open(f, "wb") as gf:
            gf.write(graph_json.encode())
        f.seek(0)
        return (
            dcc.send_file(f.name, f"data-e{event_num:02}-s{seed}.json.gz"),
            None,
        )
