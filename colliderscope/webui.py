"""
``colliderscope.webui``
=======================

Functions providing callbacks and parsing utilities to the web user
interface.
"""
import gzip as gz
import itertools as it
import json
import logging as lg
import operator as op
import tempfile as tf
import typing as ty

import graphicle as gcl
import numpy as np
import plotly.graph_objs as go
from dash import dcc
from dash import exceptions as dash_exceptions
from plotly.graph_objs._figure import Figure

from . import eta_phi_scatter, shower_dag


def gen_masks(graph_json: str) -> str:
    lg.info("Making masks")
    graph = json_to_gcl(graph_json)
    hier = gcl.select.hierarchy(graph)
    parton_composites = gcl.MaskGroup()
    clusters = gcl.select.clusters(graph, radius=1.0)
    for key, parton_mask in hier.items():
        if not isinstance(parton_mask, gcl.MaskGroup):
            raise RuntimeError("Parton mask is not a group.")
        constit_masks = parton_mask.recursive_drop().flatten("leaves")
        parton_composites[key] = clusters[list(constit_masks.keys())]
    lg.info(hier)
    return maskgroup_to_json(clusters)


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
    final_graph = graph[graph.final]
    cut = np.ones(shape=np.sum(graph.final), dtype=np.bool_)
    if eta_max is not None:
        cut &= abs(final_graph.pmu.eta) < eta_max
    if pt_min is not None:
        cut &= final_graph.pmu.pt > pt_min
    final_graph = final_graph[cut]
    masks = masks[cut]
    indices = np.flatnonzero(cut)
    lg.info("Updating figure.")
    lg.debug(f"{indices=}")
    fig = eta_phi_scatter(
        final_graph.pmu,
        final_graph.pdg,
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
    return json.dumps(graph.serialize(), separators=(",", ":"))


def maskgroup_to_json(masks: gcl.MaskGroup[gcl.MaskArray]) -> str:
    json_str = json.dumps(masks.serialize(), separators=(",", ":"))
    if json_str is None:
        raise RuntimeError("JSON serialisation failed.")
    lg.info("Serialising MaskGroup object as JSON.")
    return json_str


def json_to_gcl(json_str: str) -> gcl.Graphicle:
    data = json.loads(json_str)
    return gcl.Graphicle(
        adj=gcl.AdjacencyList(data.pop("adj")),
        particles=gcl.ParticleSet.from_numpy(**data),
    )


def json_to_maskgroup(json_str: str) -> gcl.MaskGroup[gcl.MaskArray]:
    masks_serialized = json.loads(json_str)
    masks = gcl.MaskGroup(masks_serialized)
    lg.info("Reading JSON string into MaskGroup object.")
    return masks


def select_mass(
    select: ty.Optional[ty.Dict[str, ty.Any]],
    graph_json: str,
    mask_json: str,
    eta_max: ty.Optional[float],
    pt_min: ty.Optional[float],
) -> str:
    graph = json_to_gcl(graph_json)
    root_masks_ = json_to_maskgroup(mask_json)
    select_mask = np.ones(np.sum(graph.final), dtype=np.bool_)
    if select is not None:
        select_mask[...] = False
        points: ty.List[ty.Dict[str, ty.Any]] = select["points"]
        has_custom = map(op.contains, points, it.repeat("customdata"))
        custom_points = it.compress(points, has_custom)
        custom = map(op.getitem, custom_points, it.repeat("customdata"))
        indices = map(op.getitem, custom, it.repeat(0))
        select_mask[list(indices)] = True
    root_pmu = {
        name: graph.pmu[graph.final][select_mask & mask]
        for name, mask in root_masks_.items()
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
    event_num: int,
    seed: int,
) -> ty.Tuple[ty.Dict[str, ty.Any], None]:
    if n_clicks is None:
        raise dash_exceptions.PreventUpdate
    graph = json_to_gcl(graph_json)
    masks = gcl.select.hierarchy(graph).recursive_drop().flatten("leaves")
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
