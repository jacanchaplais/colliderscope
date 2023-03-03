import logging as lg
import sys
from pathlib import Path

import click

DATA_DIR = Path("/scratch/jlc1n20/data/jacan/data/mg_runs/boosted/")
SRC_DIR = Path(__file__).parent
ROOT_DIR = SRC_DIR.parent


def next_event(_):
    import graphicle as gcl

    from . import webui

    if "gen" not in globals():
        raise RuntimeError("Generator not initialised.")
    count, event = next(gen)
    lg.info(f"Generated event {count}")
    lg.info(event)
    graph = gcl.Graphicle.from_event(event)
    return (
        webui.gcl_to_json(graph),
        count,
    )


@click.command()
@click.argument(
    "lhe-path", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    "--log-level",
    type=click.Choice(
        ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"), case_sensitive=False
    ),
    default="WARNING",
    show_default=True,
)
@click.option("-p", "--port", type=click.INT, default=8050, show_default=True)
def main(lhe_path: Path, log_level: str, port: int) -> None:
    """jeti: jets made interactive."""
    import showerpipe as shp
    from dash import Dash, Input, Output, dcc, html

    from . import webui

    lg.basicConfig(
        filename=ROOT_DIR / "server.log",
        encoding="utf-8",
        level=getattr(lg, log_level.upper()),
    )
    app = Dash(__name__)
    CONFIG_PATH = SRC_DIR / "../pythia-settings.cmnd"
    splits = shp.lhe.split(lhe_path, 1000)
    lhe_data = next(splits)
    global gen
    gen_ = shp.generator.PythiaGenerator(CONFIG_PATH, lhe_data, None)
    seed = int(gen_.config["Random"]["seed"])
    lg.info("Initialised PythiaGenerator")
    lg.debug(gen_)
    gen = enumerate(gen_)
    app.callback(
        Output("graph-data", "data"),
        Output("event-num", "data"),
        Input("next-event", "n_clicks"),
    )(next_event)
    app.callback(
        Output("mask-data", "data"),
        Output("root-masks", "data"),
        Input("graph-data", "data"),
        Input("parton-exp", "value"),
    )(webui.gen_masks)
    app.callback(
        Output("figure", "figure"),
        Input("graph-data", "data"),
        Input("mask-data", "data"),
        Input("eta-max", "value"),
        Input("pt-min", "value"),
    )(webui.filter_data)
    app.callback(
        Output("masses", "children"),
        Input("figure", "selectedData"),
        Input("graph-data", "data"),
        Input("root-masks", "data"),
        Input("eta-max", "value"),
        Input("pt-min", "value"),
    )(webui.select_mass)
    app.callback(
        Output("download-dag", "data"),
        Output("save-dag", "n_clicks"),
        Input("save-dag", "n_clicks"),
        Input("graph-data", "data"),
        Input("mask-data", "data"),
        Input("event-num", "data"),
        Input("seed", "data"),
        prevent_initial_call=True,
    )(webui.download_dag)
    app.callback(
        Output("download-event", "data"),
        Output("save-event", "n_clicks"),
        Input("save-event", "n_clicks"),
        Input("graph-data", "data"),
        Input("event-num", "data"),
        Input("seed", "data"),
        prevent_initial_call=True,
    )(webui.download_event)
    app.layout = html.Div(
        children=[
            html.H1(children="jeti"),
            html.Div(id="masses"),
            dcc.Graph(id="figure", mathjax=True),
            html.H3("Maximum pseudorapidity:"),
            dcc.Slider(id="eta-max", min=0.1, max=10.0, value=2.5),
            html.H3("Minimum transverse momentum:"),
            dcc.Slider(id="pt-min", min=0.0, max=10.0, value=0.5, step=0.25),
            html.Div(
                id="exponents",
                children=[
                    dcc.Input(id="parton-exp", type="number", value=-0.1),
                    dcc.Input(id="hadron-exp", type="number", value=-0.1),
                ],
            ),
            html.Div(
                id="controls",
                children=[
                    html.Button(
                        id="next-event", n_clicks=0, children="Next event"
                    ),
                    html.Button(id="save-dag", children="Save DAG"),
                    dcc.Download(id="download-dag"),
                    html.Button(id="save-event", children="Download Event"),
                    dcc.Download(id="download-event"),
                ],
            ),
            dcc.Store(id="graph-data"),
            dcc.Store(id="mask-data"),
            dcc.Store(id="root-masks"),
            dcc.Store(id="event-num"),
            dcc.Store(id="seed", data=seed),
        ]
    )
    lg.info("Server started.")
    app.run(debug=True, port=str(port))


if __name__ == "__main__":
    sys.exit(main())
