import pandas as pd
import numpy as np
import os
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
import ruamel.yaml
import plotly.graph_objects as go

import dashboard.helper as helper


def get_layout(match_video):

    layout_match_video = html.Div(
        children=[
            dbc.Row([
                dcc.Graph(
                    id="match_video",
                    figure=match_video
                )
            ], justify="center")
        ]
    )
    return layout_match_video


def plotly_figure_game():

    with open("config.yml", "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

    path = config["data"]["path"]
    fname_pos_data = config["data"]["positional_data"]

    df = pd.read_csv(os.path.join(path, fname_pos_data))

    # notice that player 0 is always the ball
    df["color"] = np.where(df["player"] == 0, "white", np.where(df["team"] == "attack", "red", "blue"))
    df["size"] = np.where(df["player"] == 0, 15, 10)

    frame_ids = sorted(list(df["frame"].unique()))

    frames=[go.Frame(
            data=[go.Scatter(
                x=np.array(df[df["frame"]==k]["x"]),
                y=np.array(df[df["frame"]==k]["y"]),
                mode="markers",
                marker=dict(color=np.array(df[df["frame"]==k]["color"]), size=12))])

            for k in frame_ids]

    fig = helper.create_empty_field(below=True)

    fig.update_layout(updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None,
                            {"frame": {"duration": 50, "redraw": False}, "fromcurrent": True}, ]),
                 dict(label="Pause",
                      method="animate",
                      args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}])
                 ])]
    )

    fig.frames = frames
    return fig

