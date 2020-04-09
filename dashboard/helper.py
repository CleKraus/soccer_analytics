# -*- coding: utf-8 -*-

import dash_bootstrap_components as dbc
import base64
import os
import dash_html_components as html
import ruamel.yaml
import plotly.graph_objects as go


def create_navigation_bar():

    with open("config.yml", "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

    path_assets = config["assets"]["path"]

    # this example that adds a logo to the navbar brand
    navbar = dbc.Navbar(
        dbc.Container(
            [
                html.A(
                    # Use row and col to control vertical alignment of logo / brand
                    dbc.Row(
                        [
                            dbc.Col(
                                html.Img(
                                    id="bundesliga",
                                    src="data:image/png;base64,{}".format(
                                        base64.b64encode(open(
                                            os.path.join(path_assets, config["assets"]["img_bundesliga"]),
                                            "rb", ).read()).decode()
                                    ), height="40px"
                                )
                            ),
                            dbc.Col(dbc.NavbarBrand("Soccer Analysis", className="ml-2")),
                        ],
                        align="left",
                        no_gutters=True,
                    ),
                ),
            ]
        ),
        color="dark",
        dark=True,
        className="mb-5",
    )
    return navbar


def create_empty_field(below=False):
    rad_circle = 9.15
    len_field = 105
    wid_field = 68

    circle_y = (wid_field / 2 - rad_circle) * 100 / wid_field
    circle_x = (len_field / 2 - rad_circle) * 100 / len_field

    y_box = ((wid_field - 40.32) / 2) * 100 / wid_field

    layout = go.Layout(
        plot_bgcolor="rgba(0,255,112,1)",
        xaxis=dict(range=[0, 100],
                   showgrid=False,
                   showticklabels=False),
        yaxis=dict(range=[0, 100],
                   showgrid=False,
                   showticklabels=False),
    )

    # Create traces
    fig = go.Figure(layout=layout)

    fig.add_shape(
        # Line Vertical
        dict(
            layer="below",
            type="line",
            x0=50,
            y0=0,
            x1=50,
            y1=100,
            line=dict(
                color="white",
                width=2
            )))

    # draw left box
    x_vals = [0, (16 / len_field) * 100, (16 / len_field) * 100, 0]
    y_vals = [100 - y_box, 100 - y_box, y_box, y_box]

    for i in range(len(x_vals) - 1):
        fig.add_shape(
            # Line Vertical
            dict(
                layer="below",
                type="line",
                x0=x_vals[i],
                y0=y_vals[i],
                x1=x_vals[i + 1],
                y1=y_vals[i + 1],
                line=dict(
                    color="white",
                    width=2
                )))

    # draw right box
    x_vals = [100, (1 - 16 / len_field) * 100, (1 - 16 / len_field) * 100, 100]
    y_vals = [100 - y_box, 100 - y_box, y_box, y_box]

    for i in range(len(x_vals) - 1):
        fig.add_shape(
            # Line Vertical
            dict(
                layer="below",
                type="line",
                x0=x_vals[i],
                y0=y_vals[i],
                x1=x_vals[i + 1],
                y1=y_vals[i + 1],
                line=dict(
                    color="white",
                    width=2
                )))

        # add left penalty point
    size_point = 0.5
    pen_point = (11, 50)
    x_vals = [pen_point[0] - size_point / len_field * 100, pen_point[0] + size_point / len_field * 100]
    y_vals = [pen_point[1] - size_point / wid_field * 100, pen_point[1] + size_point / wid_field * 100]

    fig.add_shape(
        # unfilled circle
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=x_vals[0],
            y0=y_vals[0],
            x1=x_vals[1],
            y1=y_vals[1],
            line_color="white",
            fillcolor="white",
            layer="below"
        )
    )

    # add right penalty point
    size_point = 0.5
    pen_point = ((1 - 11 / len_field) * 100, 50)
    x_vals = [pen_point[0] - size_point / len_field * 100, pen_point[0] + size_point / len_field * 100]
    y_vals = [pen_point[1] - size_point / wid_field * 100, pen_point[1] + size_point / wid_field * 100]

    fig.add_shape(
        # unfilled circle
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=x_vals[0],
            y0=y_vals[0],
            x1=x_vals[1],
            y1=y_vals[1],
            line_color="white",
            fillcolor="white",
            layer="below"
        )
    )

    # add middle point
    size_point = 0.5
    pen_point = (50, 50)
    x_vals = [pen_point[0] - size_point / len_field * 100, pen_point[0] + size_point / len_field * 100]
    y_vals = [pen_point[1] - size_point / wid_field * 100, pen_point[1] + size_point / wid_field * 100]

    fig.add_shape(
        # unfilled circle
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=x_vals[0],
            y0=y_vals[0],
            x1=x_vals[1],
            y1=y_vals[1],
            line_color="white",
            fillcolor="white",
            layer="below"
        )
    )

    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[50],
        y=[50],
        mode="markers",
        marker=dict(
            color='white',
            size=7
        ),
    ))

    # Add circles
    if below:
        fig.add_shape(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=circle_x,
                y0=circle_y,
                x1=100 - circle_x,
                y1=100 - circle_y,
                line_color="white",
                layer="below"
            )
        )
    else:
        fig.add_shape(
            dict(
                type="circle",
                xref="x",
                yref="y",
                x0=circle_x,
                y0=circle_y,
                x1=100 - circle_x,
                y1=100 - circle_y,
                line_color="white",
                layer="below"
            )
        )

    fig.update_layout(
        autosize=False,
        width=len_field * 8,
        height=wid_field * 8)

    return fig

