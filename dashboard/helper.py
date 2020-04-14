# -*- coding: utf-8 -*-

# import packages
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

