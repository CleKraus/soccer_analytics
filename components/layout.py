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
                            dbc.Col(dbc.NavbarBrand("Bundesliga Analysis", className="ml-2")),
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
    """
    Function draws an empty soccer field using plotly
    :param below:
    :return:
    """

    # radius of the middle circle
    rad_circle = 9.15
    # length of the field (in m)
    len_field = 105
    # width of the field (in m)
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
    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[50, 50],
        y=[100, 0],
        mode='lines',
        line=dict(color='white', width=2)))

    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[0, (16 / len_field) * 100, (16 / len_field) * 100, 0],
        y=[100 - y_box, 100 - y_box, y_box, y_box],
        mode='lines',
        line=dict(color='white', width=2),

    ))

    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[100, (1 - 16 / len_field) * 100, (1 - 16 / len_field) * 100, 100],
        y=[100 - y_box, 100 - y_box, y_box, y_box],
        mode='lines',
        line=dict(color='white', width=2),
    ))

    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[11 / len_field * 100],
        y=[50],
        mode="markers",
        marker=dict(
            color='white',
            size=7
        ),
    ))

    fig.add_trace(go.Scatter(
        showlegend=False,
        x=[(1 - 11 / len_field) * 100],
        y=[50],
        mode="markers",
        marker=dict(
            color='white',
            size=7
        ),
    ))

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
        fig.update_layout(
            shapes=[
                # unfilled circle
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
                ),
            ]
        )
    else:
        fig.update_layout(
            shapes=[
                # unfilled circle
                dict(
                    type="circle",
                    xref="x",
                    yref="y",
                    x0=circle_x,
                    y0=circle_y,
                    x1=100 - circle_x,
                    y1=100 - circle_y,
                    line_color="white"
                ),
            ]
        )

    fig.update_layout(
        autosize=False,
        width=105 * 8,
        height=68 * 8)

    return fig


def get_plotly_figure_of_match(analyzer, home_team, away_team, side, graph_type="passes"):

    # get the Ids of the teams, the match as well as the score of the match
    home_id, away_id, match_id, home_score, away_score = analyzer.get_match_overview(home_team, away_team)

    # get the match summary for the plot title
    match_sum = f"{home_team}-{away_team}  {home_score}:{away_score}"

    # get the id of the team we are interested in
    team_id = home_id if side == "home" else away_id

    # compute the position of the players for the relevant team
    df_pos_team = analyzer.get_team_field_positions(match_id, team_id, side)

    # if the away team is shown, we show the team from right to left
    if side == "away":
        df_pos_team["centerX"] = 100 - df_pos_team["centerX"]
    else:
        df_pos_team["centerY"] = 100 - df_pos_team["centerY"]

    # compute the share of passes between each two players for the relevant team
    df_pass_team = analyzer.get_pass_share(match_id, team_id, df_pos_team)

    # get an empty soccer field
    field = create_empty_field(below=True)

    # only add if the passes between the different players should be displayed
    if graph_type == "passes":
        for _, row in df_pass_team.iterrows():
            field.add_trace(go.Scatter(
                            showlegend=False,
                            x=[row["centerX1"], row["centerX2"]],
                            y=[row["centerY1"], row["centerY2"]],
                            mode='lines',
                            line=dict(color='red', width=30*row["sharePasses"])))

    # add the name of the player
    field.add_trace(go.Scatter(
            x=df_pos_team["centerX"],
            y=df_pos_team["centerY"],
            showlegend=False,
            mode="markers+text",
            text=df_pos_team["shortName"],
            textposition="bottom center",
            marker=dict(color='red',
                        size=12)
    ))

    # add a title with the summary of the match
    field.update_layout(
            title={
                'text': match_sum,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

    return field
