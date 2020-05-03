import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objects as go
from dash.dependencies import Input, Output

import dashboard.helper as helper


def get_layout(analyzer):
    all_teams = sorted(list(analyzer.teams.teams_in_area("Germany")["teamName"]))

    layout_pass_analysis = html.Div(
        children=[
            dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div("Home team"),
                            dcc.Dropdown(
                                id="home_team",
                                options=[
                                    {"label": name, "value": name} for name in all_teams
                                ],
                                placeholder="Select home team",
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div("Away team"),
                            dcc.Dropdown(
                                id="away_team",
                                options=[
                                    {"label": name, "value": name} for name in all_teams
                                ],
                                placeholder="Select away team",
                            ),
                        ],
                        width=3,
                    ),
                    dbc.Col(
                        [
                            html.Div("Side"),
                            dcc.Dropdown(
                                id="side_show",
                                options=[
                                    {"label": name, "value": name}
                                    for name in ["Home", "Away"]
                                ],
                                value="Home",
                            ),
                        ],
                        width=2,
                    ),
                    dbc.Col(
                        [
                            html.Div("Graph type"),
                            dcc.Dropdown(
                                id="graph_type",
                                options=[
                                    {"label": name, "value": name}
                                    for name in ["Position only", "Passes"]
                                ],
                                value="Home",
                            ),
                        ],
                        width=2,
                    ),
                ],
                justify="around",
            ),
            dbc.Row(
                [dcc.Graph(id="soccer_field", figure=helper.create_empty_field())],
                justify="center",
            ),
        ]
    )
    return layout_pass_analysis


def add_match_overview_callbacks(app, analyzer):
    @app.callback(
        Output("soccer_field", "figure"),
        [
            Input("home_team", "value"),
            Input("away_team", "value"),
            Input("side_show", "value"),
            Input("graph_type", "value"),
        ],
    )
    def update_soccer_field(home_team, away_team, side, graph_type):
        if home_team is None or away_team is None or home_team == away_team:
            return helper.create_empty_field()
        else:
            return get_plotly_figure_of_match(
                analyzer, home_team, away_team, side.lower(), graph_type.lower()
            )


def get_plotly_figure_of_match(
    analyzer, home_team, away_team, side, graph_type="passes"
):

    # get the Ids of the teams, the match as well as the score of the match
    home_id, away_id, match_id, home_score, away_score = analyzer.get_match_overview(
        home_team, away_team
    )

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
    field = helper.create_empty_field(below=True)

    # only add if the passes between the different players should be displayed
    if graph_type == "passes":
        for _, row in df_pass_team.iterrows():
            field.add_trace(
                go.Scatter(
                    showlegend=False,
                    x=[row["centerX1"], row["centerX2"]],
                    y=[row["centerY1"], row["centerY2"]],
                    mode="lines",
                    line=dict(color="red", width=30 * row["sharePasses"]),
                )
            )

    # add the name of the player
    field.add_trace(
        go.Scatter(
            x=df_pos_team["centerX"],
            y=df_pos_team["centerY"],
            showlegend=False,
            mode="markers+text",
            text=df_pos_team["shortName"],
            textposition="bottom center",
            marker=dict(color="red", size=12),
        )
    )

    # add a title with the summary of the match
    field.update_layout(
        title={
            "text": match_sum,
            "y": 0.9,
            "x": 0.5,
            "xanchor": "center",
            "yanchor": "top",
        }
    )

    return field
