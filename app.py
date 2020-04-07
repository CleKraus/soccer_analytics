# -*- coding: utf-8 -*-
import logging
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objects as go

import components.layout as layout
from components.analyzer import Analyzer

logging.basicConfig(level=logging.DEBUG)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

analyzer = Analyzer()

all_teams = sorted(list(analyzer.teams.teams_in_area("Germany")["teamName"]))


def get_plotly_figure_of_match(home_team, away_team, side, graph_type="passes"):

    # get the Ids of the teams, the match as well as the score of the match
    home_id, away_id, match_id, home_score, away_score = analyzer.get_match_overview(home_team, away_team)

    # get the match summary for the plot title
    match_sum = f"{home_team}-{away_team}  {home_score}:{away_score}"

    # get the id of the team we are interested in
    team_id = home_id if side == "home" else away_id

    # compute the position of the players for the relevant team
    df_pos_team = analyzer.get_team_field_positions(match_id, team_id, side)

    # compute the share of passes between each two players for the relevant team
    df_pass_team = analyzer.get_pass_share(match_id, team_id, df_pos_team)

    field = layout.create_empty_field(below=True)

    if graph_type == "passes":
        for _, row in df_pass_team.iterrows():
            field.add_trace(go.Scatter(
                            showlegend=False,
                            x=[row["centerX1"], row["centerX2"]],
                            y=[row["centerY1"], row["centerY2"]],
                            mode='lines',
                            line=dict(color='red', width=30*row["sharePasses"])))

    field.add_trace(go.Scatter(
        x=df_pos_team["centerX"],
        y=df_pos_team["centerY"],
        showlegend=False,
        mode="markers+text",
        text=df_pos_team["shortName"],
        textposition="bottom center",
        marker=dict(color='red',
                    size=12
        ),
    ))

    field.update_layout(
            title={
                'text': match_sum,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

    return field


app.layout = html.Div(
    children=[
            html.Label('Home team'),
            dcc.Dropdown(
                id="home_team",
                options=[
                    {'label': name, 'value': name} for name in all_teams
                ],
                placeholder="Select home team"
            ),

            html.Label('Away team'),
            dcc.Dropdown(
                id="away_team",
                options=[
                    {'label': name, 'value': name} for name in all_teams
                ],
                placeholder="Select away team"
            ),

            html.Label('Side'),
            dcc.Dropdown(
                id="side_show",
                options=[
                    {'label': name, 'value': name} for name in ["Home", "Away"]
                ],
                value="Home"
            ),

            html.Label('Graph type'),
            dcc.Dropdown(
                id="graph_type",
                options=[
                    {'label': name, 'value': name} for name in ["Position only", "Passes"]
                ],
                value="Home"
            ),

            dcc.Graph(
                id="soccer_field",
                figure=layout.create_empty_field()
            ),
            html.Div(id='my-div')
    ])


@app.callback(
    Output("soccer_field", "figure"),
    [Input("home_team", "value"),
     Input("away_team", "value"),
     Input("side_show", "value"),
     Input("graph_type", "value")]
)
def update_output_div(home_team, away_team, side, graph_type):
    if home_team is None or away_team is None or home_team == away_team:
        return layout.create_empty_field()
    else:
        return get_plotly_figure_of_match(home_team, away_team, side.lower(), graph_type.lower())


if __name__ == '__main__':
    app.run_server(debug=True)