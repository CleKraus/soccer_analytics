# -*- coding: utf-8 -*-
import logging
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

import components.layout as layout
from components.analyzer import Analyzer
import dashboard.layout.layout_positional_tab as pos_tab

logging.basicConfig(level=logging.DEBUG)

analyzer = Analyzer()

external_stylesheets = [dbc.themes.BOOTSTRAP]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


match_video = pos_tab.plotly_figure_game()

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


all_teams = sorted(list(analyzer.teams.teams_in_area("Germany")["teamName"]))

layout_pass_analysis = html.Div(
            children=[
                dbc.Row([
                  dbc.Col([
                      html.Div('Home team'),
                      dcc.Dropdown(
                          id="home_team",
                          options=[
                              {'label': name, 'value': name} for name in all_teams
                          ],
                          placeholder="Select home team"
                      ),
                  ], width=3),
                  dbc.Col([
                      html.Div('Away team'),
                      dcc.Dropdown(
                          id="away_team",
                          options=[
                              {'label': name, 'value': name} for name in all_teams
                          ],
                          placeholder="Select away team"
                      ),
                  ], width=3),
                  dbc.Col([
                      html.Div('Side'),
                      dcc.Dropdown(
                          id="side_show",
                          options=[
                              {'label': name, 'value': name} for name in ["Home", "Away"]
                          ],
                          value="Home"
                      ),
                  ], width=2),
                  dbc.Col([
                      html.Div('Graph type'),
                      dcc.Dropdown(
                          id="graph_type",
                          options=[
                              {'label': name, 'value': name} for name in ["Position only", "Passes"]
                          ],
                          value="Home"
                      ),
                  ], width=2),
                ], justify="around"),

                dbc.Row([
                    dcc.Graph(
                        id="soccer_field",
                        figure=layout.create_empty_field()
                    )
                ], justify="center")
        ])


app.layout = html.Div([
    layout.create_navigation_bar(),
    dcc.Tabs(id='tabs-example', value='tab-1', children=[
        dcc.Tab(label='Match overview', value='tab-1'),
        dcc.Tab(label='Match scenes', value='tab-2'),
    ]),
    html.Div(id='tabs-example-content')
])
app.config.suppress_callback_exceptions = True


@app.callback(Output('tabs-example-content', 'children'),
              [Input('tabs-example', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return layout_pass_analysis
    elif tab == 'tab-2':
        return layout_match_video


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
        return layout.get_plotly_figure_of_match(analyzer, home_team, away_team, side.lower(), graph_type.lower())


if __name__ == '__main__':
    app.run_server(debug=True)
