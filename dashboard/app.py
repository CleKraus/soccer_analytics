# -*- coding: utf-8 -*-
import logging

import dash
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import dashboard.helper as helper
import dashboard.tabs.tab_match_overview as tab_match_overview
import dashboard.tabs.tab_match_scenes as tab_match_scenes
from dashboard.components import Analyzer

logging.basicConfig(level=logging.DEBUG)

if True:
    raise ValueError("Dashboard does not work at the moment!")

# pre-processing steps
######################

# analyzer takes care of storage and computation of all event driven data
analyzer = Analyzer()

# create and load the video shown in the match scene tab
match_video = tab_match_scenes.plotly_figure_game()

# set up dash app
#################

# initialize dash dashboard
external_stylesheets = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# add a navigation bar to the dashboard
app.layout = html.Div(
    [
        helper.create_navigation_bar(),
        dcc.Tabs(
            id="tabs-example",
            value="tab-1",
            children=[
                dcc.Tab(label="Match overview", value="tab-1"),
                dcc.Tab(label="Match scenes", value="tab-2"),
            ],
        ),
        html.Div(id="tabs-example-content"),
    ]
)
# make sure no callback exceptions are thrown - this is needed if one has different tabs
app.config.suppress_callback_exceptions = True

# add callbacks for the different tabs
tab_match_overview.add_match_overview_callbacks(app, analyzer)


# add logic to switch between the different tabs
@app.callback(
    Output("tabs-example-content", "children"), [Input("tabs-example", "value")]
)
def select_tab(tab):
    if tab == "tab-1":
        return tab_match_overview.get_layout(analyzer)
    elif tab == "tab-2":
        return tab_match_scenes.get_layout(match_video)


if __name__ == "__main__":
    app.run_server(debug=True)
