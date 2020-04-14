# -*- coding: utf-8 -*-

# import packages
import numpy as np
import pandas as pd
import ruamel.yaml
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS


def create_empty_field(below=False, len_field=105, wid_field=68):
    """
    Function returns a plotly figure of a soccer field.
    :param below: (bool) If true, any additional traces will overlay the field; otherwise, the field will overlay the
                         additional traces
    :param len_field: (int) Length of soccer field in meters (needs to be between 90m and 120m)
    :param wid_field: (int) Width of soccer field in meters (needs to be between 60m and 90m)
    :return: go.Figure with a soccer field
    """

    # check the input for correctness
    assert 90 <= len_field <= 120
    assert 60 <= wid_field <= 90
    assert type(below) is bool

    # size for center point and penalty points
    size_point = 0.5

    # set the overall layout of the field
    layout = go.Layout(
        # make sure the field is green
        plot_bgcolor="rgba(0,255,112,1)",
        xaxis=dict(range=[0, len_field],
                   showgrid=False,
                   showticklabels=False),
        yaxis=dict(range=[0, wid_field],
                   showgrid=False,
                   showticklabels=False),
    )

    # create an empty figure for which only the layout is set
    fig = go.Figure(layout=layout)

    # add the halfway line
    ######################
    fig.add_shape(
        dict(
            type="line",
            x0=len_field / 2,
            y0=0,
            x1=len_field / 2,
            y1=wid_field,
            line=dict(
                color="white",
                width=2
            )))

    # add left penalty area
    ########################
    y_box = ((wid_field - 40.32) / 2)
    x_vals = [0, 16, 16, 0]
    y_vals = [wid_field - y_box, wid_field - y_box, y_box, y_box]

    for i in range(len(x_vals) - 1):
        fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=x_vals[i],
                y0=y_vals[i],
                x1=x_vals[i + 1],
                y1=y_vals[i + 1],
                line=dict(
                    color="white",
                    width=2
                )))

    # add left goal area
    ####################
    y_small_box = 7.32 / 2 + 5.5
    x_vals = [0, 5.5, 5.5, 0]
    y_vals = [wid_field / 2 - y_small_box, wid_field / 2 - y_small_box, wid_field / 2 + y_small_box,
              wid_field / 2 + y_small_box]

    for i in range(len(x_vals) - 1):
        fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=x_vals[i],
                y0=y_vals[i],
                x1=x_vals[i + 1],
                y1=y_vals[i + 1],
                line=dict(
                    color="white",
                    width=2
                )))

    # add right penalty area
    ########################
    x_vals = [len_field, len_field - 16, len_field - 16, len_field]
    y_vals = [wid_field - y_box, wid_field - y_box, y_box, y_box]

    for i in range(len(x_vals) - 1):
        fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=x_vals[i],
                y0=y_vals[i],
                x1=x_vals[i + 1],
                y1=y_vals[i + 1],
                line=dict(
                    color="white",
                    width=2
                )))

    # add right goal area
    #####################
    y_small_box = 7.32 / 2 + 5.5
    x_vals = [len_field, len_field - 5.5, len_field - 5.5, len_field]
    y_vals = [wid_field / 2 - y_small_box, wid_field / 2 - y_small_box, wid_field / 2 + y_small_box,
              wid_field / 2 + y_small_box]

    for i in range(len(x_vals) - 1):
        fig.add_shape(
            # Line Vertical
            dict(
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
    ########################
    pen_point = (11, wid_field / 2)
    x_vals = [pen_point[0] - size_point, pen_point[0] + size_point]
    y_vals = [pen_point[1] - size_point, pen_point[1] + size_point]

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
            fillcolor="white"
        )
    )

    # add right penalty point
    #########################
    pen_point = (len_field - 11, wid_field / 2)
    x_vals = [pen_point[0] - size_point, pen_point[0] + size_point]
    y_vals = [pen_point[1] - size_point, pen_point[1] + size_point]

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
            fillcolor="white"
        )
    )

    # add center spot
    #################
    pen_point = (len_field / 2, wid_field / 2)
    x_vals = [pen_point[0] - size_point, pen_point[0] + size_point]
    y_vals = [pen_point[1] - size_point, pen_point[1] + size_point]

    fig.add_shape(
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=x_vals[0],
            y0=y_vals[0],
            x1=x_vals[1],
            y1=y_vals[1],
            line_color="white",
            fillcolor="white"
        )
    )

    # add center circle
    ###################

    # radius of the center circle (in meters)
    rad_circle = 9.15

    circle_y = (wid_field / 2 - rad_circle)
    circle_x = (len_field / 2 - rad_circle)

    fig.add_shape(
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=circle_x,
            y0=circle_y,
            x1=len_field - circle_x,
            y1=wid_field - circle_y,
            line_color="white"
        )
    )

    # configure the layout such that additional traces overlay the field
    if below:
        for shape in fig.layout["shapes"]:
            shape["layer"] = "below"

    # update the layout such that the field looks symmetrical
    fig.update_layout(
        autosize=False,
        width=len_field * 8,
        height=wid_field * 9)

    return fig


def prepare_event_plot(df, x_col, y_col):
    """
    Function to prepare the event plot (see *create_event_plot*). Expects a slice of the event data frame which is
    than prepared in a way that it can be directly handled by the *create_event_plot* function. Notice that as a data
    frame is returned, additional changes can be easily made by the user if required.
    :param df: (pd.DataFrame) DataFrame as slice of the event data. Data frame contains all the data to be displayed
                in the event plot.
    :param x_col: (str) Name of the column containing the x-coordinates
    :param y_col: (str) Name of the column containing the y-coordinates
    :return: pd.DataFrame to be used in the *create_event_plot* function
    """

    with open("config.yml", "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

    df = df.copy()
    # make sure the home team is always playing left to right and the away team right to left
    df[x_col] = np.where(df["teamId"] == df["awayTeam"], config["general"]["field_length"] - df[x_col], df[x_col])
    df[y_col] = np.where(df["teamId"] != df["awayTeam"], config["general"]["field_width"] - df[y_col], df[y_col])

    # make sure that duels are mapped together - if not we would have two points at exactly the same position
    df_duel = df[df["eventName"] == "Duel"].copy()
    df_duel[x_col] = np.round(df_duel[x_col], 2)
    df_duel[y_col] = np.round(df_duel[y_col], 2)
    df_group_duel = df_duel.groupby([x_col, y_col]).agg({"eventSec": "min", "shortName": " vs. ".join}).reset_index()
    df_group_duel["subEventName"] = df_group_duel["shortName"]
    df_group_duel["shortName"] = ""
    df_group_duel["teamId"] = -1

    df = df[df["eventName"] != "Duel"].copy()
    df = pd.concat([df, df_group_duel], axis=0)
    df.sort_values("eventSec", inplace=True)

    # add the color of the marker
    df["color"] = np.where(df["teamId"] == df["awayTeam"], "blue", np.where(df["teamId"] == -1, "black", "red"))
    # show the label (e.g. players name) either above or below the marker depending on the marker position on the field
    df["positionLabel"] = np.where(df[y_col] > 5, "bottom center", "top center")
    # set the information that is displayed when hovering over the marker
    df["hoverInfo"] = df["subEventName"]
    return df


def create_event_plot(df, x_col, y_col):
    """
    Creation of an event plot, i.e. a sequence of events is plotted on the field and connected through a line.
    :param df: (pd.DataFrame) Data frame containing all the events to be plotted; output of
                the *prepare_event_plot* function
                Notice that besides the coordinates, the data frame also needs to have information about the color
                of the markers, the label text and position as well as the hover information
    :param x_col: (str) Name of the column containing the x-coordinates
    :param y_col: (str) Name of the column containing the y-coordinates
    :return: go.Figure with the events plotted on top of the soccer field
    """

    # creation of an empty soccer field
    field = create_empty_field(below=True)

    # add all lines connecting the different markers on the field
    field.add_trace(
        go.Scatter(showlegend=False,
                   x=np.array(df[x_col]),
                   y=np.array(df[y_col]),
                   mode="lines",
                   line=dict(color="black", width=1),
                   hoverinfo='none'
                   )
    )

    # set all the markers; notice that the position + text of the label, the color of the marker and the
    # hover info needs to be part of the *df*
    field.add_trace(
        go.Scatter(showlegend=False,
                   x=np.array(df[x_col]),
                   y=np.array(df[y_col]),
                   mode="markers+text",
                   text=np.array(df["shortName"]),
                   textposition=np.array(df["positionLabel"]),
                   marker=dict(color=np.array(df["color"]), size=12),
                   hovertext=df["hoverInfo"],
                   hoverinfo="text",
                   )
    )
    return field


def prepare_event_animation(df, x_col_bef, x_col_aft, y_col_bef, y_col_aft):
    """
    Function to prepare the event animation (see *create_event_animation*); this means that relevant information needed
    for the event animation (e.g. an update on the exact event time) is pre-computed in this function.
    :param df: (pd.DataFrame) DataFrame as slice of the event data. Data frame contains all the data to be displayed
                in the event animation.
    :param x_col_bef: (str) Column name of the x-coordinate at event start
    :param x_col_aft: (str) Column name of the x-coordinate at event end
    :param y_col_bef: (str) Column name of the y-coordinate at event start
    :param y_col_aft: (str) Column name of the y-coordinate at event end
    :return: pd.DataFrame containing all information to allow for usage of *create_event_animation*
    """

    # steps required for the event plot are also required for event animation
    df = prepare_event_plot(df, x_col_bef, y_col_bef)

    # update the time of the events such that the ball has always the same speed
    df["eventSec"] = df["eventSec"] - min(df["eventSec"])
    df[x_col_aft] = df[x_col_bef].shift(-1)
    df[y_col_aft] = df[y_col_bef].shift(-1)
    df["distX"] = df[x_col_aft] - df[x_col_bef]
    df["distY"] = df[y_col_aft] - df[y_col_bef]
    df["distCovered"] = np.sqrt(df["distX"] * df["distX"] + df["distY"] * df["distY"])
    df["cumDistCovered"] = df["distCovered"].cumsum().shift(1).fillna(0)
    total_dist = np.sum(df["distCovered"])
    df["shareDistCovered"] = df["cumDistCovered"] / total_dist
    df["eventSecNew"] = max(df["eventSec"]) * df["shareDistCovered"]

    # compute the duration of each event
    df["durationEvent"] = (df["eventSecNew"].shift(-1) - df["eventSecNew"]).fillna(0)
    df["shortName"] = np.where(df["shortName"] == "", "Duel", df["shortName"])

    return df


def _get_ball_position_per_frame(df, total_seconds, fps, x_col_bef, x_col_aft, y_col_bef, y_col_aft):
    """
    Helper function to compute the ball position for each frame. Function is needed as part of the
    "create_event_animation" function.
    :param df: (pd.DataFrame) Data frame containing the event positions and their duration
    :param total_seconds: (float) Total number of seconds (w.r.t. time in the soccer match) to be animated
    :param fps: (int) Number of frames per second to be displayed
    :param x_col_bef: (str) Column name of the x-coordinate at event start
    :param x_col_aft: (str) Column name of the x-coordinate at event end
    :param y_col_bef: (str) Column name of the y-coordinate at event start
    :param y_col_aft: (str) Column name of the y-coordinate at event end
    :return: List of type [[second, column, x-position of ball, y-position of ball]]
    """

    # save all relevant information as array to be faster
    pos_before_x = np.array(df[x_col_bef])
    pos_after_x = np.array(df[x_col_aft])
    pos_before_y = np.array(df[y_col_bef])
    pos_after_y = np.array(df[y_col_aft])
    event_durations = np.array(df["durationEvent"])
    times = np.array(df["eventSecNew"])

    col = 0
    ball_positions = list()
    # loop through frames
    for i in range(total_seconds * fps):

        sec = i / fps
        # in case it is the last event, just leave the ball there
        if col + 1 == len(pos_before_x):
            ball_positions.append([sec, col, pos_before_x[col], pos_before_y[col]])
            continue

        # if event is over, jump to the next event
        end_time = times[col + 1]
        if sec > end_time:
            col += 1

        # compute the ball position in frame i (i.e. after i/fps seconds)
        start_point_x = pos_before_x[col]
        start_point_y = pos_before_y[col]
        end_point_x = pos_after_x[col]
        end_point_y = pos_after_y[col]
        duration = event_durations[col]
        start_time = times[col]

        share_time = (sec - start_time) / duration
        current_pos_x = start_point_x + (end_point_x - start_point_x) * share_time
        current_pos_y = start_point_y + (end_point_y - start_point_y) * share_time
        ball_positions.append([sec, col, current_pos_x, current_pos_y])
    return ball_positions


def create_event_animation(df, total_seconds, fps,
                           x_col_bef="posBeforeXMeters",
                           x_col_aft="posAfterXMeters",
                           y_col_bef="posBeforeYMeters",
                           y_col_aft="posAfterYMeters"):
    """
    Creation of an event animation, i.e. a sequence of events is animated on the field and connected through a line.
    Notice that unlike the create_xxx_plot functions a dictionary instead of a go.Figure is returned. This is due to
    the fact that a creation of a go.Figure with many frames is very slow.
    :param df: (pd.DataFrame) Data frame as output of the *prepare_event_animation* function
    :param total_seconds: (float) Total number of seconds (w.r.t. time in the soccer match) to be animated
    :param fps: (int) Number of frames per second to be displayed
    :param x_col_bef: (str) Column name of the x-coordinate at event start
    :param x_col_aft: (str) Column name of the x-coordinate at event end
    :param y_col_bef: (str) Column name of the y-coordinate at event start
    :param y_col_aft: (str) Column name of the y-coordinate at event end
    :return: dict containing all the information needed for the animation. Can be easily displayed through
             plotly.offline.iplot(animation, validate=False, auto_play=False)
    """
    # get the ball position in each picture
    ball_positions = _get_ball_position_per_frame(df, total_seconds, fps, x_col_bef, x_col_aft, y_col_bef, y_col_aft)

    # compute data for first picture
    ################################
    data = []
    tmp_df = df[df["eventSecNew"] == 0].copy()

    line_data = dict(
        showlegend=False,
        x=np.array(tmp_df[x_col_bef]),
        y=np.array(tmp_df[y_col_bef]),
        mode="lines",
        line=dict(color="black", width=1),
        hoverinfo="none"
    )

    data.append(line_data)

    ball_data = dict(
        showlegend=False,
        x=np.array([ball_positions[0][2]]),
        y=np.array([ball_positions[0][3]]),
        mode="markers",
        marker=dict(color="white", size=15),
        hoverinfo="none"
    )
    data.append(ball_data)

    player_data = dict(
        showlegend=False,
        x=np.array(tmp_df[x_col_bef]),
        y=np.array(tmp_df[y_col_bef]),
        mode="markers+text",
        text=np.array(tmp_df["shortName"]),
        textposition=np.array(tmp_df["positionLabel"]),
        marker=dict(color=np.array(tmp_df["color"]), size=12),
        hovertext=tmp_df["hoverInfo"],
        hoverinfo="text",
    )

    data.append(player_data)

    # compute all frames
    ####################

    frames = list()
    for i in range(total_seconds * fps):
        sec = i / fps

        amount_player = len(df[df["eventSecNew"] <= sec])
        tmp_df = df.iloc[:(amount_player + 1)].copy()

        x_pos = np.array(tmp_df[x_col_bef])
        y_pos = np.array(tmp_df[y_col_bef])

        player_data = dict(
            showlegend=False,
            x=x_pos,
            y=y_pos,
            mode="markers+text",
            text=np.array(tmp_df["shortName"]),
            textposition=np.array(tmp_df["positionLabel"]),
            marker=dict(color=np.array(tmp_df["color"]), size=12),
            hovertext=tmp_df["hoverInfo"],
            hoverinfo="text",
        )
        ball_data = dict(
            showlegend=False,
            x=np.array([ball_positions[i][2]]),
            y=np.array([ball_positions[i][3]]),
            mode="markers",
            marker=dict(color="white", size=14),
            hoverinfo="none"
        )

        x_pos_line = np.array(x_pos)
        y_pos_line = np.array(y_pos)
        x_pos_line[-1] = ball_positions[i][2]
        y_pos_line[-1] = ball_positions[i][3]

        line_data = dict(
            showlegend=False,
            x=x_pos_line,
            y=y_pos_line,
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="none"
        )

        frame = dict(
            data=[line_data, ball_data, player_data]
        )

        frames.append(frame)

    # compute the layout
    ####################

    field = create_empty_field(below=True)
    layout = dict(
        # make sure the field is green
        plot_bgcolor=field["layout"]["plot_bgcolor"],
        xaxis=field["layout"]["xaxis"],
        yaxis=field["layout"]["yaxis"],
        shapes=field["layout"]["shapes"],
        width=field["layout"]["width"],
        height=field["layout"]["height"],
        autosize=field["layout"]["autosize"],
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None,
                                {"frame": {"duration": 100, "redraw": False}, "fromcurrent": True}, ]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False},
                                         "mode": "immediate",
                                         "transition": {"duration": 0}}])
                     ])],
    )

    fig = dict(
        data=data,
        layout=layout,
        frames=frames
    )

    return fig


def create_univariate_variable_graph(df, col, target_col, y1_axis_name, y2_axis_name, binned_cols=False,
                                     title_name=None):
    """
    Function creates a plotly figure showing the distribution over the different values in column *col* as well as the share
    of positives for the *target_col* per value in *col*.

    :param df: (pd.DataFrame) Data frame with all relevant data
    :param col: (str) Column for which the different values should be analyzed
    :param target_col: (str) Column with the target, e.g. "Goal", "Successful pass", ...
    :param y1_axis_name: (str) String to be displayed on left y-axis
    :param y2_axis_name: (str) String to be displayed on right y-axis
    :param binned_cols: (bool, default=False) Set to true of columns were manually binned and x-axis tick names should
                                              therefore be updated
    :param title_name: (str) Title of the plotly plot
    :return: go.Figure containing the univariate variable graph
    """
    # if columns were manually binned before, we show the x-axis ticks like ("<3", "3-6", ">=6")
    if binned_cols:
        diff_vals = sorted(df[col].unique())
        lst_x_title = list()
        for i in range(len(diff_vals)):
            if i == 0:
                lst_x_title.append(f"<{diff_vals[i + 1]}")
            elif i == len(diff_vals) - 1:
                lst_x_title.append(f">={diff_vals[i]}")
            else:
                lst_x_title.append(f"{diff_vals[i]} - {diff_vals[i + 1]}")

    # compute the share of observations for each group and the probability of the target
    df_group = df.groupby(col).agg(total_count=(col, "count"),
                                   total_target=(target_col, "sum")).reset_index()
    df_group["share"] = df_group["total_count"] / len(df) * 100
    df_group["share_target"] = df_group["total_target"] / df_group["total_count"] * 100

    # create the plotly figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # right y-axis corresponds to the share of observations per group
    fig.add_trace(
        go.Bar(
            x=df_group[col],
            y=df_group["share"],
            name=y2_axis_name,
            marker=dict(color=DEFAULT_PLOTLY_COLORS[0])
        ))

    # left y-axis corresponds to the probability of the target for each group
    fig.add_trace(
        go.Scatter(
            x=df_group[col],
            y=df_group["share_target"],
            name=y1_axis_name,
            marker=dict(color=DEFAULT_PLOTLY_COLORS[1])
        ), secondary_y=True)

    if title_name is None:
        title_name = col

    # update the layout of the figure
    fig.update_layout(
        title=title_name,
        hovermode=False,
        yaxis=dict(
            title=y1_axis_name,
            titlefont_size=16,
            tickfont_size=14,
            rangemode="tozero"
        ),
        yaxis2=dict(
            title=y2_axis_name,
            titlefont_size=16,
            tickfont_size=14,
            rangemode="tozero"
        ),
    )

    # update the x-axis title in case of a binned column
    if binned_cols:
        fig.data[0]["x"] = np.array(lst_x_title)
        fig.data[1]["x"] = np.array(lst_x_title)

    return fig


def combine_univariate_variable_graphs(figures, cols, rows, shared_axis=False):
    titles = []
    for tmp_fig in figures:
        titles.append(tmp_fig["layout"]["title"]["text"])

    if shared_axis:
        fig = make_subplots(
            rows=rows, cols=cols, subplot_titles=titles, shared_yaxes=True)
    else:
        fig = make_subplots(
            rows=rows, cols=cols, subplot_titles=titles, shared_yaxes=True)

        # add the data to the plots
    for row in range(rows):
        for col in range(cols):
            for fig_data in figures[row * cols + col]["data"]:
                fig.add_trace(fig_data, row=row + 1, col=col + 1)

    # add name to left y-axis
    for i, tmp_fig in enumerate(figures):
        axis_name = "yaxis" if i == 0 else "yaxis" + str(i + 1)
        fig.layout[axis_name]["title"] = tmp_fig["layout"]["yaxis"]["title"]["text"]
        fig.layout[axis_name]["tickfont"]["size"] = 8
        fig.layout[axis_name]["title"]["font"]["size"] = 10

    # add name to right y-axis
    for i, tmp_fig in enumerate(figures):
        axis_name = "yaxis" + str(len(figures) + 1 + i)
        anchor_name = "x" if i == 0 else "x" + str(i + 1)
        overlay_name = "y" if i == 0 else "y" + str(i + 1)
        fig.layout[axis_name] = tmp_fig["layout"]["yaxis2"]
        fig.layout[axis_name]["anchor"] = anchor_name
        fig.layout[axis_name]["overlaying"] = overlay_name
        fig.layout[axis_name]["tickfont"]["size"] = 8
        fig.layout[axis_name]["title"]["font"]["size"] = 10
        fig["data"][(2 * i) + 1].update(yaxis="y" + str(len(figures) + 1 + i))

    if shared_axis:
        for i, tmp_fig in enumerate(figures):
            # make sure the right axis match
            if i > 0:
                axis_name = "yaxis" + str(len(figures) + 1 + i)
                fig["layout"][axis_name]["matches"] = "y" + str(len(figures) + 1)

            # delete unneeded titles on right axis
            if i % cols != (cols - 1):
                axis_name = "yaxis" + str(len(figures) + 1 + i)
                fig["layout"][axis_name]["title"] = None

            # delete unneed titles on left axis
            if i % cols != 0:
                axis_name = "yaxis" + str(i + 1)
                fig["layout"][axis_name]["title"] = None

    fig.update_layout(showlegend=False, hovermode=False)

    return fig


def _calculate_bucket_for_position(series, nb_buckets, min_pos_val, max_pos_val):
    buckets = np.arange(min_pos_val, max_pos_val + 0.001, max_pos_val / nb_buckets)

    df_buckets = pd.DataFrame()
    df_buckets["id"] = np.arange(len(buckets) - 1)
    df_buckets["minValueBucket"] = list(buckets)[:-1]
    df_buckets["maxValueBucket"] = list(buckets)[1:]
    df_buckets["meanValueBucket"] = (df_buckets["minValueBucket"] + df_buckets["maxValueBucket"]) / 2

    buckets[-1] = buckets[-1] + 0.001

    return pd.cut(series, buckets, labels=False, include_lowest=True), df_buckets


def prepare_heatmap(df, col_x, col_y, nb_buckets_x, nb_buckets_y, min_val_x=0, max_val_x=105, min_val_y=0,
                    max_val_y=68):
    df = df.copy()

    df[col_x + "Bucket"], df_lookup_x_buckets = _calculate_bucket_for_position(df[col_x], nb_buckets_x, min_val_x,
                                                                               max_val_x)
    df[col_y + "Bucket"], df_lookup_y_buckets = _calculate_bucket_for_position(df[col_y], nb_buckets_y, min_val_y,
                                                                               max_val_y)

    df_pos = df.groupby([col_x + "Bucket", col_y + "Bucket"]). \
        agg(nbEvents=(col_x + "Bucket", "count")).reset_index()

    df_all_pos = pd.DataFrame([(x, y) for x in df_lookup_x_buckets["id"] for y in df_lookup_y_buckets["id"]],
                              columns=[col_x + "Bucket", col_y + "Bucket"])

    df_lookup_x_buckets.rename(columns={"id": col_x + "Bucket", "meanValueBucket": col_x + "BucketMean"}, inplace=True)
    df_lookup_y_buckets.rename(columns={"id": col_y + "Bucket", "meanValueBucket": col_y + "BucketMean"}, inplace=True)

    df_all_pos = pd.merge(df_all_pos, df_lookup_x_buckets[[col_x + "Bucket", col_x + "BucketMean"]], how="left")
    df_all_pos = pd.merge(df_all_pos, df_lookup_y_buckets[[col_y + "Bucket", col_y + "BucketMean"]], how="left")

    df_pos = pd.merge(df_all_pos, df_pos, how="left").fillna(0)
    df_img = df_pos.pivot(col_y + "BucketMean", col_x + "BucketMean", "nbEvents")

    x = list(df_img.columns)
    y = list(df_img.index)

    img = np.array(df_img)

    return img, x, y


def create_heatmap(x, y, z, dict_info, title_name=None):

    # Prepare the text to be shown when hovering over the heatmap
    hovertext = list()
    for idy in range(len(z)):
        hovertext.append(list())
        for idx in range(len(z[1])):
            text = ""
            for key in dict_info.keys():
                text += "{}: {:^{display_type}}<br />".format(key, dict_info[key]["values"][idy][idx],
                                                              display_type=dict_info[key]["display_type"])
            hovertext[-1].append(text)

    # get the empty soccer field
    fig = create_empty_field()
    # overlay field with the heatmap
    fig.add_trace(
        go.Heatmap(x=x, y=y, z=z, hoverinfo='text', text=hovertext)
    )

    if title_name is not None:
        fig.update_layout(
            title={
                'text': title_name,
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})

    return fig
