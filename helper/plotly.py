# -*- coding: utf-8 -*-

# import packages
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import ruamel.yaml
import scipy.spatial

import helper.event_data as ed_help
import helper.general as gen_help


def create_empty_field(
    below=False, colour="green", line_colour=None, size=1, len_field=105, wid_field=68
):
    """
    Function returns a plotly figure of a soccer field.
    :param below: (bool) If true, any additional traces will overlay the field; otherwise, the field will overlay the
                         additional traces
    :param colour: (str) Colour of the field; currently only "green" and "white" are supported
    :param line_colour: (str) Colour of the line; if none it is automatically set based on the field colour
    :param size: (float) Size relative to the standard size
    :param len_field: (int) Length of soccer field in meters (needs to be between 90m and 120m)
    :param wid_field: (int) Width of soccer field in meters (needs to be between 60m and 90m)
    :return: go.Figure with a soccer field
    """

    # check the input for correctness
    assert 90 <= len_field <= 120
    assert 60 <= wid_field <= 90
    assert colour in ["green", "white"]
    assert type(below) is bool

    # size for center point and penalty points
    size_point = 0.5

    field_colour = "rgba(0,255,112,1)" if colour == "green" else "white"

    if line_colour is None:
        line_colour = "white" if colour == "green" else "black"

    # set the overall layout of the field
    layout = go.Layout(
        # make sure the field is green
        plot_bgcolor=field_colour,
        xaxis=dict(
            range=[-5, len_field + 5],
            zeroline=False,
            showgrid=False,
            showticklabels=False,
        ),
        yaxis=dict(
            range=[-5, wid_field + 5],
            zeroline=False,
            showgrid=False,
            showticklabels=False,
        ),
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
            line=dict(color=line_colour, width=2),
        )
    )

    # add left penalty area
    ########################
    y_box = (wid_field - 40.32) / 2
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
                line=dict(color=line_colour, width=2),
            )
        )

    # add left goal area
    ####################
    y_small_box = 7.32 / 2 + 5.5
    x_vals = [0, 5.5, 5.5, 0]
    y_vals = [
        wid_field / 2 - y_small_box,
        wid_field / 2 - y_small_box,
        wid_field / 2 + y_small_box,
        wid_field / 2 + y_small_box,
    ]

    for i in range(len(x_vals) - 1):
        fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=x_vals[i],
                y0=y_vals[i],
                x1=x_vals[i + 1],
                y1=y_vals[i + 1],
                line=dict(color=line_colour, width=2),
            )
        )

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
                line=dict(color=line_colour, width=2),
            )
        )

    # add right goal area
    #####################
    y_small_box = 7.32 / 2 + 5.5
    x_vals = [len_field, len_field - 5.5, len_field - 5.5, len_field]
    y_vals = [
        wid_field / 2 - y_small_box,
        wid_field / 2 - y_small_box,
        wid_field / 2 + y_small_box,
        wid_field / 2 + y_small_box,
    ]

    for i in range(len(x_vals) - 1):
        fig.add_shape(
            # Line Vertical
            dict(
                type="line",
                x0=x_vals[i],
                y0=y_vals[i],
                x1=x_vals[i + 1],
                y1=y_vals[i + 1],
                line=dict(color=line_colour, width=2),
            )
        )

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
            line_color=line_colour,
            fillcolor=line_colour,
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
            line_color=line_colour,
            fillcolor=line_colour,
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
            line_color=line_colour,
            fillcolor=line_colour,
        )
    )

    # add center circle
    ###################

    # radius of the center circle (in meters)
    rad_circle = 9.15

    circle_y = wid_field / 2 - rad_circle
    circle_x = len_field / 2 - rad_circle

    fig.add_shape(
        dict(
            type="circle",
            xref="x",
            yref="y",
            x0=circle_x,
            y0=circle_y,
            x1=len_field - circle_x,
            y1=wid_field - circle_y,
            line_color=line_colour,
        )
    )

    # add outer lines
    ###################

    fig.add_shape(
        dict(
            type="line",
            x0=0,
            y0=0,
            x1=len_field,
            y1=0,
            line=dict(color=line_colour, width=2),
        )
    )

    # add the out lines
    fig.add_shape(
        dict(
            type="line",
            x0=0,
            y0=0,
            x1=0,
            y1=wid_field,
            line=dict(color=line_colour, width=2),
        )
    )

    # add the out lines
    fig.add_shape(
        dict(
            type="line",
            x0=0,
            y0=wid_field,
            x1=len_field,
            y1=wid_field,
            line=dict(color=line_colour, width=2),
        )
    )

    # add the out lines
    fig.add_shape(
        dict(
            type="line",
            x0=len_field,
            y0=0,
            x1=len_field,
            y1=wid_field,
            line=dict(color=line_colour, width=2),
        )
    )

    # add goals
    ###########

    goal_width = 7.32

    # left goal
    fig.add_shape(
        dict(
            type="line",
            x0=0,
            y0=(wid_field - goal_width) / 2,
            x1=-2,
            y1=(wid_field - goal_width) / 2,
            line=dict(color=line_colour, width=2),
        )
    )

    fig.add_shape(
        dict(
            type="line",
            x0=0,
            y0=(wid_field + goal_width) / 2,
            x1=-2,
            y1=(wid_field + goal_width) / 2,
            line=dict(color=line_colour, width=2),
        )
    )

    fig.add_shape(
        dict(
            type="line",
            x0=-2,
            y0=(wid_field - goal_width) / 2,
            x1=-2,
            y1=(wid_field + goal_width) / 2,
            line=dict(color=line_colour, width=2),
        )
    )

    # right goal
    fig.add_shape(
        dict(
            type="line",
            x0=len_field,
            y0=(wid_field - goal_width) / 2,
            x1=len_field + 2,
            y1=(wid_field - goal_width) / 2,
            line=dict(color=line_colour, width=2),
        )
    )

    fig.add_shape(
        dict(
            type="line",
            x0=len_field,
            y0=(wid_field + goal_width) / 2,
            x1=len_field + 2,
            y1=(wid_field + goal_width) / 2,
            line=dict(color=line_colour, width=2),
        )
    )

    fig.add_shape(
        dict(
            type="line",
            x0=len_field + 2,
            y0=(wid_field - goal_width) / 2,
            x1=len_field + 2,
            y1=(wid_field + goal_width) / 2,
            line=dict(color=line_colour, width=2),
        )
    )

    # configure the layout such that additional traces overlay the field
    if below:
        for shape in fig.layout["shapes"]:
            shape["layer"] = "below"

    # update the layout such that the field looks symmetrical
    fig.update_layout(
        autosize=False, width=len_field * 8 * size, height=wid_field * 9 * size
    )

    return fig


def _build_hover_text(row, dict_info):
    """
    Helper function to build the hover text
    """
    text = ""
    for key in dict_info.keys():
        if "display_type" in dict_info[key]:
            text += "{}: {:^{display_type}}<br />".format(
                key,
                row[dict_info[key]["values"]],
                display_type=dict_info[key]["display_type"],
            )
        else:
            text += "{}: {}<br />".format(key, row[dict_info[key]["values"]])
    return text


def prepare_event_plot(df, x_col, y_col, label_info=None, left_team=None):
    """
    Function to prepare the event plot (see *create_event_plot*). Expects a slice of the event data frame which is
    than prepared in a way that it can be directly handled by the *create_event_plot* function. Notice that as a data
    frame is returned, additional changes can be easily made by the user if required.
    :param df: (pd.DataFrame) DataFrame as slice of the event data. Data frame contains all the data to be displayed
                in the event plot.
    :param x_col: (str) Name of the column containing the x-coordinates
    :param y_col: (str) Name of the column containing the y-coordinates
    :param label_info: (dict) Describes which information is shown as hover info
    :param left_team: (int) Integer indicating the team that plays from left to right on the field
    :return: pd.DataFrame to be used in the *create_event_plot* function
    """

    if label_info is None:
        label_info = {
            "Time": {"values": "eventSec", "display_type": ".1f"},
            "Event": {"values": "subEventName"},
            "Player": {"values": "playerName"},
            "Position": {"values": "playerPosition"},
            "EventId": {"values": "id"},
        }

    with open("config.yml", "r", encoding="utf-8") as f:
        config = ruamel.yaml.YAML().load(f)

    df = df.copy()

    if left_team is not None:
        df["leftTeamId"] = left_team
    else:
        df["leftTeamId"] = df["homeTeamId"]

    # make sure the id is a string and no float
    df["id"] = df["id"].astype(str)

    # make sure the left team is always playing left to right and the other team right to left
    df[x_col] = np.where(
        df["teamId"] != df["leftTeamId"],
        config["general"]["field_length"] - df[x_col],
        df[x_col],
    )
    df[y_col] = np.where(
        df["teamId"] == df["leftTeamId"],
        config["general"]["field_width"] - df[y_col],
        df[y_col],
    )

    # make sure that duels are mapped together - if not we would have two points at exactly the same position
    df_duel = df[df["eventName"] == "Duel"].copy()
    df_duel[x_col] = np.round(df_duel[x_col], 2)
    df_duel[y_col] = np.round(df_duel[y_col], 2)
    df_duel["playerName"].fillna("", inplace=True)
    df_duel["playerPosition"].fillna("", inplace=True)
    df_group_duel = (
        df_duel.groupby([x_col, y_col])
        .agg(
            {
                "eventSec": "min",
                "playerName": " vs. ".join,
                "playerPosition": " & ".join,
                "teamId": "count",
                "id": " & ".join,
            }
        )
        .reset_index()
    )

    # we delete all duels with only one player. Reason: We noticed that these duels often do not make a lot of
    # sense and their coordinates seem to be off...
    df_group_duel.rename(columns={"teamId": "nbPlayersInvolved"}, inplace=True)
    df_group_duel = df_group_duel[df_group_duel["nbPlayersInvolved"] > 1].copy()

    df_group_duel["subEventName"] = "Duel"
    df_group_duel["teamId"] = -1

    df = df[df["eventName"] != "Duel"].copy()
    df = pd.concat([df, df_group_duel], axis=0)
    df.sort_values("eventSec", inplace=True)

    # get the event second starting with the first event
    df["eventSec"] = df["eventSec"] - min(df["eventSec"])

    # add the color of the marker
    df["color"] = np.where(
        df["eventName"].isin(["Interruption", "Offside", "Foul"]),
        "yellow",
        np.where(
            df["teamId"] == -1,
            "black",
            np.where(df["teamId"] == df["leftTeamId"], "blue", "red"),
        ),
    )
    df["labelText"] = np.where(df["color"].isin(["blue", "red"]), df["playerName"], "")

    # show the label (e.g. players name) either above or below the marker depending on the marker position on the field
    df["positionLabel"] = np.where(df[y_col] > 5, "bottom center", "top center")
    # set the information that is displayed when hovering over the marker
    df["hoverInfo"] = df.apply(lambda row: _build_hover_text(row, label_info), axis=1)
    return df


def create_event_plot(df, x_col, y_col, size=1):
    """
    Creation of an event plot, i.e. a sequence of events is plotted on the field and connected through a line.
    :param df: (pd.DataFrame) Data frame containing all the events to be plotted; output of
                the *prepare_event_plot* function
                Notice that besides the coordinates, the data frame also needs to have information about the color
                of the markers, the label text and position as well as the hover information
    :param x_col: (str) Name of the column containing the x-coordinates
    :param y_col: (str) Name of the column containing the y-coordinates
    :param size: (float) Relative size of the field
    :return: go.Figure with the events plotted on top of the soccer field
    """

    # creation of an empty soccer field
    field = create_empty_field(below=True, size=size)

    # add all lines connecting the different markers on the field
    field.add_trace(
        go.Scatter(
            showlegend=False,
            x=np.array(df[x_col]),
            y=np.array(df[y_col]),
            mode="lines",
            line=dict(color="black", width=1),
            hoverinfo="none",
        )
    )

    # set all the markers; notice that the position + text of the label, the color of the marker and the
    # hover info needs to be part of the *df*
    field.add_trace(
        go.Scatter(
            showlegend=False,
            x=np.array(df[x_col]),
            y=np.array(df[y_col]),
            mode="markers+text",
            text=np.array(df["labelText"]),
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

    # always compute where the ball is going to be next
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
    df["playerName"] = np.where(df["playerName"] == "", "Duel", df["playerName"])

    return df


def _get_ball_position_per_frame(
    df, total_seconds, fps, x_col_bef, x_col_aft, y_col_bef, y_col_aft
):
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


def create_event_animation(
    df,
    total_seconds,
    fps=10,
    size=1,
    x_col_bef="posBeforeXMeters",
    x_col_aft="posAfterXMeters",
    y_col_bef="posBeforeYMeters",
    y_col_aft="posAfterYMeters",
):
    """
    Creation of an event animation, i.e. a sequence of events is animated on the field and connected through a line.
    Notice that unlike the create_xxx_plot functions a dictionary instead of a go.Figure is returned. This is due to
    the fact that a creation of a go.Figure with many frames is very slow.
    :param df: (pd.DataFrame) Data frame as output of the *prepare_event_animation* function
    :param total_seconds: (float) Total number of seconds (w.r.t. time in the soccer match) to be animated
    :param fps: (int) Number of frames per second to be displayed
    :param size: (float) Relative size of the field
    :param x_col_bef: (str) Column name of the x-coordinate at event start
    :param x_col_aft: (str) Column name of the x-coordinate at event end
    :param y_col_bef: (str) Column name of the y-coordinate at event start
    :param y_col_aft: (str) Column name of the y-coordinate at event end
    :return: dict containing all the information needed for the animation. Can be easily displayed through
             plotly.offline.iplot(animation, validate=False, auto_play=False)
    """
    # get the ball position in each picture
    ball_positions = _get_ball_position_per_frame(
        df, total_seconds, fps, x_col_bef, x_col_aft, y_col_bef, y_col_aft
    )

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
        hoverinfo="none",
    )

    data.append(line_data)

    ball_data = dict(
        showlegend=False,
        x=np.array([ball_positions[0][2]]),
        y=np.array([ball_positions[0][3]]),
        mode="markers",
        marker=dict(color="white", size=15),
        hoverinfo="none",
    )
    data.append(ball_data)

    player_data = dict(
        showlegend=False,
        x=np.array(tmp_df[x_col_bef]),
        y=np.array(tmp_df[y_col_bef]),
        mode="markers+text",
        text=np.array(tmp_df["playerName"]),
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
        tmp_df = df.iloc[: (amount_player + 1)].copy()

        x_pos = np.array(tmp_df[x_col_bef])
        y_pos = np.array(tmp_df[y_col_bef])

        player_data = dict(
            showlegend=False,
            x=x_pos,
            y=y_pos,
            mode="markers+text",
            text=np.array(tmp_df["playerName"]),
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
            hoverinfo="none",
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
            hoverinfo="none",
        )

        frame = dict(data=[line_data, ball_data, player_data])

        frames.append(frame)

    # compute the layout
    ####################

    field = create_empty_field(below=True, size=size)
    layout = dict(
        # make sure the field is green
        plot_bgcolor=field["layout"]["plot_bgcolor"],
        xaxis=field["layout"]["xaxis"],
        yaxis=field["layout"]["yaxis"],
        shapes=field["layout"]["shapes"],
        width=field["layout"]["width"],
        height=field["layout"]["height"],
        autosize=field["layout"]["autosize"],
        updatemenus=[
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 100, "redraw": False},
                                "fromcurrent": True,
                            },
                        ],
                    ),
                    dict(
                        label="Pause",
                        method="animate",
                        args=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
    )

    fig = dict(data=data, layout=layout, frames=frames)

    return fig


def _calculate_bucket_for_position(series, nb_buckets, min_pos_val, max_pos_val):
    """
    Helper function to calculate the bucket for each position
    """
    buckets = np.arange(min_pos_val, max_pos_val + 0.001, max_pos_val / nb_buckets)

    df_buckets = pd.DataFrame()
    df_buckets["id"] = np.arange(len(buckets) - 1)
    df_buckets["minValueZone"] = list(buckets)[:-1]
    df_buckets["maxValueZone"] = list(buckets)[1:]
    df_buckets["meanValueZone"] = (
        df_buckets["minValueZone"] + df_buckets["maxValueZone"]
    ) / 2

    buckets[-1] = buckets[-1] + 0.001

    return pd.cut(series, buckets, labels=False, include_lowest=True), df_buckets


def prepare_heatmap(
    df,
    col_x,
    col_y,
    nb_buckets_x,
    nb_buckets_y,
    agg_type="count",
    agg_col=None,
    return_df=False,
    length_field=105,
    width_field=68,
    tracking_data=False,
):
    """
    Helper function to prepare a heatmap. It is most often used in combination with the function *create_heatmap*
    below.
    :param df: (pd.DataFrame) Data frame containing all the relevant data
    :param col_x: (str) Column indicating the position in x-direction
    :param col_y: (str) Column indicating the position in y-direction
    :param nb_buckets_x: (int) Split the field into *nb_buckets_x* buckets in x-direction
    :param nb_buckets_y: (int) Split the field into *nb_buckets_y* buckets in y-direction
    :param agg_type: (str) Aggregation type, e.g. mean, median etc. If None, if defaults to *count*
    :param agg_col: (str) Column name for which aggregation should be made. If None, number of appearances per grid
                     cell are computed
    :param return_df: (bool) If True, function returns *df* with additional columns indicating the grid cell
    :param length_field (int) Length of the field in meters
    :param width_field: (int) Width of the field in meters
    :param tracking_data: (bool) Whether the underlying data is tracking data or not
    :return: Returns three np.arrays for
            1. The center points of the grid cells in x-direction
            2. The center points of the grid cells in y-direction
            3. The values for each grid cell
    """

    df = df.copy()

    if tracking_data:
        df[col_y] = -1 * (df[col_y] - width_field / 2) + width_field / 2

    df[col_x + "Zone"], df_lookup_x_buckets = _calculate_bucket_for_position(
        df[col_x], nb_buckets_x, 0, length_field
    )
    df[col_y + "Zone"], df_lookup_y_buckets = _calculate_bucket_for_position(
        df[col_y], nb_buckets_y, 0, width_field
    )

    if agg_col is None:
        agg_col = col_x + "Zone"

    df_pos = (
        df.groupby([col_x + "Zone", col_y + "Zone"])
        .agg(aggVal=(agg_col, agg_type))
        .reset_index()
    )

    df_all_pos = pd.DataFrame(
        [(x, y) for x in df_lookup_x_buckets["id"] for y in df_lookup_y_buckets["id"]],
        columns=[col_x + "Zone", col_y + "Zone"],
    )

    df_lookup_x_buckets.rename(
        columns={"id": col_x + "Zone", "meanValueZone": col_x + "ZoneMean"},
        inplace=True,
    )
    df_lookup_y_buckets.rename(
        columns={"id": col_y + "Zone", "meanValueZone": col_y + "ZoneMean"},
        inplace=True,
    )

    df_all_pos = pd.merge(
        df_all_pos,
        df_lookup_x_buckets[[col_x + "Zone", col_x + "ZoneMean"]],
        how="left",
    )
    df_all_pos = pd.merge(
        df_all_pos,
        df_lookup_y_buckets[[col_y + "Zone", col_y + "ZoneMean"]],
        how="left",
    )

    df_pos = pd.merge(df_all_pos, df_pos, how="left").fillna(0)
    df_img = df_pos.pivot(col_y + "ZoneMean", col_x + "ZoneMean", "aggVal")

    x = list(df_img.columns)
    y = [width_field - x for x in df_img.index]

    img = np.array(df_img)

    if return_df:
        return img, x, y, df

    return img, x, y


def create_heatmap(
    x,
    y,
    z,
    dict_info,
    title_name=None,
    colour_scale=None,
    zsmooth=False,
    legend_name=None,
    size=1,
):
    """
    Function to create a coloured heatmap on top of a soccer field
    :param x: (np.array) Center points of the grid cells in x-direction, i.e. length of the field
    :param y: (np.array) Center points of the grid cells in y-direction, i.e. width of the field
    :param z: (np.array) Two-dimensional array containing the values for all grid cells
    :param dict_info: (dict) Defines what and how information should be shown when hovering over the grid cells.
                       If None, no information is displayed
    :param title_name: (str) Title to be added above the plot
    :param colour_scale: (tuple) Contains the min and max values for the colour scale
    :param zsmooth: (str or False) Smoothing parameter as used by go.Heatmap
    :param legend_name: (str) Name to be added on top of the colour legend bar
    :param size: (float) Relative size of the field
    :return: go.Figure with a heatmap plotted on top of the soccer field
    """

    if dict_info is not None:
        # Prepare the text to be shown when hovering over the heatmap
        hovertext = list()
        for idy in range(len(z)):
            hovertext.append(list())
            for idx in range(len(z[1])):
                text = ""
                for key in dict_info.keys():
                    text += "{}: {:^{display_type}}<br />".format(
                        key,
                        dict_info[key]["values"][idy][idx],
                        display_type=dict_info[key]["display_type"],
                    )
                hovertext[-1].append(text)

    # get the empty soccer field
    fig = create_empty_field(colour="white", line_colour="white", size=size)

    # overlay field with the heatmap

    # if no information should be displayed
    if dict_info is None:
        fig.add_trace(go.Heatmap(x=x, y=y, z=z, zsmooth=zsmooth, hoverinfo="none"))
    # if some information should be displayed
    else:
        fig.add_trace(
            go.Heatmap(x=x, y=y, z=z, zsmooth=zsmooth, hoverinfo="text", text=hovertext)
        )

    if colour_scale is not None:
        fig["data"][-1]["zmin"] = colour_scale[0]
        fig["data"][-1]["zmax"] = colour_scale[1]

    if title_name is not None:
        fig.update_layout(
            title={
                "text": title_name,
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )

    if legend_name is not None:
        fig.update_layout(
            annotations=[
                dict(
                    x=1.07,
                    y=1.03,
                    align="right",
                    valign="top",
                    text=legend_name,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="top",
                )
            ]
        )

    return fig


def get_match_title(match_id, df_matches, df_teams, perspective="home"):
    """
    Given the *match_id* function computes a match title such as "Bayern München - Bayer Leverkusen 3:1"
    :param match_id: (int) Identifier of the match
    :param df_matches: (pd.DataFrame) Data frame with matches (needs to contain *match_id*)
    :param df_teams: (pd.DataFrame) Data frame with the teams
    :param perspective: (str) Whether the title should be written from the home team or the away team perspective.
                         Only accepts values "home" and "away".
    :return: String with the match title
    """

    # get relevant information of the match
    df_match = df_matches[df_matches["matchId"] == match_id]
    row = df_match[df_match["side"] == "home"].iloc[0]

    # get the team names and the score
    home_team_name = ed_help.get_team_name(df_teams, row["teamId"])
    away_team_name = ed_help.get_team_name(df_teams, row["oppTeamId"])
    home_score = row["score"]
    away_score = row["oppScore"]

    # depending on the perspective, compute the match title
    if perspective == "away":
        match_title = f"{away_team_name} @ {home_team_name} {away_score}:{home_score}"
    else:
        match_title = f"{home_team_name} - {away_team_name} {home_score}:{away_score}"

    return match_title


def prepare_passes_for_position_plot(
    df_events, df_stats, show_top_k_percent=None, view_90_min=False
):
    """
    Function to prepare the passes for the position plot (see *create_position_plot*). It does so by aggregating the
    accurate passes between any two players that appear in *df_stats*
    :param df_events: (pd.DataFrane) Data frame with all events of e.g. the match. Notice that it should not only
                       contain passing events but all events
    :param df_stats: (pd.DataFrame) Data frame with player statistics (especially their centroids). Output of the
                      *compute_statics* function can be used
    :param show_top_k_percent: (int, optional) If not None, only the most important passes are returned in a way that
                                *show_top_k_percent* of all passes are being displayed
    :param view_90_min: (bool, optional) If True, passes are rescaled to 90 minutes, i.e. as if the players would have
                        played 90 minutes together. Caveat: Does not work if one player was substituted out and the other
                        substituted in and they only had a short overlap.
    :return: pd.DataFrame with passes between any two players. This data frame can be directly used in the
             *create_position_plot* function
    """
    if df_stats["teamId"].nunique() > 1:
        raise ValueError("Position plot should only contain 1 team")

    team_id = df_stats["teamId"].unique()[0]

    # get the number of passes between any two players
    df_passes = ed_help.number_of_passes_between_players(df_events, team_id)

    # only consider players that are plotted
    players = df_stats["playerId"].unique()
    df_passes = df_passes[
        df_passes["player1Id"].isin(players) & df_passes["player2Id"].isin(players)
    ]

    # get the centroid for each of the players
    df_centroid = df_stats[
        ["playerId", "centroidX", "centroidY", "minutesPlayed"]
    ].copy()

    # add the position of player 1
    df_pos_player1 = df_centroid.rename(
        columns={
            "playerId": "player1Id",
            "centroidX": "centroidX1",
            "centroidY": "centroidY1",
            "minutesPlayed": "minutesPlayed1",
        }
    )
    df_pass_share = pd.merge(df_passes, df_pos_player1, on="player1Id")

    # add the position of player 2
    df_pos_player2 = df_centroid.rename(
        columns={
            "playerId": "player2Id",
            "centroidX": "centroidX2",
            "centroidY": "centroidY2",
            "minutesPlayed": "minutesPlayed2",
        }
    )
    df_pass_share = pd.merge(df_pass_share, df_pos_player2, on="player2Id")

    if view_90_min:
        df_pass_share["totalPasses"] = (
            df_pass_share["totalPasses"]
            / df_pass_share[["minutesPlayed1", "minutesPlayed2"]].min(axis=1)
            * 90
        )

    # compute the share of the passes for each player tuple
    df_pass_share["sharePasses"] = df_pass_share["totalPasses"] / sum(
        df_pass_share["totalPasses"]
    )

    # return only those passes such that *show_top_k_percent* of all passes are being returned
    if show_top_k_percent is not None:
        df_pass_share.sort_values("sharePasses", inplace=True, ascending=False)
        df_pass_share["cumShare"] = df_pass_share["sharePasses"].cumsum()
        df_pass_share = df_pass_share[
            df_pass_share["cumShare"] * 100 < show_top_k_percent
        ].copy()

    return df_pass_share


def create_position_plot(
    df_stats,
    title=None,
    dict_info=None,
    colour_kpi=None,
    colour_scale=None,
    df_passes=None,
    convex_hull=False,
    size=1,
):
    """
    Creation of the position plot, i.e. each player of the team is plotted on the field depending on their position.
    :param df_stats: (pd.DataFrame) Data frame with stats on each player. Usually output of *compute_statistics* (see
                      above)
    :param title: (str, optional) Title for the plot
    :param dict_info: (dict, optional) Defined what and how information should be shown when hovering over the players.
                       If none, some default information will be shown
    :param colour_kpi: (str, optional) Column name of *df_stats* that sets the colour of the markers
    :param colour_scale: (tuple, optional) Tuple with the cmin and cmax of the colour scale
    :param df_passes: (pd.DataFrame, optional) If not None, passes between the players will be displayed. Usually
                       output of the *prepare_passes_for_position_plot* function
    :param convex_hull: (bool) If True, a convex hull around all field players will be drawn
    :param size: (float) Relative size of the plot
    :return: go.Figure with the position plot
    """
    # Default dictionary for hover information
    default_dict = {
        "Player name": {"values": "playerName"},
        "Total passes": {"values": "totalPasses", "display_type": ".0f"},
        "Total passes/90": {"values": "totalPasses90", "display_type": ".0f"},
        "Accurate passes (in %)": {
            "values": "shareAccuratePasses",
            "display_type": ".1f",
        },
        "Total shots": {"values": "totalShots", "display_type": ".0f"},
        "Total goals": {"values": "totalGoals", "display_type": ".0f"},
        "Total duels": {"values": "totalDuels", "display_type": ".0f"},
        "Minutes played": {"values": "minutesPlayed", "display_type": ".0f"},
        "Total distance": {"values": "totalDistance", "display_type": ".1f"},
        "Total distance/90": {"values": "totalDistance90", "display_type": ".1f"},
        "Max speed (km/h)": {"values": "maxSpeed", "display_type": ".2f"},
    }

    df_stats = df_stats.copy()

    # make sure the centroids
    df_stats["centroidY"] = 68 - df_stats["centroidY"]

    field = create_empty_field(below=True, size=size)

    # compute the convex hull and add it to the plot
    ################################################
    if convex_hull:
        centroids = df_stats[df_stats["playerPosition"] != "GK"][
            ["centroidX", "centroidY"]
        ].to_numpy()
        hull = scipy.spatial.ConvexHull(centroids)
        convex_x, convex_y = centroids[hull.vertices, 0], centroids[hull.vertices, 1]

        field.add_trace(
            go.Scatter(
                x=convex_x,
                y=convex_y,
                fill="toself",
                mode="lines",
                showlegend=False,
                fillcolor="black",
                opacity=0.2,
                name="",
                line_color="black",
            )
        )

    # add passes between the players
    ################################

    # only add if the passes between the different players should be displayed
    if df_passes is not None:
        df_passes = df_passes.copy()

        df_passes["centroidY1"] = 68 - df_passes["centroidY1"]
        df_passes["centroidY2"] = 68 - df_passes["centroidY2"]
        for _, row in df_passes.iterrows():
            field.add_trace(
                go.Scatter(
                    showlegend=False,
                    x=[row["centroidX1"], row["centroidX2"]],
                    y=[row["centroidY1"], row["centroidY2"]],
                    mode="lines",
                    line=dict(color="red", width=50 * row["sharePasses"]),
                )
            )

    # set the actual plot
    ############################
    field.add_trace(
        go.Scatter(
            x=df_stats["centroidX"],
            y=df_stats["centroidY"],
            mode="markers+text",
            text=df_stats["playerName"],
            textposition="bottom center",
            name="",
            marker=dict(color="red", size=12),
        )
    )

    # create and add the hover information
    ######################################
    if dict_info is None:
        dict_info = default_dict

    if dict_info is not False:
        hovertext = list()
        for i, row in df_stats.iterrows():
            text = ""
            for key in dict_info.keys():
                # make sure that the column really exists
                if dict_info[key]["values"] not in row:
                    continue

                # check whether there is a display type or not
                if "display_type" in dict_info[key].keys():
                    text += "{}: {:^{display_type}}<br />".format(
                        key,
                        row[dict_info[key]["values"]],
                        display_type=dict_info[key]["display_type"],
                    )
                else:
                    text += "{}: {}<br />".format(key, row[dict_info[key]["values"]])
            hovertext.append(text)

        field.data[-1]["hovertemplate"] = hovertext

    # update the colour of the markers
    if colour_kpi is not None:
        marker = dict(
            color=df_stats[colour_kpi], colorscale="Reds", showscale=True, size=12
        )

        if colour_scale is not None:
            marker["cmin"] = colour_scale[0]
            marker["cmax"] = colour_scale[1]

        field.data[-1]["marker"] = marker

        # get a nice legend name based on the dict_info
        nice_legend_name = [
            key for key in dict_info if dict_info[key]["values"] == colour_kpi
        ]
        if len(nice_legend_name) > 0:
            legend_name = nice_legend_name[0]
        else:
            legend_name = colour_kpi

        field.update_layout(
            annotations=[
                dict(
                    x=1.07,
                    y=1.03,
                    align="right",
                    valign="top",
                    text=legend_name,
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    xanchor="center",
                    yanchor="top",
                )
            ]
        )

    # add the title
    ###############
    if title is not None:
        field.update_layout(
            title={
                "text": title,
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )

    return field


def _hex_to_rgb(hex):
    """
    Helper function to convert hex colour into RGB vector
    """
    # Pass 16 to the integer function for change of base
    return [int(hex[i : i + 2], 16) for i in range(1, 6, 2)]


def _rgb_to_hex(rgb):
    """
    Helper function to convert RGB colour vector into hex
    """
    rgb = [int(x) for x in rgb]
    return "#" + "".join(
        ["0{0:x}".format(v) if v < 16 else "{0:x}".format(v) for v in rgb]
    )


def colour_scale(start_hex, finish_hex, n=101):
    """
    Function returns a gradient list of *n* colors between
    the two hex colors *start_hex* and *end_hex*
    :param start_hex: (str) Six-digit color string of the start colour, e.g. #FFFFFF"
    :param finish_hex: (str) Six-digit color string of the start colour, e.g. #FFFFFF"
    :param n: (int) Number of colours to be produced
    """
    # Starting and ending colors in RGB form
    s = _hex_to_rgb(start_hex)
    f = _hex_to_rgb(finish_hex)
    # Initilize a list of the output colors with the starting color
    rgb_list = [s]
    # Calcuate a color at each evenly spaced value of t from 1 to n
    for t in range(1, n):
        # Interpolate RGB vector for color at the current value of t
        curr_vector = [
            int(s[j] + (float(t) / (n - 1)) * (f[j] - s[j])) for j in range(3)
        ]
        # Add it to our list of output colors
        rgb_list.append(curr_vector)

    return [_rgb_to_hex(RGB) for RGB in rgb_list]


def prepare_pass_polar_plot(
    df, group_cols, length_scale_col, colour_col, colour_scale=None, centroids_xy=None
):
    """
    Preparation function for the polar plot (see function *create_pass_polar_plot*. One polar will be drawn per unique
    combination of *group_cols*
    :param df: (pd.DataFrame) Data frame containing event data
    :param group_cols: (list) Columns used for grouping the event, e.g. the grids from a heatmap, player, team, ...
    :param length_scale_col: (str) Column that defines the length of the triangles in the polar plot
    :param colour_col: (str) Column that defines the colour of the triangles in the polar plot
    :param colour_scale: (tuple, optional) Tuple with the min and max values of the colour scale
    :param centroids_xy: (list of arrays) Centroids of the different polars, e.g. the centroids of the grids of a
                          heatmap
    :return: pd.DataFrame containing all relevant information for using the *create_pass_polar_plot* function
    """

    df = df.copy()

    # compute the length of each pass
    df = ed_help.compute_length(df)

    # compute the degree for each pass - notice that
    #   0 degrees = forward
    #   90 degrees = to the right
    #   - 90 degrees = to the left
    #   180 or -180 degrees = backwards
    df["dx"] = df["posAfterXMeters"] - df["posBeforeXMeters"]
    df["dy"] = df["posAfterYMeters"] - df["posBeforeYMeters"]
    df["degree"] = df.apply(
        lambda row: math.degrees(math.atan2(row["dy"], row["dx"])), axis=1
    )

    # put the degrees into 45 degree bins
    bins = np.arange(-202.5, 203, 45)
    labels = np.arange(-180, 181, 45)
    df["degreeBin"] = pd.cut(df["degree"], bins=bins, labels=labels)
    df["degreeBin"] = np.where(df["degreeBin"] == -180, 180, df["degreeBin"])

    # group by group_cols and degree bins and compute relevant KPIs
    df_group = (
        df.groupby(group_cols + ["degreeBin"])
        .agg(
            totalPasses=("degreeBin", "size"),
            totalAccuratePasses=("accurate", "sum"),
            shareAccuratePasses=("accurate", "mean"),
            meanLengthMeters=("lengthMeters", "mean"),
        )
        .reset_index()
    )
    df_group["shareAccuratePasses"] *= 100

    # compute the scale factor (in [0,1]) which will define the length of each arrow in the polar graph
    df_group["scaleFactor"] = (
        df_group[length_scale_col] / df_group[length_scale_col].max()
    )

    # compute the centroid for each of the polar graphs
    # in case the centroid is given externally
    if centroids_xy is not None:

        x_vals = centroids_xy[0]
        y_vals = centroids_xy[1]

        df_group["centroidX"] = df_group["posBeforeXMetersZone"].map(
            lambda i: x_vals[i]
        )
        df_group["centroidY"] = df_group["posBeforeYMetersZone"].map(
            lambda i: y_vals[i]
        )

    # or the centroid is already part of *df*
    elif "centroidX" in df.columns and "centroidY" in df.columns:
        df_centroid = df.groupby(group_cols).agg(
            centroidX=("centroidX", "min"), centroidY=("centoridY", "min")
        )

        df_group = pd.merge(df_group, df_centroid, how="left", on=group_cols)

    else:
        raise ValueError(
            "Centroids must either be part of *df* or given externally through *centroids_xy*"
        )

    # set the colour value (number between 0 and 100) depending on the colour kpi
    df_group["colourCol"] = df_group[colour_col].copy()

    if colour_scale is not None:
        df_group["colourCol"] = df_group["colourCol"].clip(lower=colour_scale[0])
        df_group["colourCol"] = df_group["colourCol"].clip(upper=colour_scale[1])

    max_val = df_group["colourCol"].max()
    min_val = df_group["colourCol"].min()

    df_group["colourValue"] = (
        (df_group["colourCol"] - min_val) / (max_val - min_val) * 100
    )

    return df_group


def create_pass_polar_plot(df, dict_info=None, title_name=None, size=1):
    """
    Function creates the pass polar plot which indicates the direction in which passes where being made. It also
    conveys other information by using size and colour. Before running this function one usually needs to run the
    *prepare_pass_polar_plot* function
    :param df: (pd.DataFrame) Contains all relevant data; usually the output of the *prepare_pass_polar_plot* function
    :param dict_info: (dict, optional) Defines what and how information should be shown when hovering over the players.
    :param title_name: (str) Title of the graph
    :param size: (float) Relative size of the plot
    :return: go.Figure containing the pass polar plot
    """

    # create the hover text - there will be hover text for each triangle
    hovertext = list()
    for i, row in df.iterrows():
        text = ""
        if dict_info is None:
            continue
        for key in dict_info.keys():
            # make sure that the column really exists
            if dict_info[key]["values"] not in row:
                continue

            # check whether there is a display type or not
            if "display_type" in dict_info[key].keys():
                text += "{}: {:^{display_type}}<br />".format(
                    key,
                    row[dict_info[key]["values"]],
                    display_type=dict_info[key]["display_type"],
                )
            else:
                text += "{}: {}<br />".format(key, row[dict_info[key]["values"]])
        hovertext.append(text)

    # create a continuous colour scale from white to red - will be the colour of the triangles
    white = "#FFFFFF"
    red = "#FF0000"
    colours = colour_scale(white, red, n=101)

    # create the empty field
    fig = create_empty_field(below=True, size=size)

    # loop through all the different centroid / degree combinations and draw the triangles
    for i, row in df.iterrows():

        x_center = row["centroidX"]
        y_center = row["centroidY"]

        # get the length in x and y direction
        y_length = 10 * row["scaleFactor"]
        x_length = 3 * (y_length / 10)

        # set up the triangle
        tri_x = [-x_length / 2, 0, x_length / 2, -x_length / 2]
        tri_y = [-y_length, 0, -y_length, -y_length]

        # rotate the triangle according to the degree
        rot_x, rot_y = gen_help.rotate_vectors(
            tri_x, tri_y, math.radians(row["degreeBin"] - 90)
        )

        # add the center point to end up in the centroid
        x_vals = x_center + np.array(rot_x)
        y_vals = y_center + np.array(rot_y)

        # add the triangles with the hover text as traces
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                showlegend=False,
                mode="lines",
                fill="toself",
                fillcolor=colours[int(row["colourValue"])],
                text=hovertext[i],
                hoverinfo="text",
                line_color=colours[int(row["colourValue"])],
            )
        )

    # add all the black dots in the middle of the polar plots
    df_dots = df[["centroidX", "centroidY"]].drop_duplicates()

    for i, row in df_dots.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["centroidX"]],
                y=[row["centroidY"]],
                fillcolor="black",
                line_color="black",
                showlegend=False,
                hoverinfo="skip",
                marker=dict(size=10),
            )
        )

    # add a title to the chart
    if title_name is not None:
        fig.update_layout(
            title={
                "text": title_name,
                "y": 0.9,
                "x": 0.5,
                "xanchor": "center",
                "yanchor": "top",
            }
        )

    return fig
