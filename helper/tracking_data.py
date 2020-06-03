# -*- coding: utf-8 -*-

# import packages

import math
import pandas as pd
import numpy as np
import scipy.signal as signal

FIELD_LENGTH = 105
FIELD_WIDTH = 68


def add_position_delta(df):
    """
    Function computes for each player the position and time difference / delta between the previous frame
    and the current frame
    :param df: (pd.DataFrame) Tracking data, ordered by frame
    :return: pd.DataFrame containing the delta to the previous frame
    """

    players = df["playerId"].unique()
    periods = df["period"].unique()

    # loop through all players and compute the difference to the previous frame. Notice that this is way faster than
    # running *diff()* on a groupyby object in case of large data frames
    lst_all_players = list()
    for playerId in players:
        for period in periods:
            df_player = df[
                (df["playerId"] == playerId) & (df["period"] == period)
            ].copy()
            df_player["dx"] = df_player["xPos"].diff()
            df_player["dy"] = df_player["yPos"].diff()
            df_player["dt"] = df_player["time"].diff()

            lst_all_players.append(df_player)

    df_all_players = pd.concat(lst_all_players)
    df_all_players.sort_values(["frame", "playerId"], inplace=True)
    return df_all_players


def add_speed(df):
    """
    Compute the speed in km/h between two frames
    :param df: (pd.DataFrame) Tracking data containing the position and time deltas
    :return: pd.DataFrame with tracking data containing the speed for each frame
    """

    print("Do not use this any more but consider using *add_player_velocities*")

    # in km/h
    if "dx" not in df.columns or "dy" not in df.columns or "dt" not in df.columns:
        raise ValueError(
            "Requires columns *dx*, *dy* and *dt*. Need to run function *add_position_delta* beforehand."
        )

    df["speed"] = (
        np.sqrt(df["dx"] * df["dx"] + df["dy"] * df["dy"]) / df["dt"] * 3600 / 1000
    )
    return df


def add_player_velocities(df_track, smoothing=True, window=7, polyorder=1, maxspeed=10):
    """
    :param df_track: (pd.DataFrame) Tracking data
    :param smoothing: (bool) If True, Savatzky-Golay filter is used for smoothing
    :param window: (int) Window used for Savatzky-Golay filter
    :param polyorder: (int) Order of polynom used for Savatzky-Golay filter
    :param maxspeed: (float) Maximum player speed assumed (in m/s)
    :return: pd.DataFrame with tracking data containing player's (smoothed) speed in m/s for each frame
    """

    # compute the raw speed in x- and y-direction
    df_track = add_position_delta(df_track)
    df_track["vx"] = df_track["dx"] / df_track["dt"]
    df_track["vy"] = df_track["dy"] / df_track["dt"]

    # if no smoothing is required we just return the raw speed
    if not smoothing:
        return df_track

    lst_all_players = list()

    for tmp_player in df_track["playerId"].unique():

        df_player = df_track[df_track["playerId"] == tmp_player].copy()

        # do not clean the ball speed
        if tmp_player != -1:
            df_player["rawSpeed"] = np.sqrt(
                df_player["vx"] * df_player["vx"] + df_player["vy"] * df_player["vy"]
            )

            # if raw speed is faster than *maxspeed* we assume a data error
            df_player["vx"] = np.where(
                df_player["rawSpeed"] > maxspeed, np.nan, df_player["vx"]
            )
            df_player["vy"] = np.where(
                df_player["rawSpeed"] > maxspeed, np.nan, df_player["vy"]
            )

            # use Savatzky-Golay filter for fitting
            vx_sav = signal.savgol_filter(
                df_player["vx"], window_length=window, polyorder=polyorder
            )
            vy_sav = signal.savgol_filter(
                df_player["vy"], window_length=window, polyorder=polyorder
            )

            df_player["vx"] = vx_sav
            df_player["vy"] = vy_sav

        lst_all_players.append(df_player)

    df_all_players = pd.concat(lst_all_players)
    df_all_players["vx"].fillna(0, inplace=True)
    df_all_players["vy"].fillna(0, inplace=True)
    df_all_players.drop(["dx", "dy", "dt", "rawSpeed"], axis=1, inplace=True)

    return df_all_players


def add_angle_of_direction(df):
    """
    Function computes the angle of the direction between the previous frame and the current frame
    :param df: (pd.DataFrame) Data frame containing position deltas
    :return: pd.DataFrame with an additional *angle* column containing the angle
    """
    if "dx" not in df.columns or "dy" not in df.columns:
        raise ValueError(
            "Requires columns *dx* and *dy*. Need to run function *add_position_delta* beforehand."
        )

    df["angle"] = df.apply(
        lambda row: math.degrees(math.atan2(row["dy"], row["dx"])), axis=1
    )
    return df


def add_touch_id(df, min_angle_change=1):
    """
    Function to identify touches based on the ball data
    :param df: (pd.DataFrame) Data frame with tracking data containing either the ball or one player only
    :param min_angle_change: (float) A touch is identifed by an angle change of more than *min_angle_change* degrees
    :return: pd.DataFrame with an additional column containing an touchId identifying the different touches
    """
    df = df.copy()

    if len(df["playerId"].unique()) > 1:
        raise ValueError("Data can only contain the ball or one player")

    # compute the angle only if it is not already in the data frame
    angle_in_input = "angle" in df.columns
    if not angle_in_input:
        df = add_angle_of_direction(df)

    # get the change in the angle
    df["dAngle"] = df["angle"].diff()

    # flag all frames with an angle change of > *min_angle_change* degress
    df["angleChange"] = 1 * (
        (np.abs(df["dAngle"]) > min_angle_change) | df["angle"].isnull()
    )

    # fill in the touchId
    df["touchId"] = df["angleChange"].cumsum()

    df.drop(["dAngle", "angleChange"], axis=1, inplace=True)

    if not angle_in_input:
        df.drop("angle", axis=1, inplace=True)

    return df


def add_events_to_tracking_data(df, df_events, cols_events="all"):
    """
    Function to attach the event data to the tracking data
    :param df: (pd.DataFrame) Data frame with tracking data
    :param df_events: (pd.DataFrame) Data frame with event data
    :param cols_events: (list) Columns to be kept from the event data
    :return: pd.DataFrame with the tracking data but enriched by the event data
    """
    df_events = df_events.sort_values(["startFrame", "endFrame"]).copy()
    # keep only one event per frame
    df_events.drop_duplicates("startFrame", keep="last", inplace=True)

    # only keep the necessary columns
    if cols_events == "all":
        cols_events = list(df_events.columns)

    if "startFrame" not in cols_events:
        cols_events.append("startFrame")

    df_events = df_events[cols_events].copy()

    # rename the columns to always start with *event*
    colnames_new = ["event" + col[0].upper() + col[1:] for col in df_events.columns]
    df_events.columns = colnames_new

    # merge the event data to the tracking data
    df_events.rename(columns={"eventStartFrame": "frame"}, inplace=True)
    df = pd.merge(df, df_events, how="left", on="frame")

    return df


def compute_distance_to_point(df_players, df_point, point_coords, colname_out=None):
    """
    Function to compute the distance (in meters) to a point for each frame and different *playerId* in *df_players*
    :param df_players: (pd.DataFrame) Data frame containing tracking data
    :param df_point: (pd.DataFrame) Data frame containing one positional data point for each frame, it could e.g.
                     be the position of the ball for each frame
    :param point_coords: (set) Columns names of the x-coord and y-coord in the *df_point* data frame
    :param colname_out: (str) Name of the distance column to be output; "distToPoint" if colname_out=None
    :return: pd.DataFrame containing all data from *df_players* and an additional column indicating the distance to
            the point in *df_point* for each frame
    """
    df_players["xxx_id_xxx"] = np.arange(len(df_players))
    df = df_players.copy()
    df_point = df_point.copy()

    if len(df_point) != len(df_point["frame"].unique()):
        return ValueError("Multiple entries for same frame in *df_point*")

    if colname_out is None:
        colname_out = "distToPoint"

    df_point.rename(
        columns={point_coords[0]: "xxxPos", point_coords[1]: "yyyPos"}, inplace=True
    )
    df = pd.merge(df, df_point[["frame", "xxxPos", "yyyPos"]], how="left", on="frame")

    df["dx"] = df["xPos"] - df["xxxPos"]
    df["dy"] = df["yPos"] - df["yyyPos"]

    # function to compute the distance in meters
    df[colname_out] = np.sqrt(df["dx"] * df["dx"] + df["dy"] * df["dy"])

    df_players = pd.merge(df_players, df[["xxx_id_xxx", colname_out]], on="xxx_id_xxx")
    df_players.drop("xxx_id_xxx", axis=1, inplace=True)

    return df_players


def closest_player_to_point(df_players, df_point, point_coords, colname_out=None):
    """
    Compute the player with the shortest distance to a point for each frame
    :param df_players: (pd.DataFrame) Data frame containing tracking data
    :param df_point: (pd.DataFrame) Data frame containing one positional data point for each frame, it could e.g.
                     be the position of the ball for each frame
    :param point_coords: (set) Columns names of the x-coord and y-coord in the *df_point* data frame
    :param colname_out: (str) Name of the distance column to be output; "closestPlayerToPoint" if colname_out=None
    :return: pd.DataFrame containing all data from *df_point* and two additional columns indicating the distance (in m)
             and the name of the closest player
    """

    if colname_out is None:
        colname_out = "closestPlayerToPoint"

    # compute the distance to the point for all players
    df = compute_distance_to_point(df_players, df_point, point_coords)

    # for each frame get the player with the minimal distance
    df_min_dist = (
        df.groupby("frame").agg(minDistToPoint=("distToPoint", "min")).reset_index()
    )

    df = pd.merge(df, df_min_dist, on="frame")

    df_player_min = df[df["distToPoint"] == df["minDistToPoint"]].copy()

    df_player_min = (
        df_player_min.groupby(["frame"])["playerId"]
        .agg(lambda x: ",".join([str(y) for y in sorted(x)]))
        .reset_index()
    )
    df_player_min.columns = ["frame", colname_out]

    # attach the minimal distance and the player with the minimal distance
    df_point = pd.merge(df_point, df_min_dist, on="frame")
    df_point = pd.merge(df_point, df_player_min, on="frame")

    return df_point


def transform_position_to_event_format(df, x_col, y_col):
    """
    Helper function to transform the position of the tracking data into the format used by the event data
    """
    df = df.copy()

    df[x_col] = np.where(
        df["teamId"] == 1,
        df[x_col],
        -1 * (df[x_col] - FIELD_LENGTH / 2) + FIELD_LENGTH / 2,
    )
    df[y_col] = np.where(
        df["teamId"] == 1,
        -1 * (df[y_col] - FIELD_WIDTH / 2) + FIELD_WIDTH / 2,
        df[y_col],
    )

    return df


def add_long_break_identifier(df_track, df_events, min_secs_long_break=20):
    """
    Function to identify breaks longer than *min_secs_long_break* seconds in the game and attach the information
    to the tracking data
    :param df_track: (pd.DataFrame) Data frame containing the tracking data
    :param df_events: (pd.DataFrame) Data frame containing event data
    :param min_secs_long_break: (float) Number of seconds without event to be considered a long break
    :return:
    """

    # get the start point of the following event for all events
    df_events = df_events.copy()
    df_events.sort_values(["startFrame", "endFrame"], inplace=True)
    df_events["eventSecAfter"] = df_events["eventSec"].shift(-1)
    df_events["startFrameAfter"] = df_events["startFrame"].shift(-1)
    df_events["secsToNextEvent"] = df_events["eventSecAfter"] - df_events["eventSec"]

    # get breaks by considering breaks longer than *min_secs_long_break* seconds
    df_break = df_events[df_events["secsToNextEvent"] > min_secs_long_break].copy()

    # build a data frame containing all frames that are within any of the "long breaks"
    lst_break_frames = list()
    for i, row in df_break.iterrows():
        df = pd.DataFrame(
            {"frame": np.arange(row["startFrame"], row["startFrameAfter"])}
        )
        df["longBreak"] = 1

        lst_break_frames.append(df)
    df_long_breaks = pd.concat(lst_break_frames)
    df_long_breaks["frame"] = df_long_breaks["frame"].astype(int)

    # indicate for each frame whether it belongs to a long frame
    if "longBreak" in df_track.columns:
        df_track.drop("longBreak", axis=1, inplace=True)

    df_track = pd.merge(df_track, df_long_breaks, how="left", on="frame")
    df_track["longBreak"].fillna(0, inplace=True)
    df_track["longBreak"] = df_track["longBreak"].astype(int)

    return df_track


def compute_players_behind_ball(df_track):
    """
    Compute the number of players behind the ball for each team and each frame
    :param df_track: (pd.DataFrame) Data frame containing tracking data
    :return: pd.DataFrame containing the number of players behind the ball for each team and frame
    """
    df = df_track.copy()

    # compute the distance to the home goal
    df_home_goal = df_track[["frame"]].drop_duplicates()
    df_home_goal["xPos"] = 0
    df_home_goal["yPos"] = FIELD_WIDTH / 2
    df = compute_distance_to_point(
        df, df_home_goal, ("xPos", "yPos"), "distanceToHomeGoal"
    )

    # compute the distance to the away goal
    df_away_goal = df_home_goal.copy()
    df_away_goal["xPos"] = FIELD_LENGTH
    df_away_goal["yPos"] = FIELD_WIDTH / 2
    df = compute_distance_to_point(
        df, df_away_goal, ("xPos", "yPos"), "distanceToAwayGoal"
    )

    # split the data into ball data and player data
    df_ball = df[df["playerId"] == -1].copy()
    df_players = df[df["playerId"] != -1].copy()

    # check whether the ball was closer to the goal than the defender
    df_ball = df_ball[["frame", "distanceToHomeGoal", "distanceToAwayGoal"]].copy()

    df_ball.rename(
        columns={
            "distanceToHomeGoal": "ballDistanceToHomeGoal",
            "distanceToAwayGoal": "ballDistanceToAwayGoal",
        },
        inplace=True,
    )
    df_players = pd.merge(df_players, df_ball, how="left", on="frame")

    df_players["potentialDefender"] = 1 * np.where(
        df_players["team"] == "Home",
        df_players["distanceToHomeGoal"] < df_players["ballDistanceToHomeGoal"],
        df_players["distanceToAwayGoal"] < df_players["ballDistanceToAwayGoal"],
    )

    # get the total number of players behind the ball
    df_players_behind_ball = (
        df_players.groupby(["frame", "team"])
        .agg(playersBehindBall=("potentialDefender", "sum"))
        .reset_index()
    )

    df_players_behind_ball = pd.merge(df_players_behind_ball, df_ball, how="left")
    return df_players_behind_ball


def compute_packed_players_per_event(df_events, df_def):
    """
    Compute the packed players for each event in *df_events*
    :param df_events: (pd.DataFrame) Events for which packing KPI should be computed
    :param df_def: (pd.DataFrame) Data frame with number of players behind the ball for each frame
                   (see *compute_players_behind_ball*)
    :return: pd.DataFrame with number of packed players for each event
    """
    df_events["oppTeam"] = np.where(df_events["team"] == "Home", "Away", "Home")

    # compute the number of opponents behind the ball at begin of event
    cols = {
        "team": "oppTeam",
        "frame": "startFrame",
        "playersBehindBall": "startPlayersBehindBall",
    }
    df_events = pd.merge(
        df_events,
        df_def[cols.keys()].rename(columns=cols),
        on=["oppTeam", "startFrame"],
    )

    # compute the number of opponents behind the ball at end of event
    cols = {
        "team": "oppTeam",
        "frame": "endFrame",
        "playersBehindBall": "endPlayersBehindBall",
    }

    df_events = pd.merge(
        df_events, df_def[cols.keys()].rename(columns=cols), on=["oppTeam", "endFrame"]
    )

    df_events["packedPlayers"] = (
        df_events["startPlayersBehindBall"] - df_events["endPlayersBehindBall"]
    ).clip(lower=0)
    df_events.drop("oppTeam", axis=1, inplace=True)

    return df_events
