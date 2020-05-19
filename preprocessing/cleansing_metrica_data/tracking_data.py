# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd

import helper.io as io
import helper.tracking_data as td_help

logging.basicConfig(level=logging.DEBUG)

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)


def extract_ball_data(df_home):
    """
    Helper function to extract the ball position for each frame
    :param df_home: (pd.DataFrame) Data frame containing both the positions of the home team per frame as well as the
                    position of the ball
    :return: pd.DataFrame containing the ball positions in each frame
    """

    # get the ball data
    cols = ["Period", "Frame", "Time [s]"] + ["ball_x", "ball_y"]
    df_ball = df_home[cols].copy()
    df_ball.columns = ["period", "frame", "time", "xPos", "yPos"]
    df_ball.sort_values("frame", inplace=True)
    df_ball["playerId"] = -1
    df_ball["team"] = "Ball"
    return df_ball


def convert_to_long_data_frame(df, teamname):
    """
    Helper function to convert *df* into a long data frame, i.e. instead of having one column for every player,
    we have one row for each player and frame
    """
    players = set([col.split("_")[1] for col in df.columns if col.startswith(teamname)])

    lst_df_long = list()
    colnames_new = ["period", "frame", "time", "xPos", "yPos"]

    for player in players:
        tmp_df = df[
            [
                "Period",
                "Frame",
                "Time [s]",
                f"{teamname}_{player}_x",
                f"{teamname}_{player}_y",
            ]
        ].copy()
        tmp_df.columns = colnames_new

        tmp_df = tmp_df[tmp_df["xPos"].notnull()].copy()

        tmp_df["playerId"] = int(player)
        tmp_df["team"] = teamname

        lst_df_long.append(tmp_df)

    df_long = pd.concat(lst_df_long)

    return df_long


def convert_positions_to_meters(df, field_length=105, field_width=68):
    """
    Helper function to convert the positions to meters
    """

    # convert position into meters
    df["xPos"] -= 0.5
    df["yPos"] -= 0.5

    # make sure the right full-back plays at position close to (0,0)
    df["yPos"] = -1 * df["yPos"]

    # make sure that the teams play into the same direction in both half times
    df["xPos"] = np.where(df["period"] == 2, -1 * df["xPos"], df["xPos"])
    df["yPos"] = np.where(df["period"] == 2, -1 * df["yPos"], df["yPos"])

    df["xPos"] = (df["xPos"] + 0.5) * field_length
    df["yPos"] = (df["yPos"] + 0.5) * field_width

    return df


def identify_frames_with_issue_ball(df, max_speed=135):
    """Identify frames with ball speed > *max_speed* meters/second"""
    return df[df["speed"] > max_speed].copy()


def touch_overview_table(df):
    """
    Compute an overview table for the touches. Each row of the output table will be one touch with information
    about where and when the touch started and where and when the next touch started
    :param df: (pd.DataFrame) Data frame with ball tracking data already containing a touchID
    :return: pd.DataFrame with an overview over all the touches
    """

    # aggregate the ball tracking data to touch level
    df_touches = (
        df.groupby("touchId")
        .agg(frameStart=("frame", "min"), frameEnd=("frame", "max"))
        .reset_index()
    )

    # get the starting point and frame for each touch
    df_points_start = df[["frame", "xPos", "yPos"]].copy()
    df_points_start.columns = [x + "Start" for x in df_points_start.columns]

    # get the starting point of the next touch
    df_touches = pd.merge(df_touches, df_points_start, on="frameStart")
    df_touches["xPosStartNext"] = df_touches["xPosStart"].shift(-1)
    df_touches["yPosStartNext"] = df_touches["yPosStart"].shift(-1)

    # get the number of frames that are involved in the touch
    df_touches["totalFrames"] = df_touches["frameEnd"] - df_touches["frameStart"] + 1

    # compute the steps the ball needs to make in each direction for it to be at the starting point of the new touch
    df_touches["dx"] = (
        df_touches["xPosStartNext"] - df_touches["xPosStart"]
    ) / df_touches["totalFrames"]
    df_touches["dy"] = (
        df_touches["yPosStartNext"] - df_touches["yPosStart"]
    ) / df_touches["totalFrames"]

    return df_touches


def identify_touches_to_fix(df_touches, df_issues):
    """
    Identify all touches that do need to be recomputed to fix the error
    :param df_touches: (pd.DataFrame) Data frame with an overview over all touches
    :param df_issues: (pd.DataFrame) Data frame with all frames in which there is an issue with the ball data
    :return: pd.DataFrame with all touches that need to be fixed
    """
    df_issues = (
        df_issues.groupby("touchId")
        .agg(nbTouches=("touchId", "size"), firstFrameIssue=("frame", "min"))
        .reset_index()
    )

    touches_with_issue = pd.merge(df_touches, df_issues, on="touchId", how="inner")

    touches_to_fix = touches_with_issue[
        touches_with_issue["frameStart"] == touches_with_issue["firstFrameIssue"]
    ].copy()
    touches_to_fix["touchIdBefore"] = touches_to_fix["touchId"] - 1

    df_touches_fix = pd.merge(
        df_touches,
        touches_to_fix[["touchIdBefore"]],
        left_on="touchId",
        right_on="touchIdBefore",
    )

    return df_touches_fix


def compute_new_ball_positions(df):
    """
    Helper function to compute smoothed ball positions to avoid hickups
    """
    lst_new_positions = list()

    for i, row in df.iterrows():
        frame_start = row["frameStart"]
        frames = [int(frame) for frame in np.arange(frame_start, row["frameEnd"] + 1)]
        x_pos_new = row["xPosStart"] + (frames - frame_start) * row["dx"]
        y_pos_new = row["yPosStart"] + (frames - frame_start) * row["dy"]
        tmp_df = pd.DataFrame(
            {"frame": frames, "xPosNew": x_pos_new, "yPosNew": y_pos_new}
        )
        lst_new_positions.append(tmp_df)

    df_new_positions = pd.concat(lst_new_positions)

    return df_new_positions


def set_out_ball(df_ball, game, out_to_nan=15):
    """
    Function to compute whether the ball is out of bounds and/or out of the game
    """

    # ball is out out bounds if the position is out of the field
    df_ball["outOfBounds"] = 1 * (
        (df_ball["xPos"] < 0)
        | (df_ball["xPos"] > 105)
        | (df_ball["yPos"] < 0)
        | (df_ball["yPos"] > 68)
    )

    # ball is considered out if it is out of bounds or there is no ball position
    df_ball["ballOut"] = 1 * (df_ball["outOfBounds"] | df_ball["xPos"].isnull())

    # manually overwrite a couple of situations
    if game == 1:
        out_ball_game = [[38478, 42475]]
    elif game == 2:
        out_ball_game = [[7891, 8147], [21199, 21636]]
    else:
        raise ValueError(
            f"Please double check that no manual out balls need to be set for game {game}"
        )

    for frames in out_ball_game:
        df_ball["ballOut"] = np.where(
            (df_ball["frame"] >= frames[0]) & (df_ball["frame"] <= frames[1]),
            1,
            df_ball["ballOut"],
        )

    # for each frame in which the ball is out attach how long the ball has been out (in frames)
    k = 0
    period = 0
    count_out = list()
    out_periods = list()
    for i, row in df_ball.iterrows():
        if row["ballOut"] == 0:
            k = 0
            out_periods.append(-1)
        else:
            if k == 0:
                period += 1
            k += 1
            out_periods.append(period)
        count_out.append(k)

    df_ball["counterOut"] = count_out
    df_ball["outPeriod"] = out_periods

    # update whether the ball was out of bounds by checking that it was out of bounds in at least one frame
    # during the out-of-play period
    df_out_of_bounds = (
        df_ball.groupby("outPeriod")
        .agg(outOfBounds=("outOfBounds", "max"))
        .reset_index()
    )
    df_ball.drop("outOfBounds", axis=1, inplace=True)
    df_ball = pd.merge(df_ball, df_out_of_bounds, on="outPeriod")
    df_ball.sort_values("frame", inplace=True)

    # if the ball has been out for more than *out_to_nan* frames, we set the position to NaN
    df_ball["xPos"] = np.where(
        df_ball["counterOut"] > out_to_nan, np.nan, df_ball["xPos"]
    )
    df_ball["yPos"] = np.where(
        df_ball["counterOut"] > out_to_nan, np.nan, df_ball["yPos"]
    )

    return df_ball


def cleanse_ball_tracking_data(df_ball, game, nb_out_frames=25):
    """
    Function to specifically cleanse the ball tracking data
    :param df_ball: (pd.DataFrame) Data frame containing the ball tracking data
    :param nb_out_frames: (int) Number of frames to actually consider a ball out of bound and not just a tracking error
    :return: pd.DataFrame with processed ball tracking data
    """

    # convert the positions to meters
    df_ball = convert_positions_to_meters(df_ball)

    # save the initial positions that came with the Metrica data
    df_ball["xPosMetrica"] = df_ball["xPos"].copy()
    df_ball["yPosMetrica"] = df_ball["yPos"].copy()

    # Add a flag telling us for each frame whether the ball is in play or not
    df_ball = set_out_ball(df_ball, game)

    # compute the difference in space to the previous frame
    df_ball = td_help.add_position_delta(df_ball)
    df_ball = td_help.add_speed(df_ball)

    # identify ball touches by looking at the angle and speed difference between two frames
    # one touch might e.g. be a pass (both the speed and the angle do not change considerably during the pass)
    df_ball = td_help.add_touch_id(df_ball)

    # number of frames the ball is out or tracking data it missing
    df_frames_out = (
        df_ball.groupby("outPeriod").agg(nbFramesOut=("frame", "count")).reset_index()
    )
    df_ball = pd.merge(df_ball, df_frames_out, on="outPeriod", how="left")

    # update the touchId for situations with tracking error
    df_ball["touchId"] = np.where(
        (df_ball["nbFramesOut"] < nb_out_frames) & (df_ball["outOfBounds"] == 0),
        np.nan,
        df_ball["touchId"],
    )
    df_ball["touchId"].fillna(method="ffill", inplace=True)

    # update the ball out columns in case there are only few frames missing in the middle of the game
    df_ball["ballOutNew"] = np.where(
        (df_ball["nbFramesOut"] <= nb_out_frames)
        & (df_ball["ballOut"] == 1)
        & (df_ball["outOfBounds"] == 0),
        0,
        df_ball["ballOut"],
    )

    # update the ball position for the situations with missing frames
    df_ball["xPosTemp"] = df_ball["xPos"].copy()
    df_ball["yPosTemp"] = df_ball["yPos"].copy()

    df_ball["xPos"].fillna(method="ffill", inplace=True)
    df_ball["yPos"].fillna(method="ffill", inplace=True)

    df_ball["xPos"] = np.where(
        df_ball["ballOutNew"] != df_ball["ballOut"],
        df_ball["xPos"],
        df_ball["xPosTemp"],
    )
    df_ball["yPos"] = np.where(
        df_ball["ballOutNew"] != df_ball["ballOut"],
        df_ball["yPos"],
        df_ball["yPosTemp"],
    )

    # delete columns that are not needed any more
    df_ball["ballOut"] = df_ball["ballOutNew"]
    df_ball.drop(["ballOutNew", "xPosTemp", "yPosTemp"], axis=1, inplace=True)

    # update the speed calculation to find frames with issues
    df_ball = td_help.add_position_delta(df_ball)
    df_ball = td_help.add_speed(df_ball)

    # update the touchId to have a sequence only increasing by 1 (required by functions below)
    lst_touch_id = list()
    touch_id = 0
    curr_touch = 0
    for val in df_ball["touchId"]:
        if curr_touch != val:
            curr_touch = val
            touch_id += 1
        lst_touch_id.append(touch_id)

    df_ball["touchId"] = lst_touch_id

    # identify frames with very high speed (> 135 km/h), i.e. data errors
    df_issues = identify_frames_with_issue_ball(df_ball)

    # get the touches to these data points with errors
    df_touches = touch_overview_table(df_ball)
    touches_fix = identify_touches_to_fix(df_touches, df_issues)

    # recompute the position of the ball by smoothing the position data for the pass
    df_new_positions = compute_new_ball_positions(touches_fix)

    # update the position information of the ball
    df_ball = pd.merge(df_ball, df_new_positions, how="left", on="frame")
    df_ball.drop(["dx", "dy", "dt", "speed", "touchId"], axis=1, inplace=True)

    # set the update position as the standard one
    df_ball["xPos"] = np.where(
        df_ball["xPosNew"].notnull(), df_ball["xPosNew"], df_ball["xPos"]
    )
    df_ball["yPos"] = np.where(
        df_ball["yPosNew"].notnull(), df_ball["yPosNew"], df_ball["yPos"]
    )

    # delete columns that are not needed anymore
    df_ball.drop(
        ["counterOut", "nbFramesOut", "xPosNew", "yPosNew"], axis=1, inplace=True
    )

    return df_ball


def cleanse_tracking_data(game):
    """
    Function to clean the Metrica tracking data. Despite the obvious steps, the ball data is also cleaned in such a
    way that it floats more smoothly and has less hickups.
    :param game: (int) GameId
    :return: None
    """
    logging.info(f"Cleansing metrica tracking data for game {game}")

    # read the raw data of the home and away team
    df_home = io.read_data(
        "home_team_tracking", league=str(game), data_folder="raw_data_metrica"
    )
    df_away = io.read_data(
        "away_team_tracking", league=str(game), data_folder="raw_data_metrica"
    )

    # extract the ball data and clean it to have less
    df_ball = extract_ball_data(df_home)
    df_ball = cleanse_ball_tracking_data(df_ball, game)

    # convert the data frames into a long format to be able to work with them more easily
    df_home = convert_to_long_data_frame(df_home, "Home")
    df_away = convert_to_long_data_frame(df_away, "Away")

    # convert position into meters
    df_players = pd.concat([df_home, df_away])
    df_players = convert_positions_to_meters(df_players)
    df_players["xPosMetrica"] = df_players["xPos"].copy()
    df_players["yPosMetrica"] = df_players["yPos"].copy()

    # combine player data with ball data
    df_all = pd.concat([df_players, df_ball])
    df_all.sort_values(["frame", "playerId"], inplace=True)

    df_all.drop(["outOfBounds", "ballOut"], axis=1, inplace=True)
    df_all = pd.merge(
        df_all, df_ball[["frame", "outOfBounds", "ballOut"]], on="frame", how="left"
    )

    # consider whether the ball is in play rather than whether it is out
    df_all.rename(columns={"ballOut": "ballInPlay"}, inplace=True)
    df_all["ballInPlay"] = 1 - df_all["ballInPlay"]

    df_all.drop(["outOfBounds", "outPeriod"], axis=1, inplace=True)

    # save to parquet file
    io.write_data(df_all, "tracking_data", league=str(game), data_folder="metrica_data")


if __name__ == "__main__":
    cleanse_tracking_data(1)
