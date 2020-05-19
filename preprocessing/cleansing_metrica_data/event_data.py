# -*- coding: utf-8 -*-

import logging

import numpy as np
import pandas as pd

import helper.io as io

logging.basicConfig(level=logging.DEBUG)


def set_positions(df_events, reverse, field_length=105, field_width=68):
    """
    Helper function to compute the positions in meters. Notice that the position is always considered from the
    perspective of the team having the event.
    :param df_events: (pd.DataFrame) Data frame with all events
    :param reverse: (bool) If True, the away team is playing left to right in the first half
    :param field_length: (int) Length of the field in meters
    :param field_width: (int) Width of the field in meters
    :return:
    """

    pos_cols = ["xPosStart", "yPosStart", "xPosEnd", "yPosEnd"]

    for col in pos_cols:

        # make sure that event data always has positions on the pitch
        df_events[col].clip(lower=0, upper=1, inplace=True)

        # if home team starts from right to left, change it to left to right
        if reverse:
            df_events[col] = 1 - df_events[col]

        # make sure that the teams play into the same direction in both half times
        df_events[col] = np.where(
            df_events["period"] == 2, 1 - df_events[col], df_events[col]
        )

        # make sure we always look at the field from each teams perspective
        df_events[col] = np.where(
            df_events["team"] == "Away", 1 - df_events[col], df_events[col]
        )

    # get the position in meters
    df_events["xPosStart"] *= field_length
    df_events["xPosEnd"] *= field_length
    df_events["yPosStart"] *= field_width
    df_events["yPosEnd"] *= field_width

    return df_events


def add_player_positions(df):
    """
    Helper function to add the player positions (e.g. GK, MD, ...) to each player
    :param df: (pd.DataFrame) Data frame with event data
    :return: pd.DataFrame containing event data and the player position
    """

    # read the position for all players
    df_positions = io.read_data(
        "player_positions", sep=";", data_folder="raw_data_metrica"
    )

    df = pd.merge(df, df_positions, on=["matchId", "playerId"])
    return df


def identify_pass(row):
    """
    Identify passes from the data
    """
    if type(row["subEventName"]) == float:
        return row["eventName"]
    elif (
        row["eventName"] == "Ball lost"
        and "interception" in row["subEventName"].lower()
    ):
        return "Pass"
    else:
        return row["eventName"]


def compute_wyscout_columns(df_events, game):
    """
    Function computes all columns that are needed in order to fit with the Wyscout format
    :param df_events: (pd.DataFrame) Data frame containing all event data
    :param game: (int) Id of the game
    :return: pd.DataFrame containing all relevant columns needed for the Wyscout format
    """

    # extract the playerId
    df_events["playerId"] = df_events["from"].map(lambda x: int(x[6:]))
    df_events["toPlayerId"] = np.nan
    df_events.loc[df_events["to"].notnull(), "toPlayerId"] = df_events.loc[
        df_events["to"].notnull(), "to"
    ].map(lambda x: int(x[6:]))

    # set the teamId - home team is always 1, away team always 2
    df_events["teamId"] = np.where(df_events["team"] == "Home", 1, 2)

    # make sure that eventName and subEventName are written lower case
    df_events["eventName"] = df_events["type"].map(lambda x: x[0] + x[1:].lower())
    df_events["subEventName"] = df_events["subtype"].map(lambda x: x[0] + x[1:].lower())

    # add a columns indicating that a pass was accurate
    df_events["accurate"] = 1 * (df_events["eventName"] == "Pass")

    # compute the event name
    df_events["eventName"] = df_events.apply(identify_pass, axis=1)
    df_events["eventName"] = np.where(
        df_events["eventName"] == "Challenge", "Duel", df_events["eventName"]
    )

    # bring the match period to the same format
    df_events["matchPeriod"] = df_events["period"].map(lambda x: str(x) + "H")

    # set a general id and a match id
    df_events["id"] = np.arange(len(df_events))
    df_events["matchId"] = game

    # get the position (e.g. GK) for all players
    df_events = add_player_positions(df_events)

    # set home and away team
    df_events["homeTeamId"] = 1
    df_events["awayTeamId"] = 2

    # mark all passes that resulted from a set piece
    df_set = df_events[df_events["eventName"] == "Set piece"][
        ["startFrame", "eventName", "subEventName"]
    ].copy()
    df_set.columns = ["startFrame", "eventNameSet", "subEventNameSet"]

    df_events = pd.merge(df_events, df_set, how="left", on="startFrame")
    df_events["eventName"] = np.where(
        df_events["eventNameSet"].isnull(),
        df_events["eventName"],
        df_events["eventNameSet"],
    )
    df_events["subEventName"] = np.where(
        df_events["subEventNameSet"].isnull(),
        df_events["subEventName"],
        df_events["subEventNameSet"],
    )

    return df_events


def cleanse_metrica_event_data(game, reverse):
    """
    Function to clean the Metrica event data. Notice that quite a lot of the code is needed to make the Metrica data
    compatible with the Wyscout format
    :param game: (int) GameId
    :param reverse: (bool) If True, the away team is playing left to right in the first half
    :return: None
    """

    logging.info(f"Cleansing metrica event data for game {game}")

    df_events = io.read_data(
        "event_data", league=str(game), sep=",", data_folder="raw_data_metrica"
    )

    # rename columns to camelStyle
    df_events.columns = [
        "team",
        "type",
        "subtype",
        "period",
        "startFrame",
        "startTime",
        "endFrame",
        "endTime",
        "from",
        "to",
        "xPosStart",
        "yPosStart",
        "xPosEnd",
        "yPosEnd",
    ]

    # make sure that the position is in meters and events are always from the perspective of the
    # team having the event
    df_events = set_positions(df_events, reverse)

    # make sure that the end frame is always at least the start frame
    df_events["endFrame"] = df_events[["startFrame", "endFrame"]].max(axis=1)
    df_events["endTime"] = df_events[["startTime", "endTime"]].max(axis=1)

    df_events["subtype"].fillna("  ", inplace=True)

    # identify goals and own goals
    df_events["goal"] = 1 * (
        df_events.apply(
            lambda row: row["type"] == "SHOT" and "-GOAL" in row["subtype"], axis=1
        )
    )
    df_events["ownGoal"] = 1 * (
        df_events.apply(
            lambda row: row["type"] == "BALL OUT" and "-GOAL" in row["subtype"], axis=1
        )
    )

    df_events = compute_wyscout_columns(df_events, game)

    df_events.sort_values(["startFrame", "endFrame"], inplace=True)
    io.write_data(df_events, "event_data", league=str(game), data_folder="metrica_data")


if __name__ == "__main__":
    cleanse_metrica_event_data(1, False)
