# -*- coding: utf-8 -*-

import logging

import numpy as np

import helper.io as io

logging.basicConfig(level=logging.DEBUG)


def build_formation_data(game):
    """
    Function builds the formation data, i.e. which player played for how long, based on the event data
    :param game: (int) GameId
    :return: None
    """

    logging.info(f"Building formation data for game {game}")

    df_track = io.read_tracking_data(game=game, clean=True)

    # get the first and last time the player was seen in the tracking data
    df = (
        df_track.groupby("playerId")
        .agg(minTime=("time", "min"), maxTime=("time", "max"), team=("team", "min"))
        .reset_index()
    )

    # convert seconds to minutes
    df["minuteStart"] = (df["minTime"] / 60).clip(lower=0).astype(int)
    df["minuteEnd"] = (df["maxTime"] / 60).clip(upper=90).astype(int)

    # fill other relevant columns
    df = df[df["playerId"] != -1].copy()
    df["matchId"] = game
    df["teamId"] = np.where(df["team"] == "Home", 1, 2)
    df["lineup"] = 1 * (df["minuteStart"] == 0)
    df["substituteIn"] = 1 * (df["minuteStart"] > 0)
    df["substituteOut"] = 1 * (df["maxTime"] < df["maxTime"].max())
    df["minutesPlayed"] = df["minuteEnd"] - df["minuteStart"]

    # make sure it is the same format as the wyscout data
    cols = [
        "playerId",
        "lineup",
        "matchId",
        "teamId",
        "substituteIn",
        "substituteOut",
        "minuteStart",
        "minuteEnd",
        "minutesPlayed",
    ]

    df = df[cols].copy()

    # save to parquet file
    io.write_data(df, "formation_data", league=str(game), data_folder="metrica_data")
